//! Command-line interface. `run` is generic over the service traits so the
//! whole command flow is testable with stubs.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::api::jev::DecisionModel;
use crate::api::llm::LlmClient;
use crate::core::dataset;
use crate::core::domain::{ApplicationText, Confidence};
use crate::core::evaluation::{self, Prediction};
use crate::core::pipeline::Pipeline;
use crate::core::report;
use crate::core::router::{Gate, JevRouter, LlmRouter};
use crate::error::{AppError, Result};

#[derive(Debug, Parser)]
#[command(
    name = "moe_jav",
    version,
    about = "Mixture-of-experts insurance underwriting POC gated by the Jev decision model"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Disable ANSI colours.
    #[arg(long, global = true)]
    pub no_color: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RouterKind {
    /// Jev `choice` question (calibrated).
    Jev,
    /// Local LLM with self-reported confidence.
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EvalMode {
    /// Route gold segments with both routers and compare.
    Routing,
    /// Full pipeline on every application.
    E2e,
    /// Both.
    All,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Underwrite one application.
    Apply(ApplyArgs),
    /// Evaluate routing against the golden dataset.
    Eval(EvalArgs),
}

#[derive(Debug, Args)]
pub struct ApplyArgs {
    /// Read the application from a file.
    #[arg(long, conflicts_with = "text", required_unless_present = "text")]
    pub file: Option<PathBuf>,
    /// Pass the application inline.
    #[arg(long)]
    pub text: Option<String>,
    #[arg(long, value_enum, default_value_t = RouterKind::Jev)]
    pub router: RouterKind,
    /// Routing confidence below this goes to a human.
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f64,
    /// Emit JSON instead of the coloured view.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long, default_value = "../data/applications.jsonl")]
    pub dataset: PathBuf,
    #[arg(long, value_enum, default_value_t = EvalMode::Routing)]
    pub mode: EvalMode,
    /// Router used for the end-to-end run.
    #[arg(long, value_enum, default_value_t = RouterKind::Jev)]
    pub router: RouterKind,
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f64,
    /// Output directory for the report artefacts.
    #[arg(long, default_value = "../docs/eval")]
    pub out: PathBuf,
}

fn json_err(e: &serde_json::Error) -> AppError {
    AppError::Validation(format!("serialisation failed: {e}"))
}

/// Executes a parsed command.
///
/// # Errors
/// Any validation, I/O or service failure.
pub async fn run<L: LlmClient, D: DecisionModel>(
    cli: Cli,
    llm: &L,
    jev: &D,
    out: &mut impl Write,
) -> Result<()> {
    match cli.command {
        Command::Apply(args) => apply(args, !cli.no_color, llm, jev, out).await,
        Command::Eval(args) => eval(args, llm, jev, out).await,
    }
}

async fn apply<L: LlmClient, D: DecisionModel>(
    args: ApplyArgs,
    color: bool,
    llm: &L,
    jev: &D,
    out: &mut impl Write,
) -> Result<()> {
    let raw = match (args.file, args.text) {
        (Some(path), _) => std::fs::read_to_string(path)?,
        (None, Some(t)) => t,
        (None, None) => return Err(AppError::Validation("provide --file or --text".into())),
    };
    let app = ApplicationText::new(&raw)?;
    let gate = Gate::new(Confidence::new(args.threshold)?);
    let report = match args.router {
        RouterKind::Jev => {
            let router = JevRouter::new(jev);
            Pipeline {
                llm,
                jev,
                router: &router,
                gate,
            }
            .run(&app)
            .await?
        }
        RouterKind::Llm => {
            let router = LlmRouter::new(llm);
            Pipeline {
                llm,
                jev,
                router: &router,
                gate,
            }
            .run(&app)
            .await?
        }
    };
    if args.json {
        let s = serde_json::to_string_pretty(&report).map_err(|e| json_err(&e))?;
        writeln!(out, "{s}")?;
    } else {
        write!(out, "{}", report::render_console(&report, color))?;
    }
    Ok(())
}

async fn eval<L: LlmClient, D: DecisionModel>(
    args: EvalArgs,
    llm: &L,
    jev: &D,
    out: &mut impl Write,
) -> Result<()> {
    let data = dataset::load(&args.dataset)?;
    let gate = Gate::new(Confidence::new(args.threshold)?);
    let out_dir = args.out;
    std::fs::create_dir_all(&out_dir)?;

    let mut routing = Vec::new();
    if matches!(args.mode, EvalMode::Routing | EvalMode::All) {
        let jev_preds = evaluation::eval_routing(&JevRouter::new(jev), &data).await?;
        let llm_preds = evaluation::eval_routing(&LlmRouter::new(llm), &data).await?;
        write_json(&out_dir.join("routing_predictions_jev.json"), &jev_preds)?;
        write_json(&out_dir.join("routing_predictions_llm.json"), &llm_preds)?;
        routing.push(("jev", metrics(&jev_preds, args.threshold)));
        routing.push(("llm", metrics(&llm_preds, args.threshold)));
    }

    let e2e = if matches!(args.mode, EvalMode::E2e | EvalMode::All) {
        let cases = match args.router {
            RouterKind::Jev => {
                evaluation::eval_e2e(llm, jev, &JevRouter::new(jev), gate, &data).await?
            }
            RouterKind::Llm => {
                evaluation::eval_e2e(llm, jev, &LlmRouter::new(llm), gate, &data).await?
            }
        };
        write_json(&out_dir.join("e2e_cases.json"), &cases)?;
        Some((router_name(args.router), evaluation::e2e_metrics(&cases)))
    } else {
        None
    };

    let refs: Vec<(&str, &evaluation::RoutingMetrics)> =
        routing.iter().map(|(n, m)| (*n, m)).collect();
    let md = report::render_eval_markdown(
        &args.dataset.display().to_string(),
        &refs,
        e2e.as_ref().map(|(n, m)| (*n, m)),
    );
    std::fs::write(out_dir.join("eval_report.md"), &md)?;
    if !refs.is_empty() {
        std::fs::write(
            out_dir.join("reliability.svg"),
            report::render_reliability_svg(&refs),
        )?;
    }
    write!(out, "{md}")?;
    writeln!(
        out,
        "
Artefacts written to {}",
        out_dir.display()
    )?;
    Ok(())
}

fn router_name(kind: RouterKind) -> &'static str {
    match kind {
        RouterKind::Jev => "jev",
        RouterKind::Llm => "llm",
    }
}

fn metrics(preds: &[Prediction], threshold: f64) -> evaluation::RoutingMetrics {
    evaluation::routing_metrics(preds, threshold)
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let s = serde_json::to_string_pretty(value).map_err(|e| json_err(&e))?;
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::jev::{JevRequest, Question};
    use crate::core::testing::{FnJev, FnLlm};
    use serde_json::json;

    fn llm() -> FnLlm<impl Fn(&str, &str) -> Result<serde_json::Value> + Sync> {
        FnLlm(|system: &str, user: &str| {
            if system.starts_with("You split") {
                Ok(json!({"segments": [{"text": "Hi."}, {"text": "I smoke."}]}))
            } else if system.starts_with("You route") {
                let e = if user.contains("smoke") {
                    "lifestyle"
                } else {
                    "other"
                };
                Ok(json!({"expert": e, "confidence": 0.9}))
            } else {
                Ok(json!({"summary": "smoker", "risk_flags": ["tobacco"]}))
            }
        })
    }

    fn jev() -> FnJev<impl Fn(&JevRequest) -> Result<serde_json::Value> + Sync> {
        FnJev(|req: &JevRequest| {
            if let Some(Question::Choice { .. }) = req.questions.get("expert") {
                let t = req.state["excerpt"].as_str().unwrap();
                let c = if t.contains("smoke") {
                    "lifestyle"
                } else {
                    "other"
                };
                Ok(
                    json!({"model": "j", "answers": {"expert": {"type": "choice", "choice": c, "confidence": 0.8, "probabilities": {c: 0.8}}}}),
                )
            } else {
                Ok(json!({"model": "j", "answers": {
                    "medical_risk": {"type": "score", "score": 0.0, "confidence": 0.9, "probabilities": {}},
                    "lifestyle_risk": {"type": "score", "score": 2.0, "confidence": 0.9, "probabilities": {}},
                    "coverage_proportionate": {"type": "noul", "noul": 0.9},
                    "manipulation": {"type": "noul", "noul": 0.0},
                    "concealment": {"type": "noul", "noul": 0.0},
                    "sufficient_information": {"type": "noul", "noul": 0.9}
                }}))
            }
        })
    }

    async fn exec(args: &[&str]) -> Result<String> {
        let cli = Cli::try_parse_from(std::iter::once("moe_jav").chain(args.iter().copied()))
            .map_err(|e| AppError::Validation(e.to_string()))?;
        let mut buf = Vec::new();
        run(cli, &llm(), &jev(), &mut buf).await?;
        Ok(String::from_utf8(buf).unwrap())
    }

    #[tokio::test]
    async fn apply_text_console_both_routers() {
        let out = exec(&["--no-color", "apply", "--text", "Hi. I smoke."])
            .await
            .unwrap();
        assert!(out.contains("router: jev"));
        assert!(out.contains("ACCEPT_WITH_CONDITIONS"));
        let out = exec(&[
            "--no-color",
            "apply",
            "--text",
            "Hi. I smoke.",
            "--router",
            "llm",
        ])
        .await
        .unwrap();
        assert!(out.contains("router: llm"));
    }

    #[tokio::test]
    async fn apply_file_json() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "Hi. I smoke.").unwrap();
        let out = exec(&["apply", "--file", p.to_str().unwrap(), "--json"])
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["underwriting"]["decision"], "accept_with_conditions");
    }

    #[tokio::test]
    async fn apply_validation_errors() {
        assert!(exec(&["apply"]).await.is_err());
        assert!(exec(&["apply", "--text", "   "]).await.is_err());
        assert!(
            exec(&["apply", "--text", "x", "--threshold", "2"])
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn eval_all_writes_artefacts() {
        let dir = tempfile::tempdir().unwrap();
        let ds = dir.path().join("d.jsonl");
        std::fs::write(
            &ds,
            r#"{"id":"a","text":"Hi. I smoke.","segments":[{"text":"Hi.","expert":"other"},{"text":"I smoke.","expert":"lifestyle"}],"expected_decision":"accept_with_conditions"}"#,
        )
        .unwrap();
        let out_dir = dir.path().join("out");
        for router in ["jev", "llm"] {
            let out = exec(&[
                "eval",
                "--dataset",
                ds.to_str().unwrap(),
                "--mode",
                "all",
                "--router",
                router,
                "--out",
                out_dir.to_str().unwrap(),
            ])
            .await
            .unwrap();
            assert!(out.contains("Gate comparison"));
            assert!(out.contains(&format!("End-to-end pipeline (router: {router})")));
        }
        for f in [
            "eval_report.md",
            "reliability.svg",
            "routing_predictions_jev.json",
            "routing_predictions_llm.json",
            "e2e_cases.json",
        ] {
            assert!(out_dir.join(f).exists(), "{f}");
        }
    }

    #[tokio::test]
    async fn eval_e2e_only_skips_svg() {
        let dir = tempfile::tempdir().unwrap();
        let ds = dir.path().join("d.jsonl");
        std::fs::write(
            &ds,
            r#"{"id":"a","text":"Hi. I smoke.","segments":[{"text":"I smoke.","expert":"lifestyle"}]}"#,
        )
        .unwrap();
        let out_dir = dir.path().join("o");
        exec(&[
            "eval",
            "--dataset",
            ds.to_str().unwrap(),
            "--mode",
            "e2e",
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .await
        .unwrap();
        assert!(!out_dir.join("reliability.svg").exists());
        assert!(exec(&["eval", "--dataset", "nope.jsonl"]).await.is_err());
    }
}
