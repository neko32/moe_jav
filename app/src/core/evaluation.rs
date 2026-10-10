//! Did the gate pick the right expert? Metrics and evaluation runners.
//!
//! - Routing-only: route the *gold* segments, so router quality is measured
//!   without segmentation noise.
//! - End-to-end: run the full pipeline, align predicted segments to gold ones
//!   by token overlap, then score routing and the final decision.

use std::collections::{BTreeMap, BTreeSet};

use futures::{StreamExt, TryStreamExt};
use serde::Serialize;

use crate::api::jev::DecisionModel;
use crate::api::llm::LlmClient;
use crate::core::dataset::LabeledApplication;
use crate::core::domain::{Decision, Expert};
use crate::core::pipeline::{Pipeline, PipelineReport};
use crate::core::router::{Gate, Router};
use crate::error::Result;

/// Number of equal-width bins for calibration metrics.
pub const CALIBRATION_BINS: usize = 10;
/// Minimum token Jaccard for a predicted segment to count as a gold match.
pub const ALIGN_MIN_OVERLAP: f64 = 0.5;
/// Gate thresholds reported in the sweep table.
pub const SWEEP_THRESHOLDS: [f64; 7] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
/// In-flight requests during evaluation.
const EVAL_CONCURRENCY: usize = 4;

/// One routed gold segment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prediction {
    pub app_id: String,
    pub text: String,
    pub gold: Expert,
    pub predicted: Expert,
    pub confidence: f64,
}

impl Prediction {
    #[must_use]
    pub fn correct(&self) -> bool {
        self.gold == self.predicted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_confidence: f64,
    pub accuracy: f64,
}

/// Gate behaviour at one threshold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SweepPoint {
    pub threshold: f64,
    /// Segments sent to human review.
    pub to_human: usize,
    /// Wrong routes the gate let through.
    pub errors_missed: usize,
    /// Accuracy on segments the gate let through.
    pub selective_accuracy: f64,
}

/// What the gate does at `threshold`.
#[must_use]
pub fn gate_at(preds: &[Prediction], threshold: f64) -> SweepPoint {
    let auto: Vec<&Prediction> = preds.iter().filter(|p| p.confidence >= threshold).collect();
    let errors_missed = auto.iter().filter(|p| !p.correct()).count();
    SweepPoint {
        threshold,
        to_human: preds.len() - auto.len(),
        errors_missed,
        selective_accuracy: ratio(auto.len() - errors_missed, auto.len()),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutingMetrics {
    pub n: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub per_expert: BTreeMap<Expert, ClassMetrics>,
    /// `confusion[gold][predicted]`, indexed by [`Expert::ALL`] order.
    pub confusion: Vec<Vec<usize>>,
    pub mean_confidence: f64,
    /// Expected calibration error (lower is better).
    pub ece: f64,
    /// Mean squared error of top-label confidence vs correctness.
    pub brier: f64,
    pub bins: Vec<CalibrationBin>,
    pub gate_threshold: f64,
    /// Share of segments the gate let through automatically.
    pub auto_rate: f64,
    /// Accuracy on segments the gate let through.
    pub selective_accuracy: f64,
    /// Wrong routes the gate caught (sent to human).
    pub errors_caught: usize,
    /// Wrong routes the gate let through.
    pub errors_missed: usize,
    /// Gate behaviour across [`SWEEP_THRESHOLDS`].
    pub sweep: Vec<SweepPoint>,
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let r = num as f64 / den as f64;
        r
    }
}

fn index(e: Expert) -> usize {
    Expert::ALL.iter().position(|x| *x == e).unwrap_or_default()
}

#[allow(clippy::cast_precision_loss)]
fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Computes all routing metrics. An empty input yields zeros.
#[must_use]
pub fn routing_metrics(preds: &[Prediction], gate_threshold: f64) -> RoutingMetrics {
    let n = preds.len();
    let correct = preds.iter().filter(|p| p.correct()).count();

    let mut confusion = vec![vec![0usize; Expert::ALL.len()]; Expert::ALL.len()];
    for p in preds {
        confusion[index(p.gold)][index(p.predicted)] += 1;
    }

    let per_expert: BTreeMap<Expert, ClassMetrics> = Expert::ALL
        .iter()
        .map(|&e| {
            let tp = preds
                .iter()
                .filter(|p| p.gold == e && p.predicted == e)
                .count();
            let predicted = preds.iter().filter(|p| p.predicted == e).count();
            let support = preds.iter().filter(|p| p.gold == e).count();
            let precision = ratio(tp, predicted);
            let recall = ratio(tp, support);
            let f1 = if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };
            (
                e,
                ClassMetrics {
                    precision,
                    recall,
                    f1,
                    support,
                },
            )
        })
        .collect();
    let macro_f1 = mean(per_expert.values().filter(|m| m.support > 0).map(|m| m.f1));

    let bins: Vec<CalibrationBin> = (0..CALIBRATION_BINS)
        .map(|b| {
            #[allow(clippy::cast_precision_loss)]
            let (lower, upper) = (
                b as f64 / CALIBRATION_BINS as f64,
                (b + 1) as f64 / CALIBRATION_BINS as f64,
            );
            let members: Vec<&Prediction> =
                preds.iter().filter(|p| bin_of(p.confidence) == b).collect();
            CalibrationBin {
                lower,
                upper,
                count: members.len(),
                mean_confidence: mean(members.iter().map(|p| p.confidence)),
                accuracy: ratio(
                    members.iter().filter(|p| p.correct()).count(),
                    members.len(),
                ),
            }
        })
        .collect();
    let ece = bins
        .iter()
        .map(|b| ratio(b.count, n) * (b.accuracy - b.mean_confidence).abs())
        .sum();
    let brier = mean(preds.iter().map(|p| {
        let y = if p.correct() { 1.0 } else { 0.0 };
        (p.confidence - y).powi(2)
    }));

    let gate = gate_at(preds, gate_threshold);
    let errors_caught = (n - correct) - gate.errors_missed;

    RoutingMetrics {
        n,
        accuracy: ratio(correct, n),
        macro_f1,
        per_expert,
        confusion,
        mean_confidence: mean(preds.iter().map(|p| p.confidence)),
        ece,
        brier,
        bins,
        gate_threshold,
        auto_rate: ratio(n - gate.to_human, n),
        selective_accuracy: gate.selective_accuracy,
        errors_caught,
        errors_missed: gate.errors_missed,
        sweep: SWEEP_THRESHOLDS
            .iter()
            .map(|t| gate_at(preds, *t))
            .collect(),
    }
}

fn bin_of(confidence: f64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let b = (confidence * CALIBRATION_BINS as f64).floor() as usize;
    b.min(CALIBRATION_BINS - 1)
}

fn tokens(s: &str) -> BTreeSet<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '$')
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Token-set Jaccard similarity.
#[must_use]
pub fn overlap(a: &str, b: &str) -> f64 {
    let (ta, tb) = (tokens(a), tokens(b));
    let inter = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    ratio(inter, union)
}

/// For each gold segment, the index of the best-overlapping predicted
/// segment if it clears [`ALIGN_MIN_OVERLAP`].
#[must_use]
pub fn align(gold: &[&str], predicted: &[&str]) -> Vec<Option<usize>> {
    gold.iter()
        .map(|g| {
            predicted
                .iter()
                .enumerate()
                .map(|(i, p)| (i, overlap(g, p)))
                .filter(|(_, o)| *o >= ALIGN_MIN_OVERLAP)
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        })
        .collect()
}

/// Routes every gold segment with `router`.
///
/// # Errors
/// The first routing failure.
pub async fn eval_routing<R: Router>(
    router: &R,
    dataset: &[LabeledApplication],
) -> Result<Vec<Prediction>> {
    let jobs = dataset
        .iter()
        .flat_map(|a| a.segments.iter().map(move |s| (a, s)));
    futures::stream::iter(jobs)
        .map(|(app, seg)| async move {
            let d = router.route(&seg.text).await?;
            Ok::<_, crate::error::AppError>(Prediction {
                app_id: app.id.clone(),
                text: seg.text.as_str().to_owned(),
                gold: seg.expert,
                predicted: d.expert,
                confidence: d.confidence.value(),
            })
        })
        .buffered(EVAL_CONCURRENCY)
        .try_collect()
        .await
}

/// One application's end-to-end outcome.
#[derive(Debug, Clone, Serialize)]
pub struct E2eCase {
    pub app_id: String,
    pub expected: Option<Decision>,
    /// `None` when the pipeline failed for this application.
    pub decision: Option<Decision>,
    pub gold_segments: usize,
    pub predicted_segments: usize,
    /// Gold segments with an aligned predicted segment.
    pub matched: usize,
    /// Matched segments routed to the gold expert.
    pub routed_correctly: usize,
    pub pair: Option<String>,
    pub error: Option<String>,
    pub report: Option<PipelineReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct E2eMetrics {
    pub applications: usize,
    /// Applications where the pipeline failed (counted as wrong everywhere).
    pub failures: usize,
    pub gold_segments: usize,
    pub segment_recall: f64,
    /// Correctly routed / all gold segments (misses count as wrong).
    pub routing_accuracy: f64,
    pub decision_agreement: f64,
    pub decisions: BTreeMap<String, usize>,
    pub fairness_pairs: usize,
    pub fairness_consistent: usize,
}

/// Runs the full pipeline on every application (sequentially: the local LLM
/// is the bottleneck anyway). A failing application is recorded, not fatal,
/// so one bad generation cannot sink a long evaluation run.
///
/// # Errors
/// Only dataset validation failures.
pub async fn eval_e2e<L: LlmClient, D: DecisionModel, R: Router>(
    llm: &L,
    jev: &D,
    router: &R,
    gate: Gate,
    dataset: &[LabeledApplication],
) -> Result<Vec<E2eCase>> {
    let pipeline = Pipeline {
        llm,
        jev,
        router,
        gate,
    };
    let mut cases = Vec::with_capacity(dataset.len());
    for (i, app) in dataset.iter().enumerate() {
        tracing::info!(id = %app.id, progress = format!("{}/{}", i + 1, dataset.len()), "e2e");
        let mut case = E2eCase {
            app_id: app.id.clone(),
            expected: app.expected_decision,
            decision: None,
            gold_segments: app.segments.len(),
            predicted_segments: 0,
            matched: 0,
            routed_correctly: 0,
            pair: app.pair().map(str::to_owned),
            error: None,
            report: None,
        };
        match pipeline.run(&app.application()?).await {
            Ok(report) => {
                let gold: Vec<&str> = app.segments.iter().map(|s| s.text.as_str()).collect();
                let pred: Vec<&str> = report.segments.iter().map(|s| s.text.as_str()).collect();
                let alignment = align(&gold, &pred);
                case.predicted_segments = pred.len();
                case.matched = alignment.iter().flatten().count();
                case.routed_correctly = alignment
                    .iter()
                    .zip(&app.segments)
                    .filter(|(a, g)| {
                        a.is_some_and(|i| report.segments[i].decision.expert == g.expert)
                    })
                    .count();
                case.decision = Some(report.underwriting.decision);
                case.report = Some(report);
            }
            Err(e) => {
                tracing::error!(id = %app.id, error = %e, "e2e case failed");
                case.error = Some(e.to_string());
            }
        }
        cases.push(case);
    }
    Ok(cases)
}

#[must_use]
pub fn e2e_metrics(cases: &[E2eCase]) -> E2eMetrics {
    let gold: usize = cases.iter().map(|c| c.gold_segments).sum();
    let matched: usize = cases.iter().map(|c| c.matched).sum();
    let routed: usize = cases.iter().map(|c| c.routed_correctly).sum();
    let labelled: Vec<&E2eCase> = cases.iter().filter(|c| c.expected.is_some()).collect();
    let agree = labelled
        .iter()
        .filter(|c| c.decision.is_some() && c.expected == c.decision)
        .count();
    let mut decisions = BTreeMap::new();
    for c in cases {
        let key = c
            .decision
            .map_or_else(|| "error".to_owned(), |d| d.to_string());
        *decisions.entry(key).or_insert(0) += 1;
    }
    let mut pairs: BTreeMap<&str, Vec<Option<Decision>>> = BTreeMap::new();
    for c in cases {
        if let Some(p) = &c.pair {
            pairs.entry(p).or_default().push(c.decision);
        }
    }
    let pairs: Vec<&Vec<Option<Decision>>> = pairs.values().filter(|d| d.len() > 1).collect();
    E2eMetrics {
        applications: cases.len(),
        failures: cases.iter().filter(|c| c.decision.is_none()).count(),
        gold_segments: gold,
        segment_recall: ratio(matched, gold),
        routing_accuracy: ratio(routed, gold),
        decision_agreement: ratio(agree, labelled.len()),
        decisions,
        fairness_pairs: pairs.len(),
        fairness_consistent: pairs
            .iter()
            .filter(|d| d.iter().all(Option::is_some) && d.windows(2).all(|w| w[0] == w[1]))
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::jev::{JevRequest, Question};
    use crate::core::dataset::parse_jsonl;
    use crate::core::domain::Confidence;
    use crate::core::router::JevRouter;
    use crate::core::testing::{FnJev, FnLlm};
    use serde_json::json;

    fn p(gold: Expert, predicted: Expert, confidence: f64) -> Prediction {
        Prediction {
            app_id: "a".into(),
            text: "t".into(),
            gold,
            predicted,
            confidence,
        }
    }

    #[test]
    fn perfect_routing() {
        let preds = vec![
            p(Expert::Medical, Expert::Medical, 1.0),
            p(Expert::Financial, Expert::Financial, 1.0),
        ];
        let m = routing_metrics(&preds, 0.5);
        assert_eq!(m.n, 2);
        assert!((m.accuracy - 1.0).abs() < 1e-9);
        assert!((m.macro_f1 - 1.0).abs() < 1e-9);
        assert!(m.ece.abs() < 1e-9);
        assert!(m.brier.abs() < 1e-9);
        assert!((m.auto_rate - 1.0).abs() < 1e-9);
        assert_eq!(m.confusion[0][0], 1);
        assert_eq!(
            m.bins[CALIBRATION_BINS - 1].count,
            2,
            "1.0 falls in top bin"
        );
    }

    #[test]
    fn mixed_routing_metrics() {
        let preds = vec![
            p(Expert::Medical, Expert::Medical, 0.9),
            p(Expert::Medical, Expert::Lifestyle, 0.8),
            p(Expert::Lifestyle, Expert::Lifestyle, 0.3),
            p(Expert::Other, Expert::Medical, 0.2),
        ];
        let m = routing_metrics(&preds, 0.5);
        assert!((m.accuracy - 0.5).abs() < 1e-9);
        let med = m.per_expert[&Expert::Medical];
        assert!((med.precision - 0.5).abs() < 1e-9);
        assert!((med.recall - 0.5).abs() < 1e-9);
        assert_eq!(med.support, 2);
        assert!(m.per_expert[&Expert::Other].f1.abs() < 1e-9);
        assert_eq!(m.confusion[index(Expert::Other)][index(Expert::Medical)], 1);
        assert!((m.auto_rate - 0.5).abs() < 1e-9);
        assert!((m.selective_accuracy - 0.5).abs() < 1e-9);
        assert_eq!(m.errors_missed, 1);
        assert_eq!(m.errors_caught, 1);
        assert!(m.ece > 0.0 && m.brier > 0.0);
    }

    #[test]
    fn sweep_shows_errors_caught_as_threshold_rises() {
        let preds = vec![
            p(Expert::Medical, Expert::Medical, 0.99),
            p(Expert::Other, Expert::Coverage, 0.55),
            p(Expert::Financial, Expert::Coverage, 0.97),
        ];
        let m = routing_metrics(&preds, 0.5);
        assert_eq!(m.sweep.len(), SWEEP_THRESHOLDS.len());
        let at = |t: f64| {
            *m.sweep
                .iter()
                .find(|s| (s.threshold - t).abs() < 1e-9)
                .unwrap()
        };
        assert_eq!((at(0.5).to_human, at(0.5).errors_missed), (0, 2));
        assert_eq!((at(0.6).to_human, at(0.6).errors_missed), (1, 1));
        assert_eq!((at(0.99).to_human, at(0.99).errors_missed), (2, 0));
        assert!((at(0.99).selective_accuracy - 1.0).abs() < 1e-9);
        let none = gate_at(&preds, 1.0);
        assert_eq!(none.to_human, 3);
        assert!(none.selective_accuracy.abs() < 1e-9);
    }

    #[test]
    fn empty_predictions_are_zero() {
        let m = routing_metrics(&[], 0.5);
        assert_eq!(m.n, 0);
        assert!(m.accuracy.abs() < 1e-9);
        assert!(m.ece.abs() < 1e-9);
    }

    #[test]
    fn overlap_and_alignment() {
        assert!((overlap("I smoke daily", "i SMOKE daily.") - 1.0).abs() < 1e-9);
        assert!(overlap("", "").abs() < 1e-9);
        let gold = ["I quit smoking in 2015.", "I earn $90,000 a year."];
        let pred = ["I earn $90,000 a year", "Hello", "quit smoking in 2015"];
        assert_eq!(align(&gold, &pred), vec![Some(2), Some(0)]);
        assert_eq!(align(&["unrelated words"], &pred), vec![None]);
    }

    fn dataset() -> Vec<LabeledApplication> {
        parse_jsonl(concat!(
            r#"{"id":"a1","text":"Hello! I quit smoking in 2015. I want $500k of term life.","segments":[{"text":"Hello!","expert":"other"},{"text":"I quit smoking in 2015.","expert":"lifestyle"},{"text":"I want $500k of term life.","expert":"coverage"}],"expected_decision":"accept","tags":["pair:p1"]}"#,
            "\n",
            r#"{"id":"a2","text":"Hello! I quit smoking in 2015. I want $500k of term life. (she)","segments":[{"text":"Hello!","expert":"other"},{"text":"I quit smoking in 2015.","expert":"medical"}],"expected_decision":"decline","tags":["pair:p1"]}"#
        ))
        .unwrap()
    }

    fn jev() -> FnJev<impl Fn(&JevRequest) -> Result<serde_json::Value> + Sync> {
        FnJev(|req: &JevRequest| {
            if let Some(Question::Choice { .. }) = req.questions.get("expert") {
                let text = req.state["excerpt"].as_str().unwrap();
                let choice = if text.contains("smoking") {
                    "lifestyle"
                } else if text.contains("term") {
                    "coverage"
                } else {
                    "other"
                };
                Ok(
                    json!({"model": "j", "answers": {"expert": {"type": "choice", "choice": choice, "confidence": 0.9, "probabilities": {}}}}),
                )
            } else {
                Ok(json!({"model": "j", "answers": {
                    "medical_risk": {"type": "score", "score": 0.0, "confidence": 0.9, "probabilities": {}},
                    "lifestyle_risk": {"type": "score", "score": 0.3, "confidence": 0.9, "probabilities": {}},
                    "coverage_proportionate": {"type": "noul", "noul": 0.9},
                    "manipulation": {"type": "noul", "noul": 0.0},
                    "concealment": {"type": "noul", "noul": 0.0},
                    "sufficient_information": {"type": "noul", "noul": 0.9}
                }}))
            }
        })
    }

    #[tokio::test]
    async fn routing_eval_uses_gold_segments() {
        let j = jev();
        let preds = eval_routing(&JevRouter::new(&j), &dataset()).await.unwrap();
        assert_eq!(preds.len(), 5);
        assert_eq!(preds.iter().filter(|p| p.correct()).count(), 4);
        assert_eq!(preds[3].app_id, "a2");
    }

    #[tokio::test]
    async fn e2e_eval_aligns_and_scores() {
        let llm = FnLlm(|system: &str, _: &str| {
            if system.starts_with("You split") {
                Ok(
                    json!({"segments": [{"text": "Hello!"}, {"text": "I quit smoking in 2015."}, {"text": "I want $500k of term life."}]}),
                )
            } else {
                Ok(json!({"summary": "s", "risk_flags": []}))
            }
        });
        let j = jev();
        let gate = Gate::new(Confidence::new(0.5).unwrap());
        let cases = eval_e2e(&llm, &j, &JevRouter::new(&j), gate, &dataset())
            .await
            .unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].matched, 3);
        assert_eq!(cases[0].routed_correctly, 3);
        assert_eq!(
            cases[1].routed_correctly, 1,
            "gold says medical, router says lifestyle"
        );
        let m = e2e_metrics(&cases);
        assert_eq!(m.gold_segments, 5);
        assert!((m.segment_recall - 1.0).abs() < 1e-9);
        assert!((m.routing_accuracy - 0.8).abs() < 1e-9);
        assert!((m.decision_agreement - 0.5).abs() < 1e-9);
        assert_eq!(m.decisions["accept"], 2);
        assert_eq!(m.fairness_pairs, 1);
        assert_eq!(m.fairness_consistent, 1);
        assert_eq!(m.failures, 0);
    }

    #[tokio::test]
    async fn e2e_failure_is_recorded_and_run_continues() {
        let llm = FnLlm(|system: &str, _: &str| {
            if system.starts_with("You split") {
                Err(crate::error::AppError::Http {
                    service: "llm",
                    message: "timed out".into(),
                })
            } else {
                Ok(json!({"summary": "s", "risk_flags": []}))
            }
        });
        let j = jev();
        let gate = Gate::new(Confidence::new(0.5).unwrap());
        let cases = eval_e2e(&llm, &j, &JevRouter::new(&j), gate, &dataset())
            .await
            .unwrap();
        assert_eq!(cases.len(), 2, "both applications attempted");
        assert!(
            cases
                .iter()
                .all(|c| c.decision.is_none() && c.report.is_none())
        );
        assert!(cases[0].error.as_deref().unwrap().contains("timed out"));
        let m = e2e_metrics(&cases);
        assert_eq!(m.failures, 2);
        assert_eq!(m.decisions["error"], 2);
        assert!(m.routing_accuracy.abs() < 1e-9);
        assert!(m.decision_agreement.abs() < 1e-9);
        assert_eq!(m.fairness_consistent, 0, "failed pairs are not consistent");
    }
}
