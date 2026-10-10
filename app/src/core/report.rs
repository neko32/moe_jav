//! Human-readable renderings: coloured console output for `apply`, and a
//! Markdown report plus SVG reliability diagram for `eval`.

use std::fmt::Write as _;

use crate::core::domain::{Decision, Expert};
use crate::core::evaluation::{E2eMetrics, RoutingMetrics};
use crate::core::pipeline::PipelineReport;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

fn expert_color(e: Expert) -> &'static str {
    match e {
        Expert::Medical => "\x1b[31m",
        Expert::Lifestyle => "\x1b[33m",
        Expert::Financial => "\x1b[32m",
        Expert::Coverage => "\x1b[36m",
        Expert::Compliance => "\x1b[35m",
        Expert::Other => "\x1b[37m",
    }
}

fn decision_color(d: Decision) -> &'static str {
    match d {
        Decision::Accept => "\x1b[32m",
        Decision::AcceptWithConditions => "\x1b[33m",
        Decision::Decline => "\x1b[31m",
        Decision::ReferToHuman => "\x1b[35m",
    }
}

struct Paint(bool);

impl Paint {
    fn wrap(&self, code: &str, text: &str) -> String {
        if self.0 {
            format!("{code}{text}{RESET}")
        } else {
            text.to_owned()
        }
    }
}

/// Ten-cell bar for a value in `[0, 1]`.
#[must_use]
pub fn bar(value: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let filled = (value.clamp(0.0, 1.0) * 10.0).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled))
}

/// Console rendering of a single pipeline run.
#[must_use]
pub fn render_console(r: &PipelineReport, color: bool) -> String {
    let p = Paint(color);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{} (router: {}, gate ≥ {:.2})\n",
        p.wrap(BOLD, "Segments"),
        r.router,
        r.gate_threshold
    );
    for seg in &r.segments {
        let e = seg.decision.expert;
        let tag = p.wrap(expert_color(e), &format!("[{:<10}]", e.as_str()));
        let conf = seg.decision.confidence.value();
        let gate = if seg.auto_routed {
            String::new()
        } else {
            p.wrap(decision_color(Decision::ReferToHuman), " → HUMAN")
        };
        let _ = writeln!(
            s,
            "  {tag} {} {conf:.2}{gate}\n      {}",
            bar(conf),
            p.wrap(DIM, seg.text.as_str())
        );
    }

    let _ = writeln!(s, "\n{}", p.wrap(BOLD, "Expert findings"));
    if r.findings.is_empty() {
        let _ = writeln!(s, "  (no expert activated)");
    }
    for f in &r.findings {
        let _ = writeln!(
            s,
            "  {} {}",
            p.wrap(expert_color(f.expert), &format!("[{}]", f.expert)),
            f.summary
        );
        for flag in &f.risk_flags {
            let _ = writeln!(s, "      ⚑ {flag}");
        }
    }

    let u = &r.underwriting;
    let sig = &u.signals;
    let _ = writeln!(s, "\n{}", p.wrap(BOLD, "Jev signals"));
    let _ = writeln!(
        s,
        "  medical risk     {:.2}/3 (conf {:.2})\n  lifestyle risk   {:.2}/3 (conf {:.2})\n  coverage fit     p={:.2}\n  manipulation     p={:.2}\n  concealment      p={:.2}\n  sufficient info  p={:.2}\n  composite risk   {:.2}",
        sig.medical_score,
        sig.medical_confidence,
        sig.lifestyle_score,
        sig.lifestyle_confidence,
        sig.coverage_proportionate,
        sig.manipulation,
        sig.concealment,
        sig.sufficient_information,
        u.risk_score
    );
    let _ = writeln!(
        s,
        "\n{} {}",
        p.wrap(BOLD, "Decision:"),
        p.wrap(
            decision_color(u.decision),
            &u.decision.to_string().to_uppercase()
        )
    );
    for reason in &u.reasons {
        let _ = writeln!(s, "  - {reason}");
    }
    s
}

fn pct(v: f64) -> String {
    format!("{:.1}%", v * 100.0)
}

/// Threshold sweep: how many segments each gate sends to a human, and how
/// many wrong routes still slip through, as the threshold rises.
fn render_sweep(s: &mut String, routing: &[(&str, &RoutingMetrics)]) {
    let _ = writeln!(s, "### Gate threshold sweep\n");
    let _ = writeln!(
        s,
        "Each cell: segments sent to human / wrong routes that slip through.\n"
    );
    let names: Vec<&str> = routing.iter().map(|(n, _)| *n).collect();
    let _ = writeln!(s, "| τ | {} |", names.join(" | "));
    let _ = writeln!(s, "|---:|{}", "---:|".repeat(names.len()));
    let rows = routing.first().map_or(0, |(_, m)| m.sweep.len());
    for i in 0..rows {
        let cells: Vec<String> = routing
            .iter()
            .map(|(_, m)| {
                m.sweep.get(i).map_or_else(String::new, |p| {
                    format!("{} / {}", p.to_human, p.errors_missed)
                })
            })
            .collect();
        let t = routing[0].1.sweep[i].threshold;
        let _ = writeln!(s, "| {t:.2} | {} |", cells.join(" | "));
    }
    let _ = writeln!(s);
}

/// Markdown evaluation report.
#[must_use]
pub fn render_eval_markdown(
    dataset_name: &str,
    routing: &[(&str, &RoutingMetrics)],
    e2e: Option<(&str, &E2eMetrics)>,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Routing evaluation report\n");
    let _ = writeln!(s, "Dataset: `{dataset_name}`\n");

    if !routing.is_empty() {
        let _ = writeln!(s, "## Gate comparison (gold segments)\n");
        let _ = writeln!(
            s,
            "| Router | Segments | Accuracy | Macro F1 | Mean conf. | ECE ↓ | Brier ↓ | Auto-routed | Accuracy when auto-routed | Errors caught by gate | Errors missed |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (name, m) in routing {
            let _ = writeln!(
                s,
                "| {name} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} |",
                m.n,
                pct(m.accuracy),
                m.macro_f1,
                m.mean_confidence,
                m.ece,
                m.brier,
                pct(m.auto_rate),
                pct(m.selective_accuracy),
                m.errors_caught,
                m.errors_missed
            );
        }
        if let Some((_, m)) = routing.first() {
            let _ = writeln!(s, "\nGate threshold: {:.2}\n", m.gate_threshold);
        }
        render_sweep(&mut s, routing);

        for (name, m) in routing {
            let _ = writeln!(s, "### {name}: per-expert\n");
            let _ = writeln!(s, "| Expert | Precision | Recall | F1 | Support |");
            let _ = writeln!(s, "|---|---:|---:|---:|---:|");
            for (e, c) in &m.per_expert {
                let _ = writeln!(
                    s,
                    "| {e} | {:.3} | {:.3} | {:.3} | {} |",
                    c.precision, c.recall, c.f1, c.support
                );
            }
            let _ = writeln!(
                s,
                "\n### {name}: confusion matrix (rows = gold, cols = predicted)\n"
            );
            let header: Vec<&str> = Expert::ALL.iter().map(|e| e.as_str()).collect();
            let _ = writeln!(s, "| gold \\ pred | {} |", header.join(" | "));
            let _ = writeln!(s, "|---|{}", "---:|".repeat(header.len()));
            for (i, e) in Expert::ALL.iter().enumerate() {
                let row: Vec<String> = m.confusion[i].iter().map(ToString::to_string).collect();
                let _ = writeln!(s, "| **{e}** | {} |", row.join(" | "));
            }
            let _ = writeln!(s, "\n### {name}: calibration\n");
            let _ = writeln!(s, "| Confidence bin | Count | Mean conf. | Accuracy |");
            let _ = writeln!(s, "|---|---:|---:|---:|");
            for b in m.bins.iter().filter(|b| b.count > 0) {
                let _ = writeln!(
                    s,
                    "| {:.1}–{:.1} | {} | {:.3} | {:.3} |",
                    b.lower, b.upper, b.count, b.mean_confidence, b.accuracy
                );
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(s, "![Reliability diagram](reliability.svg)\n");
    }

    if let Some((name, m)) = e2e {
        let _ = writeln!(s, "## End-to-end pipeline (router: {name})\n");
        let _ = writeln!(s, "| Metric | Value |\n|---|---:|");
        let _ = writeln!(s, "| Applications | {} |", m.applications);
        let _ = writeln!(s, "| Pipeline failures | {} |", m.failures);
        let _ = writeln!(s, "| Gold segments | {} |", m.gold_segments);
        let _ = writeln!(
            s,
            "| Segment recall (aligned) | {} |",
            pct(m.segment_recall)
        );
        let _ = writeln!(
            s,
            "| Routing accuracy (end-to-end) | {} |",
            pct(m.routing_accuracy)
        );
        let _ = writeln!(
            s,
            "| Decision agreement with labels | {} |",
            pct(m.decision_agreement)
        );
        let _ = writeln!(
            s,
            "| Fairness pairs consistent | {}/{} |",
            m.fairness_consistent, m.fairness_pairs
        );
        let _ = writeln!(s, "\n| Decision | Count |\n|---|---:|");
        for (d, c) in &m.decisions {
            let _ = writeln!(s, "| {d} | {c} |");
        }
    }
    s
}

/// Reliability diagram (confidence vs accuracy) for each router, as SVG.
#[must_use]
pub fn render_reliability_svg(routing: &[(&str, &RoutingMetrics)]) -> String {
    const W: f64 = 420.0;
    const PAD: f64 = 50.0;
    const PLOT: f64 = W - 2.0 * PAD;
    const COLORS: [&str; 4] = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"];
    let x = |v: f64| PAD + v * PLOT;
    let y = |v: f64| PAD + (1.0 - v) * PLOT;

    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12"><rect width="100%" height="100%" fill="#ffffff"/>"##,
        H = W + 20.0
    );
    let _ = write!(
        s,
        r#"<text x="{}" y="24" text-anchor="middle" font-size="14" font-weight="bold">Routing reliability (confidence vs accuracy)</text>"#,
        W / 2.0
    );
    let _ = write!(
        s,
        r##"<rect x="{PAD}" y="{PAD}" width="{PLOT}" height="{PLOT}" fill="none" stroke="#999"/><line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#bbb" stroke-dasharray="4 4"/>"##,
        x(0.0),
        y(0.0),
        x(1.0),
        y(1.0)
    );
    for t in [0.0, 0.5, 1.0] {
        let _ = write!(
            s,
            r#"<text x="{}" y="{}" text-anchor="middle">{t:.1}</text><text x="{}" y="{}" text-anchor="end">{t:.1}</text>"#,
            x(t),
            y(0.0) + 16.0,
            PAD - 6.0,
            y(t) + 4.0
        );
    }
    let _ = write!(
        s,
        r#"<text x="{}" y="{}" text-anchor="middle">confidence</text><text x="14" y="{}" text-anchor="middle" transform="rotate(-90 14 {})">accuracy</text>"#,
        W / 2.0,
        y(0.0) + 34.0,
        W / 2.0,
        W / 2.0
    );
    for (i, (name, m)) in routing.iter().enumerate() {
        let color = COLORS[i % COLORS.len()];
        let pts: Vec<String> = m
            .bins
            .iter()
            .filter(|b| b.count > 0)
            .map(|b| format!("{:.1},{:.1}", x(b.mean_confidence), y(b.accuracy)))
            .collect();
        let _ = write!(
            s,
            r#"<polyline points="{}" fill="none" stroke="{color}" stroke-width="2"/>"#,
            pts.join(" ")
        );
        for b in m.bins.iter().filter(|b| b.count > 0) {
            let _ = write!(
                s,
                r#"<circle cx="{:.1}" cy="{:.1}" r="4" fill="{color}"><title>{name}: n={} conf={:.2} acc={:.2}</title></circle>"#,
                x(b.mean_confidence),
                y(b.accuracy),
                b.count,
                b.mean_confidence,
                b.accuracy
            );
        }
        #[allow(clippy::cast_precision_loss)]
        let ly = PAD + 14.0 + 16.0 * i as f64;
        let _ = write!(
            s,
            r#"<rect x="{}" y="{}" width="10" height="10" fill="{color}"/><text x="{}" y="{}">{name} (ECE {:.3})</text>"#,
            PAD + 10.0,
            ly - 9.0,
            PAD + 26.0,
            ly,
            m.ece
        );
    }
    s.push_str("</svg>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::domain::{
        Confidence, ExpertFinding, RoutedSegment, RoutingDecision, SegmentText,
    };
    use crate::core::evaluation::{Prediction, routing_metrics};
    use crate::core::underwriting::{Signals, decide};
    use std::collections::BTreeMap;

    fn report(auto: bool, findings: bool) -> PipelineReport {
        let signals = Signals {
            medical_score: 0.2,
            medical_confidence: 0.9,
            lifestyle_score: 1.1,
            lifestyle_confidence: 0.8,
            coverage_proportionate: 0.9,
            manipulation: 0.0,
            concealment: 0.0,
            sufficient_information: 0.9,
        };
        PipelineReport {
            router: "jev",
            gate_threshold: 0.5,
            segments: vec![RoutedSegment {
                text: SegmentText::new("I smoke.").unwrap(),
                decision: RoutingDecision {
                    expert: Expert::Lifestyle,
                    confidence: Confidence::new(0.86).unwrap(),
                    probabilities: BTreeMap::new(),
                },
                auto_routed: auto,
            }],
            findings: if findings {
                vec![ExpertFinding {
                    expert: Expert::Lifestyle,
                    summary: "Current smoker".into(),
                    risk_flags: vec!["tobacco".into()],
                }]
            } else {
                vec![]
            },
            underwriting: decide(signals, usize::from(!auto)),
        }
    }

    #[test]
    fn bar_rendering() {
        assert_eq!(bar(0.0), "░░░░░░░░░░");
        assert_eq!(bar(1.0), "██████████");
        assert_eq!(bar(0.86).chars().filter(|c| *c == '█').count(), 9);
        assert_eq!(bar(7.0), "██████████");
    }

    #[test]
    fn console_plain_and_colored() {
        let plain = render_console(&report(true, true), false);
        assert!(plain.contains("[lifestyle ]"));
        assert!(plain.contains("⚑ tobacco"));
        assert!(plain.contains("Decision: ACCEPT"));
        assert!(!plain.contains('\x1b'));
        let colored = render_console(&report(false, false), true);
        assert!(colored.contains("\x1b[33m"));
        assert!(colored.contains("→ HUMAN"));
        assert!(colored.contains("(no expert activated)"));
        assert!(colored.contains("REFER_TO_HUMAN"));
    }

    fn metrics() -> RoutingMetrics {
        let mk = |gold, predicted, confidence| Prediction {
            app_id: "a".into(),
            text: "t".into(),
            gold,
            predicted,
            confidence,
        };
        routing_metrics(
            &[
                mk(Expert::Medical, Expert::Medical, 0.95),
                mk(Expert::Medical, Expert::Lifestyle, 0.4),
            ],
            0.5,
        )
    }

    #[test]
    fn markdown_report_sections() {
        let m = metrics();
        let e2e = E2eMetrics {
            applications: 1,
            failures: 0,
            gold_segments: 2,
            segment_recall: 1.0,
            routing_accuracy: 0.5,
            decision_agreement: 1.0,
            decisions: BTreeMap::from([("accept".to_owned(), 1)]),
            fairness_pairs: 0,
            fairness_consistent: 0,
        };
        let md = render_eval_markdown("d.jsonl", &[("jev", &m), ("llm", &m)], Some(("jev", &e2e)));
        assert!(md.contains("| jev | 2 | 50.0% |"));
        assert!(md.contains("### llm: confusion matrix"));
        assert!(md.contains("### Gate threshold sweep"));
        assert!(md.contains("| 0.50 | 1 / 0 | 1 / 0 |"));
        assert!(md.contains("| **medical** | 1 | 1 |"));
        assert!(md.contains("## End-to-end pipeline (router: jev)"));
        assert!(md.contains("| accept | 1 |"));
        let empty = render_eval_markdown("d", &[], None);
        assert!(!empty.contains("Gate comparison"));
    }

    #[test]
    fn svg_contains_series() {
        let m = metrics();
        let svg = render_reliability_svg(&[("jev", &m), ("llm", &m)]);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert_eq!(svg.matches("<polyline").count(), 2);
        assert!(svg.contains("jev (ECE"));
    }
}
