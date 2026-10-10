//! End-to-end flow: segment -> route (gate) -> experts -> underwrite.

use serde::Serialize;

use crate::api::jev::DecisionModel;
use crate::api::llm::LlmClient;
use crate::core::domain::{ApplicationText, ExpertFinding, RoutedSegment};
use crate::core::experts::ExpertPanel;
use crate::core::router::{Gate, Router, route_all};
use crate::core::segmenter::Segmenter;
use crate::core::underwriting::{self, UnderwritingResult};
use crate::error::Result;

#[derive(Debug, Clone, Serialize)]
pub struct PipelineReport {
    pub router: &'static str,
    pub gate_threshold: f64,
    pub segments: Vec<RoutedSegment>,
    pub findings: Vec<ExpertFinding>,
    pub underwriting: UnderwritingResult,
}

pub struct Pipeline<'a, L, D, R> {
    pub llm: &'a L,
    pub jev: &'a D,
    pub router: &'a R,
    pub gate: Gate,
}

impl<L: LlmClient, D: DecisionModel, R: Router> Pipeline<'_, L, D, R> {
    /// # Errors
    /// Any stage failure.
    pub async fn run(&self, app: &ApplicationText) -> Result<PipelineReport> {
        let segments = Segmenter::new(self.llm).split(app).await?;
        tracing::info!(count = segments.len(), "segmented application");

        let routed = route_all(self.router, self.gate, &segments).await?;
        let gated = routed.iter().filter(|r| !r.auto_routed).count();
        tracing::info!(router = self.router.name(), gated, "routed segments");

        let findings = ExpertPanel::new(self.llm).analyse(&routed).await?;
        let signals = underwriting::assess(self.jev, &routed, &findings).await?;
        let underwriting = underwriting::decide(signals, gated);
        tracing::info!(decision = %underwriting.decision, "underwriting complete");

        Ok(PipelineReport {
            router: self.router.name(),
            gate_threshold: self.gate.threshold().value(),
            segments: routed,
            findings,
            underwriting,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::jev::{JevRequest, Question};
    use crate::core::domain::{Confidence, Decision, Expert};
    use crate::core::router::JevRouter;
    use crate::core::testing::{FnJev, FnLlm};
    use serde_json::json;

    pub(crate) fn fake_llm() -> FnLlm<impl Fn(&str, &str) -> Result<serde_json::Value> + Sync> {
        FnLlm(|system: &str, _user: &str| {
            if system.starts_with("You split") {
                Ok(json!({"segments": [
                    {"text": "Hello!"},
                    {"text": "I quit smoking in 2015."},
                    {"text": "I want $500k of term life."}
                ]}))
            } else {
                Ok(json!({"summary": "ok", "risk_flags": []}))
            }
        })
    }

    pub(crate) fn fake_jev(
        route_conf: f64,
    ) -> FnJev<impl Fn(&JevRequest) -> Result<serde_json::Value> + Sync> {
        FnJev(move |req: &JevRequest| {
            if let Some(Question::Choice { .. }) = req.questions.get("expert") {
                let text = req.state["excerpt"].as_str().unwrap_or_default();
                let choice = if text.contains("smoking") {
                    "lifestyle"
                } else if text.contains("term life") {
                    "coverage"
                } else {
                    "other"
                };
                Ok(json!({"model": "jev", "answers": {"expert": {
                    "type": "choice", "choice": choice, "confidence": route_conf,
                    "probabilities": {choice: route_conf}}}}))
            } else {
                Ok(json!({"model": "jev", "answers": {
                    "medical_risk": {"type": "score", "score": 0.1, "confidence": 0.9, "probabilities": {}},
                    "lifestyle_risk": {"type": "score", "score": 0.6, "confidence": 0.8, "probabilities": {}},
                    "coverage_proportionate": {"type": "noul", "noul": 0.9},
                    "manipulation": {"type": "noul", "noul": 0.01},
                    "concealment": {"type": "noul", "noul": 0.02},
                    "sufficient_information": {"type": "noul", "noul": 0.9}
                }}))
            }
        })
    }

    #[tokio::test]
    async fn runs_all_stages() {
        let llm = fake_llm();
        let jev = fake_jev(0.9);
        let router = JevRouter::new(&jev);
        let p = Pipeline {
            llm: &llm,
            jev: &jev,
            router: &router,
            gate: Gate::new(Confidence::new(0.5).unwrap()),
        };
        let app = ApplicationText::new("Hello! I quit smoking in 2015. I want $500k of term life.")
            .unwrap();
        let r = p.run(&app).await.unwrap();
        assert_eq!(r.router, "jev");
        assert_eq!(r.segments.len(), 3);
        assert_eq!(r.segments[1].decision.expert, Expert::Lifestyle);
        assert_eq!(r.findings.len(), 2, "lifestyle + coverage, not other");
        assert_eq!(r.underwriting.decision, Decision::Accept);
    }

    #[tokio::test]
    async fn low_routing_confidence_refers_to_human() {
        let llm = fake_llm();
        let jev = fake_jev(0.3);
        let router = JevRouter::new(&jev);
        let p = Pipeline {
            llm: &llm,
            jev: &jev,
            router: &router,
            gate: Gate::new(Confidence::new(0.5).unwrap()),
        };
        let app = ApplicationText::new("whatever").unwrap();
        let r = p.run(&app).await.unwrap();
        assert_eq!(r.findings, []);
        assert_eq!(r.underwriting.decision, Decision::ReferToHuman);
    }
}
