//! Client for `TypeSafe`'s Jev decision model (`POST /v1/systemone`).
//!
//! Jev takes a `state` plus typed questions and returns typed answers with
//! calibrated probabilities. There is no Rust SDK, so we speak HTTP directly.

use std::collections::BTreeMap;
use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::http::HttpTransport;
use crate::config::JevConfig;
use crate::error::{AppError, Result};

const SERVICE: &str = "jev";

/// One typed question. Field names mirror the Jev wire format.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Pick one label; `criteria` maps label -> description.
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
    /// Position on an ordered scale; `criteria` lists levels low -> high.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    /// Probability that a statement is true; optional `yes`/`no` clarification.
    Noul {
        instructions: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, String>>,
    },
}

/// A request: shared state + named questions (names never reach the model).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JevRequest {
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
}

/// One typed answer.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    Choice {
        choice: String,
        confidence: f64,
        probabilities: BTreeMap<String, f64>,
    },
    Score {
        score: f64,
        confidence: f64,
        /// Keyed by level index as a string ("0", "1", ...).
        probabilities: BTreeMap<String, f64>,
    },
    Noul {
        noul: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct JevResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
}

impl JevResponse {
    fn get(&self, id: &str) -> Result<&Answer> {
        self.answers.get(id).ok_or_else(|| AppError::BadResponse {
            service: SERVICE,
            message: format!("missing answer '{id}'"),
        })
    }

    /// # Errors
    /// Missing answer or wrong answer type.
    pub fn choice(&self, id: &str) -> Result<(&str, f64, &BTreeMap<String, f64>)> {
        match self.get(id)? {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => Ok((choice, *confidence, probabilities)),
            other => Err(wrong_type(id, "choice", other)),
        }
    }

    /// Returns `(score, confidence)`.
    ///
    /// # Errors
    /// Missing answer or wrong answer type.
    pub fn score(&self, id: &str) -> Result<(f64, f64)> {
        match self.get(id)? {
            Answer::Score {
                score, confidence, ..
            } => Ok((*score, *confidence)),
            other => Err(wrong_type(id, "score", other)),
        }
    }

    /// # Errors
    /// Missing answer or wrong answer type.
    pub fn noul(&self, id: &str) -> Result<f64> {
        match self.get(id)? {
            Answer::Noul { noul } => Ok(*noul),
            other => Err(wrong_type(id, "noul", other)),
        }
    }
}

fn wrong_type(id: &str, expected: &str, got: &Answer) -> AppError {
    AppError::BadResponse {
        service: SERVICE,
        message: format!("answer '{id}' expected {expected}, got {got:?}"),
    }
}

/// Abstraction over the decision model so pipeline logic can be tested
/// without the network.
pub trait DecisionModel: Sync {
    fn decide(&self, req: &JevRequest) -> impl Future<Output = Result<JevResponse>> + Send;
}

/// Real HTTP client.
#[derive(Debug, Clone)]
pub struct JevClient {
    http: HttpTransport,
    config: JevConfig,
}

impl JevClient {
    #[must_use]
    pub fn new(http: HttpTransport, config: JevConfig) -> Self {
        Self { http, config }
    }
}

impl DecisionModel for JevClient {
    async fn decide(&self, req: &JevRequest) -> Result<JevResponse> {
        let body = serde_json::json!({
            "model": self.config.model,
            "state": req.state,
            "questions": req.questions,
        });
        let raw = self
            .http
            .post_json(
                SERVICE,
                &self.config.base_url.join("systemone"),
                Some(&self.config.api_key),
                &body,
            )
            .await?;
        let resp: JevResponse = serde_json::from_value(raw).map_err(|e| AppError::BadResponse {
            service: SERVICE,
            message: e.to_string(),
        })?;
        for id in req.questions.keys() {
            resp.get(id)?;
        }
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn question_serializes_to_wire_format() {
        let mut q = BTreeMap::new();
        q.insert(
            "route".to_owned(),
            Question::Choice {
                instructions: "Which?".into(),
                criteria: BTreeMap::from([("a".into(), "A".into())]),
            },
        );
        q.insert(
            "sev".to_owned(),
            Question::Score {
                instructions: "How bad?".into(),
                criteria: vec!["low".into(), "high".into()],
            },
        );
        q.insert(
            "flag".to_owned(),
            Question::Noul {
                instructions: "Is it?".into(),
                criteria: None,
            },
        );
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["route"]["type"], "choice");
        assert_eq!(v["route"]["criteria"]["a"], "A");
        assert_eq!(v["sev"]["type"], "score");
        assert_eq!(v["sev"]["criteria"][1], "high");
        assert_eq!(v["flag"], json!({"type": "noul", "instructions": "Is it?"}));
    }

    fn sample_response() -> JevResponse {
        serde_json::from_value(json!({
            "model": "jev-1.13.0",
            "answers": {
                "expert": {"type": "choice", "choice": "medical", "confidence": 0.83,
                           "probabilities": {"medical": 0.89, "lifestyle": 0.11}},
                "sev": {"type": "score", "score": 0.16, "confidence": 0.85,
                        "legend": {"0": "none", "1": "minor"},
                        "probabilities": {"0": 0.85, "1": 0.15}},
                "inj": {"type": "noul", "noul": 0.03}
            },
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap()
    }

    #[test]
    fn response_accessors() {
        let r = sample_response();
        let (c, conf, p) = r.choice("expert").unwrap();
        assert_eq!(c, "medical");
        assert!((conf - 0.83).abs() < 1e-9);
        assert_eq!(p.len(), 2);
        let (s, sc) = r.score("sev").unwrap();
        assert!((s - 0.16).abs() < 1e-9 && (sc - 0.85).abs() < 1e-9);
        assert!((r.noul("inj").unwrap() - 0.03).abs() < 1e-9);
    }

    #[test]
    fn response_accessor_errors() {
        let r = sample_response();
        assert!(r.choice("missing").is_err());
        assert!(r.choice("inj").is_err());
        assert!(r.score("expert").is_err());
        assert!(r.noul("sev").is_err());
    }
}
