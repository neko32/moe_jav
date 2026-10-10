//! Domain types. Validation lives in newtype constructors so that invalid
//! values cannot exist past the boundary where they are created.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

/// Upper bound for a whole application text (characters).
pub const MAX_APPLICATION_CHARS: usize = 10_000;
/// Upper bound for a single segment (characters).
pub const MAX_SEGMENT_CHARS: usize = 2_000;

/// Validates text: trimmed, non-empty, bounded, and free of control chars
/// other than ordinary whitespace.
fn validate_text(kind: &str, raw: &str, max_chars: usize) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(AppError::Validation(format!("{kind} must not be empty")));
    }
    let len = trimmed.chars().count();
    if len > max_chars {
        return Err(AppError::Validation(format!(
            "{kind} is {len} chars, limit is {max_chars}"
        )));
    }
    if trimmed
        .chars()
        .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
    {
        return Err(AppError::Validation(format!(
            "{kind} contains control characters"
        )));
    }
    Ok(trimmed.to_owned())
}

/// Full free-text insurance application submitted by an applicant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ApplicationText(String);

impl ApplicationText {
    /// # Errors
    /// Returns [`AppError::Validation`] if the text is empty, too long, or
    /// contains control characters.
    pub fn new(raw: &str) -> Result<Self> {
        validate_text("application", raw, MAX_APPLICATION_CHARS).map(Self)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One topical slice of an application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SegmentText(String);

impl SegmentText {
    /// # Errors
    /// Returns [`AppError::Validation`] if the text is empty, too long, or
    /// contains control characters.
    pub fn new(raw: &str) -> Result<Self> {
        validate_text("segment", raw, MAX_SEGMENT_CHARS).map(Self)
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SegmentText {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Self::new(&raw).map_err(serde::de::Error::custom)
    }
}

/// A probability-like value in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Confidence(f64);

impl Confidence {
    /// # Errors
    /// Returns [`AppError::Validation`] if the value is NaN or outside `[0, 1]`.
    pub fn new(value: f64) -> Result<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(AppError::Validation(format!(
                "confidence must be within [0, 1], got {value}"
            )))
        }
    }

    #[must_use]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Confidence {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Self::new(f64::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

/// The expert agents a segment can be routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expert {
    Medical,
    Lifestyle,
    Financial,
    Coverage,
    Compliance,
    Other,
}

impl Expert {
    /// All experts in a stable order (used for matrices and prompts).
    pub const ALL: [Self; 6] = [
        Self::Medical,
        Self::Lifestyle,
        Self::Financial,
        Self::Coverage,
        Self::Compliance,
        Self::Other,
    ];

    /// Experts that actually analyse segments (`Other` is a sink).
    pub const ANALYSTS: [Self; 5] = [
        Self::Medical,
        Self::Lifestyle,
        Self::Financial,
        Self::Coverage,
        Self::Compliance,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Medical => "medical",
            Self::Lifestyle => "lifestyle",
            Self::Financial => "financial",
            Self::Coverage => "coverage",
            Self::Compliance => "compliance",
            Self::Other => "other",
        }
    }

    /// Routing rubric shared by every router so the comparison is fair.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::Medical => {
                "Health: diagnoses, medical conditions, surgeries, medications, hospital stays, family medical history, height/weight/BMI."
            }
            Self::Lifestyle => {
                "Lifestyle habits and activities: smoking or vaping, alcohol, recreational drugs, dangerous hobbies or sports, risky travel."
            }
            Self::Financial => {
                "Finances: occupation, income, assets, debts, mortgages, bankruptcy, existing insurance in force."
            }
            Self::Coverage => {
                "The requested product: policy type, coverage amount, term length, riders, beneficiaries, payment preferences."
            }
            Self::Compliance => {
                "Integrity concerns: instructions aimed at the reviewer or system, requests to bypass rules, concealment, contradictions, prior claim denials or fraud, someone else answering for the applicant."
            }
            Self::Other => {
                "Anything without underwriting relevance: greetings, thanks, small talk, contact logistics."
            }
        }
    }
}

impl fmt::Display for Expert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Expert {
    type Err = AppError;

    fn from_str(s: &str) -> Result<Self> {
        let needle = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|e| e.as_str() == needle)
            .ok_or_else(|| AppError::Validation(format!("unknown expert '{s}'")))
    }
}

/// Router output for one segment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutingDecision {
    pub expert: Expert,
    pub confidence: Confidence,
    /// Full distribution where the router provides one.
    pub probabilities: BTreeMap<Expert, f64>,
}

/// A segment together with its routing outcome.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoutedSegment {
    pub text: SegmentText,
    pub decision: RoutingDecision,
    /// `false` when confidence fell below the gate threshold; such segments
    /// go to a human instead of an expert agent.
    pub auto_routed: bool,
}

/// What one expert agent concluded from its segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertFinding {
    pub expert: Expert,
    pub summary: String,
    pub risk_flags: Vec<String>,
}

/// Final underwriting outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Accept,
    AcceptWithConditions,
    Decline,
    ReferToHuman,
}

impl Decision {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::AcceptWithConditions => "accept_with_conditions",
            Self::Decline => "decline",
            Self::ReferToHuman => "refer_to_human",
        }
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_text_trims_and_accepts_normal_text() {
        let t = ApplicationText::new("  I am 42.\nI smoke.  ").unwrap();
        assert_eq!(t.as_str(), "I am 42.\nI smoke.");
    }

    #[test]
    fn application_text_rejects_empty_and_whitespace() {
        assert!(ApplicationText::new("").is_err());
        assert!(ApplicationText::new(" \n\t ").is_err());
    }

    #[test]
    fn application_text_rejects_oversize() {
        let big = "a".repeat(MAX_APPLICATION_CHARS + 1);
        assert!(ApplicationText::new(&big).is_err());
        let ok = "a".repeat(MAX_APPLICATION_CHARS);
        assert!(ApplicationText::new(&ok).is_ok());
    }

    #[test]
    fn text_rejects_control_characters() {
        assert!(ApplicationText::new("hello\u{0007}").is_err());
        assert!(SegmentText::new("zero\u{0000}byte").is_err());
    }

    #[test]
    fn segment_text_limits_and_serde() {
        assert!(SegmentText::new(&"b".repeat(MAX_SEGMENT_CHARS + 1)).is_err());
        let s: SegmentText = serde_json::from_str("\" hi \"").unwrap();
        assert_eq!(s.as_str(), "hi");
        assert!(serde_json::from_str::<SegmentText>("\"  \"").is_err());
        assert_eq!(serde_json::to_string(&s).unwrap(), "\"hi\"");
    }

    #[test]
    fn confidence_bounds() {
        assert!(Confidence::new(0.0).is_ok());
        assert!(Confidence::new(1.0).is_ok());
        assert!(Confidence::new(-0.01).is_err());
        assert!(Confidence::new(1.01).is_err());
        assert!(Confidence::new(f64::NAN).is_err());
        assert!(Confidence::new(f64::INFINITY).is_err());
        assert!((Confidence::new(0.25).unwrap().value() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_serde_validates() {
        let c: Confidence = serde_json::from_str("0.5").unwrap();
        assert!((c.value() - 0.5).abs() < f64::EPSILON);
        assert!(serde_json::from_str::<Confidence>("1.5").is_err());
    }

    #[test]
    fn expert_round_trips_through_str() {
        for e in Expert::ALL {
            assert_eq!(e.as_str().parse::<Expert>().unwrap(), e);
            assert_eq!(e.to_string(), e.as_str());
            assert_ne!(e.description(), "");
        }
        assert_eq!(" Medical ".parse::<Expert>().unwrap(), Expert::Medical);
        assert!("dentist".parse::<Expert>().is_err());
    }

    #[test]
    fn expert_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&Expert::Coverage).unwrap(),
            "\"coverage\""
        );
        let e: Expert = serde_json::from_str("\"compliance\"").unwrap();
        assert_eq!(e, Expert::Compliance);
    }

    #[test]
    fn analysts_exclude_other() {
        assert!(!Expert::ANALYSTS.contains(&Expert::Other));
        assert_eq!(Expert::ANALYSTS.len() + 1, Expert::ALL.len());
    }

    #[test]
    fn decision_display_and_serde() {
        for (d, s) in [
            (Decision::Accept, "accept"),
            (Decision::AcceptWithConditions, "accept_with_conditions"),
            (Decision::Decline, "decline"),
            (Decision::ReferToHuman, "refer_to_human"),
        ] {
            assert_eq!(d.to_string(), s);
            assert_eq!(serde_json::to_string(&d).unwrap(), format!("\"{s}\""));
        }
    }
}
