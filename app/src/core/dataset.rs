//! Golden dataset: synthetic applications with hand-labelled segments.

use std::path::Path;

use serde::Deserialize;

use crate::core::domain::{ApplicationText, Decision, Expert, SegmentText};
use crate::error::{AppError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct LabeledSegment {
    pub text: SegmentText,
    pub expert: Expert,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LabeledApplication {
    pub id: String,
    pub text: String,
    pub segments: Vec<LabeledSegment>,
    #[serde(default)]
    pub expected_decision: Option<Decision>,
    /// Free-form tags, e.g. `injection`, `ambiguous`, `pair:p1`.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl LabeledApplication {
    /// # Errors
    /// The application text fails validation.
    pub fn application(&self) -> Result<ApplicationText> {
        ApplicationText::new(&self.text)
    }

    /// Fairness pair id from a `pair:<id>` tag.
    #[must_use]
    pub fn pair(&self) -> Option<&str> {
        self.tags.iter().find_map(|t| t.strip_prefix("pair:"))
    }
}

/// Parses JSONL, validating that every labelled segment is a verbatim
/// substring of its application.
///
/// # Errors
/// [`AppError::Validation`] naming the offending line.
pub fn parse_jsonl(content: &str) -> Result<Vec<LabeledApplication>> {
    let mut out = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let err = |m: String| AppError::Validation(format!("dataset line {line_no}: {m}"));
        let app: LabeledApplication = serde_json::from_str(line).map_err(|e| err(e.to_string()))?;
        app.application().map_err(|e| err(e.to_string()))?;
        if app.segments.is_empty() {
            return Err(err("no segments".into()));
        }
        if let Some(missing) = app
            .segments
            .iter()
            .find(|s| !app.text.contains(s.text.as_str()))
        {
            return Err(err(format!(
                "segment not found verbatim: '{}'",
                missing.text.as_str()
            )));
        }
        if !ids.insert(app.id.clone()) {
            return Err(err(format!("duplicate id '{}'", app.id)));
        }
        out.push(app);
    }
    if out.is_empty() {
        return Err(AppError::Validation("dataset is empty".into()));
    }
    Ok(out)
}

/// # Errors
/// I/O or validation failures.
pub fn load(path: &Path) -> Result<Vec<LabeledApplication>> {
    parse_jsonl(&std::fs::read_to_string(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"id":"a1","text":"Hi. I smoke.","segments":[{"text":"Hi.","expert":"other"},{"text":"I smoke.","expert":"lifestyle"}],"expected_decision":"accept_with_conditions","tags":["pair:p1"]}

{"id":"a2","text":"I earn $40k.","segments":[{"text":"I earn $40k.","expert":"financial"}]}"#;

    #[test]
    fn parses_valid_jsonl() {
        let d = parse_jsonl(GOOD).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].segments[1].expert, Expert::Lifestyle);
        assert_eq!(d[0].expected_decision, Some(Decision::AcceptWithConditions));
        assert_eq!(d[0].pair(), Some("p1"));
        assert_eq!(d[1].pair(), None);
        assert!(d[1].expected_decision.is_none());
        assert_eq!(d[1].application().unwrap().as_str(), "I earn $40k.");
    }

    #[test]
    fn rejects_bad_rows() {
        for (bad, needle) in [
            (
                r#"{"id":"x","text":"abc","segments":[{"text":"zzz","expert":"other"}]}"#,
                "verbatim",
            ),
            (r#"{"id":"x","text":"abc","segments":[]}"#, "no segments"),
            (
                r#"{"id":"x","text":"  ","segments":[{"text":"a","expert":"other"}]}"#,
                "line 1",
            ),
            (
                r#"{"id":"x","text":"abc","segments":[{"text":"abc","expert":"vet"}]}"#,
                "line 1",
            ),
            ("not json", "line 1"),
            ("", "empty"),
        ] {
            let e = parse_jsonl(bad).unwrap_err().to_string();
            assert!(e.contains(needle), "{e}");
        }
    }

    #[test]
    fn rejects_duplicate_ids() {
        let row = r#"{"id":"x","text":"abc","segments":[{"text":"abc","expert":"other"}]}"#;
        let e = parse_jsonl(&format!("{row}\n{row}"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("duplicate"));
    }

    #[test]
    fn load_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("d.jsonl");
        std::fs::write(&p, GOOD).unwrap();
        assert_eq!(load(&p).unwrap().len(), 2);
        assert!(load(&dir.path().join("missing.jsonl")).is_err());
    }
}
