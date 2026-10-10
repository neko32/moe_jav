//! Client for a local OpenAI-compatible LLM server (LM Studio).
//!
//! Every call uses `response_format: json_schema` so the model is forced to
//! emit JSON matching the schema the caller supplies.

use std::future::Future;

use serde_json::{Value, json};

use crate::api::http::HttpTransport;
use crate::config::LlmConfig;
use crate::error::{AppError, Result};

const SERVICE: &str = "llm";

/// A structured-output request.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonPrompt<'a> {
    pub system: &'a str,
    pub user: &'a str,
    /// Schema name (letters, digits, `_`).
    pub schema_name: &'a str,
    pub schema: Value,
}

/// Abstraction over the LLM so pipeline logic can be tested offline.
pub trait LlmClient: Sync {
    fn complete_json(&self, prompt: &JsonPrompt<'_>) -> impl Future<Output = Result<Value>> + Send;
}

#[derive(Debug, Clone)]
pub struct LmStudioClient {
    http: HttpTransport,
    config: LlmConfig,
}

impl LmStudioClient {
    #[must_use]
    pub fn new(http: HttpTransport, config: LlmConfig) -> Self {
        Self { http, config }
    }

    /// Builds the request. `with_effort = false` omits `reasoning_effort`,
    /// used when the hint itself triggered runaway thinking.
    fn body(&self, prompt: &JsonPrompt<'_>, with_effort: bool) -> Value {
        let mut body = json!({
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": prompt.schema_name,
                    "strict": true,
                    "schema": prompt.schema,
                }
            }
        });
        if let Some(effort) = self
            .config
            .reasoning_effort
            .as_ref()
            .filter(|_| with_effort)
        {
            body["reasoning_effort"] = json!(effort);
        }
        body
    }
}

impl LlmClient for LmStudioClient {
    async fn complete_json(&self, prompt: &JsonPrompt<'_>) -> Result<Value> {
        let url = self.config.base_url.join("chat/completions");
        let mut raw = self
            .http
            .post_json(SERVICE, &url, None, &self.body(prompt, true))
            .await?;
        // Observed with Gemma 4 E4B: `reasoning_effort: none` occasionally
        // sends the model into endless thinking. Retry once without the hint.
        if is_truncated(&raw) && self.config.reasoning_effort.is_some() {
            tracing::warn!("llm output truncated; retrying without reasoning_effort");
            raw = self
                .http
                .post_json(SERVICE, &url, None, &self.body(prompt, false))
                .await?;
        }
        if is_truncated(&raw) {
            return Err(AppError::BadResponse {
                service: SERVICE,
                message: format!("output truncated at max_tokens={}", self.config.max_tokens),
            });
        }
        extract_json_content(&raw)
    }
}

fn is_truncated(raw: &Value) -> bool {
    raw["choices"][0]["finish_reason"] == "length"
}

/// Pulls `choices[0].message.content` and parses it as JSON. Tolerates a
/// surrounding Markdown code fence, which small models sometimes add.
///
/// # Errors
/// Returns [`AppError::BadResponse`] when the content is absent or not JSON.
pub fn extract_json_content(raw: &Value) -> Result<Value> {
    let content = raw["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AppError::BadResponse {
            service: SERVICE,
            message: "missing choices[0].message.content".into(),
        })?;
    let trimmed = content.trim();
    let unfenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed);
    serde_json::from_str(unfenced.trim()).map_err(|e| AppError::BadResponse {
        service: SERVICE,
        message: format!("content is not JSON: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BaseUrl;
    use std::time::Duration;

    fn client(effort: Option<&str>) -> LmStudioClient {
        LmStudioClient::new(
            HttpTransport::new(Duration::from_secs(1)).unwrap(),
            LlmConfig {
                base_url: BaseUrl::new("http://localhost:1").unwrap(),
                model: "m".into(),
                reasoning_effort: effort.map(str::to_owned),
                max_tokens: 256,
            },
        )
    }

    fn prompt() -> JsonPrompt<'static> {
        JsonPrompt {
            system: "sys",
            user: "usr",
            schema_name: "s",
            schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn body_contains_schema_and_effort() {
        let b = client(Some("none")).body(&prompt(), true);
        assert_eq!(b["max_tokens"], 256);
        assert_eq!(b["model"], "m");
        assert_eq!(b["messages"][0]["content"], "sys");
        assert_eq!(b["messages"][1]["content"], "usr");
        assert_eq!(b["response_format"]["json_schema"]["strict"], true);
        assert_eq!(b["reasoning_effort"], "none");
        let b = client(None).body(&prompt(), true);
        assert!(b.get("reasoning_effort").is_none());
        let b = client(Some("none")).body(&prompt(), false);
        assert!(b.get("reasoning_effort").is_none());
    }

    fn wrap(content: &str) -> Value {
        json!({"choices": [{"message": {"content": content}}]})
    }

    #[test]
    fn extracts_plain_and_fenced_json() {
        assert_eq!(
            extract_json_content(&wrap("{\"a\":1}")).unwrap(),
            json!({"a": 1})
        );
        assert_eq!(
            extract_json_content(&wrap("```json\n{\"a\":2}\n```")).unwrap(),
            json!({"a": 2})
        );
        assert_eq!(
            extract_json_content(&wrap("```\n{\"a\":3}\n```")).unwrap(),
            json!({"a": 3})
        );
    }

    #[test]
    fn extract_errors() {
        assert!(extract_json_content(&json!({})).is_err());
        assert!(extract_json_content(&wrap("not json")).is_err());
    }
}
