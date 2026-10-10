//! Runtime configuration, read from environment variables (optionally via
//! `.env`). Secrets are wrapped so they never appear in logs.

use std::fmt;
use std::time::Duration;

use crate::error::{AppError, Result};

/// Env var holding the `TypeSafe` (Jev) API key.
pub const ENV_JEV_API_KEY: &str = "API_KEY_JEV";
pub const ENV_JEV_BASE_URL: &str = "JEV_BASE_URL";
pub const ENV_JEV_MODEL: &str = "JEV_MODEL";
pub const ENV_LLM_BASE_URL: &str = "LLM_BASE_URL";
pub const ENV_LLM_MODEL: &str = "LLM_MODEL";
pub const ENV_LLM_REASONING_EFFORT: &str = "LLM_REASONING_EFFORT";
pub const ENV_LLM_MAX_TOKENS: &str = "LLM_MAX_TOKENS";
pub const ENV_HTTP_TIMEOUT_SECS: &str = "HTTP_TIMEOUT_SECS";

pub const DEFAULT_JEV_BASE_URL: &str = "https://api.typesafe.ai/v1";
pub const DEFAULT_JEV_MODEL: &str = "jev-latest";
pub const DEFAULT_LLM_BASE_URL: &str = "http://localhost:1234/v1";
pub const DEFAULT_LLM_MODEL: &str = "google/gemma-4-e4b";
pub const DEFAULT_LLM_REASONING_EFFORT: &str = "none";
pub const DEFAULT_LLM_MAX_TOKENS: u32 = 1024;
pub const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// A bearer token. `Debug`/`Display` are redacted on purpose.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// # Errors
    /// Returns [`AppError::Config`] if the key is empty or contains whitespace.
    pub fn new(raw: &str) -> Result<Self> {
        let key = raw.trim();
        if key.is_empty() {
            return Err(AppError::Config("API key must not be empty".into()));
        }
        if key.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(AppError::Config(
                "API key must not contain whitespace".into(),
            ));
        }
        Ok(Self(key.to_owned()))
    }

    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(***)")
    }
}

/// A service base URL. Plain `http` is only allowed for loopback hosts so a
/// key can never be sent unencrypted over the network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseUrl(String);

impl BaseUrl {
    /// # Errors
    /// Returns [`AppError::Config`] for unsupported schemes or non-loopback
    /// plain-http URLs.
    pub fn new(raw: &str) -> Result<Self> {
        let url = raw.trim().trim_end_matches('/');
        if let Some(rest) = url.strip_prefix("https://") {
            if rest.is_empty() {
                return Err(AppError::Config(format!("invalid URL '{raw}'")));
            }
        } else if let Some(rest) = url.strip_prefix("http://") {
            let host = rest.split(['/', ':']).next().unwrap_or_default();
            if !matches!(host, "localhost" | "127.0.0.1") {
                return Err(AppError::Config(format!(
                    "plain http is only allowed for localhost, got '{raw}'"
                )));
            }
        } else {
            return Err(AppError::Config(format!(
                "URL must start with https:// (or http://localhost), got '{raw}'"
            )));
        }
        Ok(Self(url.to_owned()))
    }

    /// Joins a path (e.g. `/chat/completions`) onto the base.
    #[must_use]
    pub fn join(&self, path: &str) -> String {
        format!("{}/{}", self.0, path.trim_start_matches('/'))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Local LLM (LM Studio, OpenAI-compatible) settings.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: BaseUrl,
    pub model: String,
    /// Sent as `reasoning_effort`; `none` keeps Gemma fast.
    pub reasoning_effort: Option<String>,
    /// Completion cap (includes reasoning tokens); bounds runaway generation.
    pub max_tokens: u32,
}

/// Jev (`TypeSafe`) settings.
#[derive(Debug, Clone)]
pub struct JevConfig {
    pub base_url: BaseUrl,
    pub model: String,
    pub api_key: ApiKey,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub llm: LlmConfig,
    pub jev: JevConfig,
    pub http_timeout: Duration,
}

impl Config {
    /// Builds the config from a key lookup (the real env in production, a
    /// map in tests).
    ///
    /// # Errors
    /// Returns [`AppError::Config`] when a value is missing or invalid.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |k: &str, default: &str| {
            lookup(k)
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| default.to_owned())
        };

        let api_key = lookup(ENV_JEV_API_KEY)
            .ok_or_else(|| AppError::Config(format!("{ENV_JEV_API_KEY} is not set")))?;

        let effort = get(ENV_LLM_REASONING_EFFORT, DEFAULT_LLM_REASONING_EFFORT);
        let reasoning_effort = (effort != "default").then_some(effort);

        let timeout_raw = get(
            ENV_HTTP_TIMEOUT_SECS,
            &DEFAULT_HTTP_TIMEOUT_SECS.to_string(),
        );
        let timeout_secs: u64 = timeout_raw
            .parse()
            .ok()
            .filter(|s| (1..=600).contains(s))
            .ok_or_else(|| {
                AppError::Config(format!(
                    "{ENV_HTTP_TIMEOUT_SECS} must be 1..=600, got '{timeout_raw}'"
                ))
            })?;

        let tokens_raw = get(ENV_LLM_MAX_TOKENS, &DEFAULT_LLM_MAX_TOKENS.to_string());
        let max_tokens: u32 = tokens_raw
            .parse()
            .ok()
            .filter(|t| (64..=32_768).contains(t))
            .ok_or_else(|| {
                AppError::Config(format!(
                    "{ENV_LLM_MAX_TOKENS} must be 64..=32768, got '{tokens_raw}'"
                ))
            })?;

        Ok(Self {
            llm: LlmConfig {
                base_url: BaseUrl::new(&get(ENV_LLM_BASE_URL, DEFAULT_LLM_BASE_URL))?,
                model: get(ENV_LLM_MODEL, DEFAULT_LLM_MODEL),
                reasoning_effort,
                max_tokens,
            },
            jev: JevConfig {
                base_url: BaseUrl::new(&get(ENV_JEV_BASE_URL, DEFAULT_JEV_BASE_URL))?,
                model: get(ENV_JEV_MODEL, DEFAULT_JEV_MODEL),
                api_key: ApiKey::new(&api_key)?,
            },
            http_timeout: Duration::from_secs(timeout_secs),
        })
    }

    /// Reads the process environment (after loading `.env` if present).
    ///
    /// # Errors
    /// See [`Config::from_lookup`].
    pub fn from_env() -> Result<Self> {
        let _ = dotenvy::dotenv();
        Self::from_lookup(|k| std::env::var(k).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn api_key_is_redacted_and_validated() {
        let k = ApiKey::new(" secret-123 ").unwrap();
        assert_eq!(k.expose(), "secret-123");
        assert_eq!(format!("{k:?}"), "ApiKey(***)");
        assert!(ApiKey::new("   ").is_err());
        assert!(ApiKey::new("abc def").is_err());
    }

    #[test]
    fn base_url_enforces_https_except_loopback() {
        assert!(BaseUrl::new("https://api.typesafe.ai/v1/").is_ok());
        assert!(BaseUrl::new("http://localhost:1234/v1").is_ok());
        assert!(BaseUrl::new("http://127.0.0.1:9999").is_ok());
        assert!(BaseUrl::new("http://evil.example.com/v1").is_err());
        assert!(BaseUrl::new("ftp://x").is_err());
        assert!(BaseUrl::new("https://").is_err());
    }

    #[test]
    fn base_url_join_handles_slashes() {
        let u = BaseUrl::new("https://h/v1/").unwrap();
        assert_eq!(u.as_str(), "https://h/v1");
        assert_eq!(u.join("/systemone"), "https://h/v1/systemone");
        assert_eq!(u.join("systemone"), "https://h/v1/systemone");
    }

    #[test]
    fn config_uses_defaults() {
        let c = Config::from_lookup(lookup(&[(ENV_JEV_API_KEY, "k")])).unwrap();
        assert_eq!(c.jev.base_url.as_str(), DEFAULT_JEV_BASE_URL);
        assert_eq!(c.jev.model, DEFAULT_JEV_MODEL);
        assert_eq!(c.llm.base_url.as_str(), DEFAULT_LLM_BASE_URL);
        assert_eq!(c.llm.model, DEFAULT_LLM_MODEL);
        assert_eq!(c.llm.reasoning_effort.as_deref(), Some("none"));
        assert_eq!(c.llm.max_tokens, DEFAULT_LLM_MAX_TOKENS);
        assert_eq!(
            c.http_timeout,
            Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)
        );
    }

    #[test]
    fn config_overrides_and_default_effort_disables_field() {
        let c = Config::from_lookup(lookup(&[
            (ENV_JEV_API_KEY, "k"),
            (ENV_LLM_MODEL, "other-model"),
            (ENV_LLM_REASONING_EFFORT, "default"),
            (ENV_HTTP_TIMEOUT_SECS, "5"),
            (ENV_JEV_MODEL, "jev-1.13.0"),
            (ENV_LLM_MAX_TOKENS, "4096"),
        ]))
        .unwrap();
        assert_eq!(c.llm.max_tokens, 4096);
        assert_eq!(c.llm.model, "other-model");
        assert!(c.llm.reasoning_effort.is_none());
        assert_eq!(c.http_timeout, Duration::from_secs(5));
        assert_eq!(c.jev.model, "jev-1.13.0");
    }

    #[test]
    fn config_rejects_bad_max_tokens() {
        for bad in ["10", "100000", "lots"] {
            assert!(
                Config::from_lookup(lookup(&[(ENV_JEV_API_KEY, "k"), (ENV_LLM_MAX_TOKENS, bad)]))
                    .is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn config_errors() {
        assert!(Config::from_lookup(lookup(&[])).is_err());
        assert!(
            Config::from_lookup(lookup(&[
                (ENV_JEV_API_KEY, "k"),
                (ENV_HTTP_TIMEOUT_SECS, "0")
            ]))
            .is_err()
        );
        assert!(
            Config::from_lookup(lookup(&[
                (ENV_JEV_API_KEY, "k"),
                (ENV_HTTP_TIMEOUT_SECS, "x")
            ]))
            .is_err()
        );
        assert!(
            Config::from_lookup(lookup(&[
                (ENV_JEV_API_KEY, "k"),
                (ENV_JEV_BASE_URL, "http://remote.example.com")
            ]))
            .is_err()
        );
    }
}
