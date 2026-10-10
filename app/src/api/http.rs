//! Shared JSON-over-HTTP transport with timeouts, bounded retries and a
//! response size cap.

use std::time::Duration;

use serde_json::Value;

use crate::config::ApiKey;
use crate::error::{AppError, Result};

/// Responses larger than this are rejected (defends against runaway bodies).
pub const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
/// Error bodies are truncated to this many chars before surfacing.
const MAX_ERROR_BODY_CHARS: usize = 500;

#[derive(Debug, Clone)]
pub struct HttpTransport {
    client: reqwest::Client,
    max_retries: u32,
    backoff: Duration,
}

impl HttpTransport {
    /// # Errors
    /// Returns [`AppError::Config`] if the underlying client cannot be built.
    pub fn new(timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| AppError::Config(format!("cannot build http client: {e}")))?;
        Ok(Self {
            client,
            max_retries: 2,
            backoff: Duration::from_millis(300),
        })
    }

    /// Overrides retry behaviour (tests use zero backoff).
    #[must_use]
    pub fn with_retries(mut self, max_retries: u32, backoff: Duration) -> Self {
        self.max_retries = max_retries;
        self.backoff = backoff;
        self
    }

    /// POSTs `body` as JSON and parses a JSON response. Retries on transport
    /// errors, HTTP 429 and 5xx.
    ///
    /// # Errors
    /// Transport, status, size or JSON failures after retries are exhausted.
    pub async fn post_json(
        &self,
        service: &'static str,
        url: &str,
        bearer: Option<&ApiKey>,
        body: &Value,
    ) -> Result<Value> {
        let mut attempt = 0;
        loop {
            match self.post_once(service, url, bearer, body).await {
                Err(e) if attempt < self.max_retries && is_retryable(&e) => {
                    attempt += 1;
                    tracing::warn!(service, attempt, error = %e, "retrying request");
                    tokio::time::sleep(self.backoff * 2u32.pow(attempt - 1)).await;
                }
                other => return other,
            }
        }
    }

    async fn post_once(
        &self,
        service: &'static str,
        url: &str,
        bearer: Option<&ApiKey>,
        body: &Value,
    ) -> Result<Value> {
        let mut req = self.client.post(url).json(body);
        if let Some(key) = bearer {
            req = req.bearer_auth(key.expose());
        }
        let mut resp = req.send().await.map_err(|e| AppError::Http {
            service,
            message: e.to_string(),
        })?;

        let status = resp.status();
        let mut buf = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| AppError::Http {
            service,
            message: e.to_string(),
        })? {
            if buf.len() + chunk.len() > MAX_RESPONSE_BYTES {
                return Err(AppError::BadResponse {
                    service,
                    message: format!("response exceeds {MAX_RESPONSE_BYTES} bytes"),
                });
            }
            buf.extend_from_slice(&chunk);
        }

        if !status.is_success() {
            let text: String = String::from_utf8_lossy(&buf)
                .chars()
                .take(MAX_ERROR_BODY_CHARS)
                .collect();
            return Err(AppError::Status {
                service,
                status: status.as_u16(),
                body: text,
            });
        }

        serde_json::from_slice(&buf).map_err(|e| AppError::BadResponse {
            service,
            message: format!("invalid JSON: {e}"),
        })
    }
}

fn is_retryable(e: &AppError) -> bool {
    match e {
        AppError::Http { .. } => true,
        AppError::Status { status, .. } => *status == 429 || *status >= 500,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        let http = AppError::Http {
            service: "x",
            message: "timeout".into(),
        };
        let st = |status| AppError::Status {
            service: "x",
            status,
            body: String::new(),
        };
        assert!(is_retryable(&http));
        assert!(is_retryable(&st(429)));
        assert!(is_retryable(&st(503)));
        assert!(!is_retryable(&st(400)));
        assert!(!is_retryable(&st(401)));
        assert!(!is_retryable(&AppError::Validation("v".into())));
    }
}
