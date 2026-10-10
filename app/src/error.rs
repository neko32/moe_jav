//! Crate-wide error type.

use thiserror::Error;

/// Every failure the pipeline can surface.
#[derive(Debug, Error)]
pub enum AppError {
    /// A value failed newtype validation.
    #[error("validation error: {0}")]
    Validation(String),

    /// Required configuration is missing or malformed.
    #[error("configuration error: {0}")]
    Config(String),

    /// Transport-level failure (connect, timeout, TLS, ...).
    #[error("http error calling {service}: {message}")]
    Http {
        service: &'static str,
        message: String,
    },

    /// The remote service answered with a non-success status.
    #[error("{service} returned HTTP {status}: {body}")]
    Status {
        service: &'static str,
        status: u16,
        body: String,
    },

    /// The remote service answered, but not in the shape we expected.
    #[error("unexpected response from {service}: {message}")]
    BadResponse {
        service: &'static str,
        message: String,
    },

    /// Local file or dataset problems.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, AppError>;
