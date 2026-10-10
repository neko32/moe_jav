//! Adapters for external services. Each exposes a trait so the core logic
//! can run against stubs.

pub mod http;
pub mod jev;
pub mod llm;
