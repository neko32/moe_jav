//! `moe_jav`: a mixture-of-experts insurance underwriting POC.
//!
//! An application is split into segments by a local LLM, each segment is
//! routed to an expert agent by a gate (Jev or the LLM), the experts analyse
//! their segments, and Jev answers atomic risk questions that explicit rules
//! combine into a decision.

pub mod api;
pub mod cli;
pub mod config;
pub mod core;
pub mod error;
