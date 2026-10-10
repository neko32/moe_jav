//! Test doubles for the external-service traits.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde_json::Value;

use crate::api::jev::{DecisionModel, JevRequest, JevResponse};
use crate::api::llm::{JsonPrompt, LlmClient};
use crate::error::{AppError, Result};

/// Returns scripted responses in order and records the user prompts it saw.
pub struct ScriptedLlm {
    replies: Mutex<VecDeque<Result<Value>>>,
    prompts: Mutex<Vec<String>>,
}

impl ScriptedLlm {
    pub fn new(replies: Vec<Result<Value>>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            prompts: Mutex::new(Vec::new()),
        }
    }

    pub fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

impl LlmClient for ScriptedLlm {
    async fn complete_json(&self, prompt: &JsonPrompt<'_>) -> Result<Value> {
        self.prompts.lock().unwrap().push(prompt.user.to_owned());
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(AppError::Validation("script exhausted".into())))
    }
}

/// LLM stub driven by a closure over `(system, user)`; order-independent,
/// so it suits concurrent callers.
pub struct FnLlm<F>(pub F);

impl<F> LlmClient for FnLlm<F>
where
    F: Fn(&str, &str) -> Result<Value> + Sync,
{
    async fn complete_json(&self, prompt: &JsonPrompt<'_>) -> Result<Value> {
        (self.0)(prompt.system, prompt.user)
    }
}

/// Decision-model stub driven by a closure over the request.
pub struct FnJev<F>(pub F);

impl<F> DecisionModel for FnJev<F>
where
    F: Fn(&JevRequest) -> Result<Value> + Sync,
{
    async fn decide(&self, req: &JevRequest) -> Result<JevResponse> {
        let v = (self.0)(req)?;
        serde_json::from_value(v).map_err(|e| AppError::BadResponse {
            service: "jev",
            message: e.to_string(),
        })
    }
}
