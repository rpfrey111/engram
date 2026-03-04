use async_trait::async_trait;

use crate::compiler::context::LLMContext;

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("generation request failed: {0}")]
    RequestFailed(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(
        &self,
        system: &str,
        user_message: &str,
        context: &LLMContext,
    ) -> Result<String, LLMError>;
    fn model_name(&self) -> &str;
}
