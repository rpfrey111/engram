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

pub fn build_system_prompt(base_system: &str, context: &LLMContext) -> String {
    let mut prompt = base_system.to_string();

    if !context.focal_memories.is_empty() {
        prompt.push_str("\n\n## Relevant Memories\n");
        for mem in &context.focal_memories {
            prompt.push_str(&format!(
                "- [relevance: {:.2}] {}\n",
                mem.relevance, mem.content
            ));
        }
    }

    if !context.gaps.is_empty() {
        prompt.push_str("\n## Knowledge Gaps\n");
        for gap in &context.gaps {
            prompt.push_str(&format!("- {gap}\n"));
        }
    }

    prompt.push_str(&format!(
        "\n[Memory confidence: {:.0}% | Coverage: {:.0}%]",
        context.confidence * 100.0,
        context.coverage * 100.0
    ));

    prompt
}
