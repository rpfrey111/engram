use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embedder::{EmbedError, Embedder};
use super::llm::{LLMError, LLMProvider};
use crate::compiler::context::LLMContext;

pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    client: reqwest::Client,
    dimensions: usize,
}

impl OllamaEmbedder {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
            dimensions: 768,
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let url = format!("{}/api/embed", self.base_url);
        let req = EmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| EmbedError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::RequestFailed(format!("{status}: {body}")));
        }

        let body: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbedError::InvalidResponse(e.to_string()))?;

        body.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::InvalidResponse("empty embeddings array".to_string()))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// --- OllamaLLM ---

pub struct OllamaLLM {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaLLM {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

fn build_system_prompt(base_system: &str, context: &LLMContext) -> String {
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

#[async_trait]
impl LLMProvider for OllamaLLM {
    async fn generate(
        &self,
        system: &str,
        user_message: &str,
        context: &LLMContext,
    ) -> Result<String, LLMError> {
        let url = format!("{}/api/chat", self.base_url);
        let system_prompt = build_system_prompt(system, context);

        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                },
            ],
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(format!("{status}: {body}")));
        }

        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(body.message.content)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
