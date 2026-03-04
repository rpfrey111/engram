use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embedder::{EmbedError, Embedder};
use super::llm::{build_system_prompt, LLMError, LLMProvider};
use crate::compiler::context::LLMContext;

// --- CloudflareEmbedder ---

pub struct CloudflareEmbedder {
    account_id: String,
    api_token: String,
    model: String,
    client: reqwest::Client,
    dimensions: usize,
}

impl CloudflareEmbedder {
    pub fn new(account_id: &str, api_token: &str) -> Self {
        Self {
            account_id: account_id.to_string(),
            api_token: api_token.to_string(),
            model: "@cf/baai/bge-base-en-v1.5".to_string(),
            client: reqwest::Client::new(),
            dimensions: 768,
        }
    }

    pub fn with_model(mut self, model: &str, dimensions: usize) -> Self {
        self.model = model.to_string();
        self.dimensions = dimensions;
        self
    }
}

#[derive(Serialize)]
struct CfEmbedRequest {
    text: String,
}

#[derive(Deserialize)]
struct CfEmbedResult {
    data: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct CfEmbedResponse {
    result: CfEmbedResult,
}

#[async_trait]
impl Embedder for CloudflareEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model
        );
        let req = CfEmbedRequest {
            text: text.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&req)
            .send()
            .await
            .map_err(|e| EmbedError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbedError::RequestFailed(format!("{status}: {body}")));
        }

        let body: CfEmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbedError::InvalidResponse(e.to_string()))?;

        body.result
            .data
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::InvalidResponse("empty embeddings result".to_string()))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// --- CloudflareLLM ---

pub struct CloudflareLLM {
    account_id: String,
    api_token: String,
    model: String,
    client: reqwest::Client,
}

impl CloudflareLLM {
    pub fn new(account_id: &str, api_token: &str) -> Self {
        Self {
            account_id: account_id.to_string(),
            api_token: api_token.to_string(),
            model: "@cf/meta/llama-3.1-8b-instruct".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[derive(Serialize)]
struct CfChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CfChatRequest {
    messages: Vec<CfChatMessage>,
}

#[derive(Deserialize)]
struct CfChatResult {
    response: String,
}

#[derive(Deserialize)]
struct CfChatResponse {
    result: CfChatResult,
}

#[async_trait]
impl LLMProvider for CloudflareLLM {
    async fn generate(
        &self,
        system: &str,
        user_message: &str,
        context: &LLMContext,
    ) -> Result<String, LLMError> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/ai/run/{}",
            self.account_id, self.model
        );
        let system_prompt = build_system_prompt(system, context);

        let req = CfChatRequest {
            messages: vec![
                CfChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                CfChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                },
            ],
        };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&req)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(format!("{status}: {body}")));
        }

        let body: CfChatResponse = resp
            .json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(body.result.response)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
