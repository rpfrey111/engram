use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embedder::{EmbedError, Embedder};

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
