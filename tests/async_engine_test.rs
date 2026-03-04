use async_trait::async_trait;
use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::compiler::context::LLMContext;
use engram::provider::embedder::{EmbedError, Embedder};
use engram::provider::llm::{LLMError, LLMProvider};
use engram::types::enums::RetrievalIntent;

struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut embedding = vec![0.0f32; 64];
        for (i, byte) in text.bytes().enumerate() {
            embedding[i % 64] += byte as f32 / 255.0;
        }
        let mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag > 0.0 {
            for v in &mut embedding {
                *v /= mag;
            }
        }
        Ok(embedding)
    }
    fn dimensions(&self) -> usize {
        64
    }
}

struct MockLLM;

#[async_trait]
impl LLMProvider for MockLLM {
    async fn generate(
        &self,
        _system: &str,
        user_message: &str,
        context: &LLMContext,
    ) -> Result<String, LLMError> {
        let mem_count = context.focal_memories.len();
        Ok(format!(
            "Response to '{user_message}' with {mem_count} memories"
        ))
    }
    fn model_name(&self) -> &str {
        "mock"
    }
}

#[tokio::test]
async fn test_async_engine_ingest_and_query() {
    let config = AsyncEngineConfig::default();
    let mut engine = AsyncEngram::new(config, Box::new(MockEmbedder), Box::new(MockLLM));

    engine
        .ingest("Rust is a systems programming language")
        .await
        .unwrap();
    engine
        .ingest("Python is great for data science")
        .await
        .unwrap();
    engine
        .ingest("Rust and Python work together via PyO3")
        .await
        .unwrap();

    let context = engine
        .retrieve(
            "How do Rust and Python work together?",
            RetrievalIntent::Recall,
        )
        .await
        .unwrap();
    assert!(!context.focal_memories.is_empty());
}

#[tokio::test]
async fn test_async_engine_chat() {
    let config = AsyncEngineConfig::default();
    let mut engine = AsyncEngram::new(config, Box::new(MockEmbedder), Box::new(MockLLM));

    engine
        .ingest("The capital of France is Paris")
        .await
        .unwrap();
    let response = engine
        .chat("What is the capital of France?")
        .await
        .unwrap();
    assert!(response.contains("Response to"));
}

#[tokio::test]
async fn test_async_engine_persistence() {
    let dir = std::env::temp_dir().join("engram_async_persist_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Create engine, ingest, save
    {
        let config = AsyncEngineConfig {
            data_dir: dir.clone(),
            ..Default::default()
        };
        let mut engine = AsyncEngram::new(config, Box::new(MockEmbedder), Box::new(MockLLM));
        engine.ingest("persistent fact").await.unwrap();
        assert_eq!(engine.node_count(), 1);
        engine.save().unwrap();
    }

    // Load into new engine
    {
        let config = AsyncEngineConfig {
            data_dir: dir.clone(),
            ..Default::default()
        };
        let engine =
            AsyncEngram::load(config, Box::new(MockEmbedder), Box::new(MockLLM)).unwrap();
        assert_eq!(engine.node_count(), 1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
