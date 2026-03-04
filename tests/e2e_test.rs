use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::ollama::{OllamaEmbedder, OllamaLLM};
use engram::types::enums::RetrievalIntent;

/// Full end-to-end test with real Ollama.
/// Requires: ollama running with nomic-embed-text and llama3.2 pulled.
#[tokio::test]
async fn test_e2e_ingest_and_chat() {
    let dir = std::env::temp_dir().join("engram_e2e_test");
    let _ = std::fs::remove_dir_all(&dir);

    let config = AsyncEngineConfig {
        data_dir: dir.clone(),
        ..Default::default()
    };

    let embedder = Box::new(OllamaEmbedder::new(
        "http://localhost:11434",
        "nomic-embed-text",
    ));
    let llm = Box::new(OllamaLLM::new("http://localhost:11434", "llama3.2"));

    let mut engine = AsyncEngram::new(config, embedder, llm);

    // Ingest some facts
    let result = engine
        .ingest("Engram is a neuromorphic memory system for LLMs")
        .await;
    match result {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Skipping e2e test (Ollama not available): {e}");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
    }

    engine
        .ingest("It uses spreading activation like the human brain")
        .await
        .unwrap();
    engine
        .ingest("Memory nodes are connected by weighted edges")
        .await
        .unwrap();

    assert_eq!(engine.node_count(), 3);

    // Retrieve
    let context = engine
        .retrieve("How does Engram work?", RetrievalIntent::Recall)
        .await
        .unwrap();
    assert!(!context.focal_memories.is_empty());

    // Chat
    let response = engine.chat("What is Engram?").await.unwrap();
    assert!(!response.is_empty());
    eprintln!("LLM response: {response}");

    // Verify episodic memory was stored
    assert!(engine.node_count() > 3);

    // Test persistence
    engine.save().unwrap();

    let config2 = AsyncEngineConfig {
        data_dir: dir.clone(),
        ..Default::default()
    };
    let embedder2 = Box::new(OllamaEmbedder::new(
        "http://localhost:11434",
        "nomic-embed-text",
    ));
    let llm2 = Box::new(OllamaLLM::new("http://localhost:11434", "llama3.2"));
    let loaded = AsyncEngram::load(config2, embedder2, llm2).unwrap();
    assert!(loaded.node_count() > 3);

    let _ = std::fs::remove_dir_all(&dir);
}
