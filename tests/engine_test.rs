use engram::engine::{Engram, EngineConfig};
use engram::types::enums::*;

#[test]
fn test_engine_ingest_and_retrieve() {
    let mut engine = Engram::new(EngineConfig::default());

    // Ingest some data
    engine.ingest("Rust is a systems programming language", ContentType::Fact, 0.7);
    engine.ingest("Python is great for data science", ContentType::Fact, 0.6);
    engine.ingest("Rust and Python can work together via PyO3", ContentType::Fact, 0.8);

    // Retrieve
    let context = engine.query("How do Rust and Python work together?", RetrievalIntent::Recall);

    assert!(!context.focal_memories.is_empty());
    assert!(context.confidence > 0.0);
}

#[test]
fn test_engine_empty_query() {
    let mut engine = Engram::new(EngineConfig::default());
    let context = engine.query("anything", RetrievalIntent::Recall);
    assert!(context.focal_memories.is_empty());
    assert!((context.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_engine_working_memory_reuse() {
    let mut engine = Engram::new(EngineConfig::default());

    engine.ingest("The capital of France is Paris", ContentType::Fact, 0.8);

    // First query populates working memory
    let ctx1 = engine.query("What is the capital of France?", RetrievalIntent::Recall);
    assert!(!ctx1.focal_memories.is_empty());

    // Second similar query should benefit from working memory
    let ctx2 = engine.query("Tell me about the capital of France", RetrievalIntent::Recall);
    assert!(!ctx2.focal_memories.is_empty());
}

#[test]
fn test_engine_node_count() {
    let mut engine = Engram::new(EngineConfig::default());
    assert_eq!(engine.node_count(), 0);
    engine.ingest("fact one", ContentType::Fact, 0.5);
    engine.ingest("fact two", ContentType::Fact, 0.5);
    assert_eq!(engine.node_count(), 2);
}
