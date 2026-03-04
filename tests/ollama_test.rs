use engram::provider::embedder::Embedder;
use engram::provider::ollama::OllamaEmbedder;

#[tokio::test]
async fn test_ollama_embedder_produces_vector() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text");
    let result = embedder.embed("Hello world").await;
    match result {
        Ok(vec) => {
            assert!(!vec.is_empty());
            assert_eq!(vec.len(), 768);
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(magnitude > 0.0);
        }
        Err(e) => {
            eprintln!("Skipping Ollama test (not running): {e}");
        }
    }
}

#[tokio::test]
async fn test_ollama_embedder_similar_texts() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "nomic-embed-text");

    let result_a = embedder.embed("The cat sat on the mat").await;
    let result_b = embedder.embed("A feline rested on the rug").await;
    let result_c = embedder.embed("Quantum computing uses qubits").await;

    match (result_a, result_b, result_c) {
        (Ok(a), Ok(b), Ok(c)) => {
            let sim_ab = engram::index::vector::cosine_similarity(&a, &b);
            let sim_ac = engram::index::vector::cosine_similarity(&a, &c);
            assert!(sim_ab > sim_ac, "sim_ab={sim_ab} should be > sim_ac={sim_ac}");
        }
        _ => {
            eprintln!("Skipping Ollama similarity test (not running)");
        }
    }
}
