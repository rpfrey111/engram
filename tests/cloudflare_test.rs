use engram::provider::cloudflare::{CloudflareEmbedder, CloudflareLLM};
use engram::provider::embedder::Embedder;
use engram::provider::llm::LLMProvider;

fn cf_creds() -> Option<(String, String)> {
    let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").ok()?;
    let api_token = std::env::var("CLOUDFLARE_API_TOKEN").ok()?;
    Some((account_id, api_token))
}

#[tokio::test]
async fn test_cloudflare_embedder_produces_vector() {
    let Some((account_id, api_token)) = cf_creds() else {
        eprintln!("Skipping Cloudflare embedder test (env vars not set)");
        return;
    };

    let embedder = CloudflareEmbedder::new(&account_id, &api_token);
    let result = embedder.embed("Hello world").await;
    match result {
        Ok(vec) => {
            assert!(!vec.is_empty());
            assert_eq!(vec.len(), 768);
            let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(magnitude > 0.0);
        }
        Err(e) => {
            panic!("Cloudflare embedder failed: {e}");
        }
    }
}

#[tokio::test]
async fn test_cloudflare_embedder_similar_texts() {
    let Some((account_id, api_token)) = cf_creds() else {
        eprintln!("Skipping Cloudflare similarity test (env vars not set)");
        return;
    };

    let embedder = CloudflareEmbedder::new(&account_id, &api_token);

    let a = embedder.embed("The cat sat on the mat").await.unwrap();
    let b = embedder.embed("A feline rested on the rug").await.unwrap();
    let c = embedder.embed("Quantum computing uses qubits").await.unwrap();

    let sim_ab = engram::index::vector::cosine_similarity(&a, &b);
    let sim_ac = engram::index::vector::cosine_similarity(&a, &c);
    assert!(sim_ab > sim_ac, "sim_ab={sim_ab} should be > sim_ac={sim_ac}");
}

#[tokio::test]
async fn test_cloudflare_llm_generates_response() {
    let Some((account_id, api_token)) = cf_creds() else {
        eprintln!("Skipping Cloudflare LLM test (env vars not set)");
        return;
    };

    let llm = CloudflareLLM::new(&account_id, &api_token);
    let context = engram::compiler::context::LLMContext {
        focal_memories: vec![],
        relationship_map: String::new(),
        confidence: 0.0,
        coverage: 0.0,
        gaps: vec![],
    };
    let result = llm
        .generate("You are a helpful assistant.", "Say hello in one sentence.", &context)
        .await;
    match result {
        Ok(response) => {
            assert!(!response.is_empty());
        }
        Err(e) => {
            panic!("Cloudflare LLM failed: {e}");
        }
    }
}
