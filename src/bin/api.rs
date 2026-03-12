use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::cloudflare::{CloudflareEmbedder, CloudflareLLM};
use engram::types::enums::RetrievalIntent;

// -- Config --

struct ServerConfig {
    api_keys: Vec<String>,
    cloudflare_account_id: String,
    cloudflare_api_token: String,
    data_dir: PathBuf,
}

impl ServerConfig {
    fn from_env() -> Self {
        let api_keys: Vec<String> = std::env::var("ENGRAM_API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let cloudflare_account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .expect("CLOUDFLARE_ACCOUNT_ID env var required");
        let cloudflare_api_token = std::env::var("CLOUDFLARE_API_TOKEN")
            .expect("CLOUDFLARE_API_TOKEN env var required");

        let data_dir = std::env::var("ENGRAM_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./engram_data"));

        Self {
            api_keys,
            cloudflare_account_id,
            cloudflare_api_token,
            data_dir,
        }
    }

    fn is_valid_key(&self, key: &str) -> bool {
        self.api_keys.iter().any(|k| k == key)
    }
}

// -- Rate limiter --

const MAX_REQUESTS_PER_MINUTE: u32 = 60;

struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }

    fn check(&mut self, key: &str) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(60);
        let timestamps = self.requests.entry(key.to_string()).or_default();
        timestamps.retain(|t| now.duration_since(*t) < window);
        if timestamps.len() >= MAX_REQUESTS_PER_MINUTE as usize {
            return false;
        }
        timestamps.push(now);
        true
    }
}

// -- App state --

struct AppState {
    config: ServerConfig,
    engines: Mutex<HashMap<String, AsyncEngram>>,
    rate_limiter: Mutex<RateLimiter>,
}

impl AppState {
    fn get_or_create_engine(&self, api_key: &str) -> AsyncEngram {
        let key_hash = format!("{:x}", md5_simple(api_key));
        let engine_dir = self.config.data_dir.join(&key_hash);

        let engine_config = AsyncEngineConfig {
            data_dir: engine_dir,
            ..Default::default()
        };
        let embedder = Box::new(CloudflareEmbedder::new(
            &self.config.cloudflare_account_id,
            &self.config.cloudflare_api_token,
        ));
        let llm = Box::new(CloudflareLLM::new(
            &self.config.cloudflare_account_id,
            &self.config.cloudflare_api_token,
        ));

        AsyncEngram::load(engine_config.clone(), embedder, llm).unwrap_or_else(|_| {
            let embedder = Box::new(CloudflareEmbedder::new(
                &self.config.cloudflare_account_id,
                &self.config.cloudflare_api_token,
            ));
            let llm = Box::new(CloudflareLLM::new(
                &self.config.cloudflare_account_id,
                &self.config.cloudflare_api_token,
            ));
            AsyncEngram::new(engine_config, embedder, llm)
        })
    }
}

/// Simple hash for directory naming (not cryptographic).
fn md5_simple(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// -- Request/Response types --

#[derive(Deserialize)]
struct IngestRequest {
    content: String,
}

#[derive(Serialize)]
struct IngestResponse {
    id: String,
    node_count: usize,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    memories_used: usize,
    confidence: f32,
}

#[derive(Deserialize)]
struct RetrieveRequest {
    query: String,
}

#[derive(Serialize)]
struct RetrieveResponse {
    memories: Vec<MemoryItem>,
    confidence: f32,
    coverage: f32,
}

#[derive(Serialize)]
struct MemoryItem {
    content: String,
    relevance: f32,
    source: String,
}

#[derive(Serialize)]
struct StatsResponse {
    node_count: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// -- Auth helper --

fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

fn unauthorized() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "invalid api key".to_string(),
        }),
    )
}

fn rate_limited() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(ErrorResponse {
            error: "rate limit exceeded (60 requests/minute)".to_string(),
        }),
    )
}

// -- Handlers --

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn ingest(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<IngestRequest>,
) -> impl IntoResponse {
    let api_key = match extract_api_key(&headers) {
        Some(k) if state.config.is_valid_key(&k) => k,
        _ => return unauthorized().into_response(),
    };

    if !state.rate_limiter.lock().await.check(&api_key) {
        return rate_limited().into_response();
    }

    let mut engines = state.engines.lock().await;
    if !engines.contains_key(&api_key) {
        let engine = state.get_or_create_engine(&api_key);
        engines.insert(api_key.clone(), engine);
    }
    let engine = engines.get_mut(&api_key).unwrap();

    match engine.ingest(&body.content).await {
        Ok(id) => {
            let node_count = engine.node_count();
            if let Err(e) = engine.save() {
                eprintln!("Warning: failed to save after ingest: {e}");
            }
            (
                StatusCode::OK,
                Json(IngestResponse {
                    id: id.to_string(),
                    node_count,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ChatRequest>,
) -> impl IntoResponse {
    let api_key = match extract_api_key(&headers) {
        Some(k) if state.config.is_valid_key(&k) => k,
        _ => return unauthorized().into_response(),
    };

    if !state.rate_limiter.lock().await.check(&api_key) {
        return rate_limited().into_response();
    }

    let mut engines = state.engines.lock().await;
    if !engines.contains_key(&api_key) {
        let engine = state.get_or_create_engine(&api_key);
        engines.insert(api_key.clone(), engine);
    }
    let engine = engines.get_mut(&api_key).unwrap();

    // Retrieve context first for metadata, then do full chat
    let context = match engine.retrieve(&body.message, RetrievalIntent::Recall).await {
        Ok(ctx) => ctx,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    };
    let memories_used = context.focal_memories.len();
    let confidence = context.confidence;

    match engine.chat(&body.message).await {
        Ok(response) => {
            if let Err(e) = engine.save() {
                eprintln!("Warning: failed to save after chat: {e}");
            }
            (
                StatusCode::OK,
                Json(ChatResponse {
                    response,
                    memories_used,
                    confidence,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn retrieve(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<RetrieveRequest>,
) -> impl IntoResponse {
    let api_key = match extract_api_key(&headers) {
        Some(k) if state.config.is_valid_key(&k) => k,
        _ => return unauthorized().into_response(),
    };

    if !state.rate_limiter.lock().await.check(&api_key) {
        return rate_limited().into_response();
    }

    let mut engines = state.engines.lock().await;
    if !engines.contains_key(&api_key) {
        let engine = state.get_or_create_engine(&api_key);
        engines.insert(api_key.clone(), engine);
    }
    let engine = engines.get_mut(&api_key).unwrap();

    match engine.retrieve(&body.query, RetrievalIntent::Recall).await {
        Ok(context) => {
            let memories = context
                .focal_memories
                .iter()
                .map(|m| MemoryItem {
                    content: m.content.clone(),
                    relevance: m.relevance,
                    source: m.source.clone(),
                })
                .collect();
            (
                StatusCode::OK,
                Json(RetrieveResponse {
                    memories,
                    confidence: context.confidence,
                    coverage: context.coverage,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let api_key = match extract_api_key(&headers) {
        Some(k) if state.config.is_valid_key(&k) => k,
        _ => return unauthorized().into_response(),
    };

    if !state.rate_limiter.lock().await.check(&api_key) {
        return rate_limited().into_response();
    }

    let mut engines = state.engines.lock().await;
    if !engines.contains_key(&api_key) {
        let engine = state.get_or_create_engine(&api_key);
        engines.insert(api_key.clone(), engine);
    }
    let engine = engines.get(&api_key).unwrap();

    (
        StatusCode::OK,
        Json(StatsResponse {
            node_count: engine.node_count(),
        }),
    )
        .into_response()
}

// -- Main --

#[tokio::main]
async fn main() {
    let config = ServerConfig::from_env();
    let key_count = config.api_keys.len();
    let data_dir = config.data_dir.clone();

    let state = Arc::new(AppState {
        config,
        engines: Mutex::new(HashMap::new()),
        rate_limiter: Mutex::new(RateLimiter::new()),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/ingest", post(ingest))
        .route("/chat", post(chat))
        .route("/retrieve", post(retrieve))
        .route("/stats", get(stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("Engram API server starting on {addr}");
    eprintln!("  API keys configured: {key_count}");
    eprintln!("  Data directory: {}", data_dir.display());

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
