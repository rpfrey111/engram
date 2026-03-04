use clap::{Parser, Subcommand};
use engram::async_engine::{AsyncEngram, AsyncEngineConfig};
use engram::provider::cloudflare::{CloudflareEmbedder, CloudflareLLM};
use engram::provider::embedder::Embedder;
use engram::provider::llm::LLMProvider;
use engram::provider::ollama::{OllamaEmbedder, OllamaLLM};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "engram", about = "Neuromorphic memory for LLMs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Provider backend: ollama or cloudflare
    #[arg(long, default_value = "ollama", global = true)]
    provider: String,

    /// Ollama server URL
    #[arg(long, default_value = "http://localhost:11434", global = true)]
    ollama_url: String,

    /// Embedding model
    #[arg(long, default_value = "nomic-embed-text", global = true)]
    embed_model: String,

    /// LLM model
    #[arg(long, default_value = "llama3.2", global = true)]
    llm_model: String,

    /// Data directory
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start an interactive chat session with persistent memory
    Chat,
    /// Ingest a file or directory into memory
    Ingest {
        /// Path to file or directory
        path: PathBuf,
    },
    /// Show memory statistics
    Stats,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let data_dir = cli.data_dir.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".engram")
    });

    let config = AsyncEngineConfig {
        data_dir: data_dir.clone(),
        ..Default::default()
    };

    let (embedder, llm): (Box<dyn Embedder>, Box<dyn LLMProvider>) = match cli.provider.as_str() {
        "cloudflare" => {
            let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").unwrap_or_else(|_| {
                eprintln!("Error: CLOUDFLARE_ACCOUNT_ID env var required for cloudflare provider");
                std::process::exit(1);
            });
            let api_token = std::env::var("CLOUDFLARE_API_TOKEN").unwrap_or_else(|_| {
                eprintln!("Error: CLOUDFLARE_API_TOKEN env var required for cloudflare provider");
                std::process::exit(1);
            });
            (
                Box::new(CloudflareEmbedder::new(&account_id, &api_token)),
                Box::new(CloudflareLLM::new(&account_id, &api_token)),
            )
        }
        "ollama" => (
            Box::new(OllamaEmbedder::new(&cli.ollama_url, &cli.embed_model)),
            Box::new(OllamaLLM::new(&cli.ollama_url, &cli.llm_model)),
        ),
        other => {
            eprintln!("Unknown provider: {other}. Use 'ollama' or 'cloudflare'.");
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Chat => run_chat(config, embedder, llm).await,
        Commands::Ingest { path } => run_ingest(config, embedder, llm, &path).await,
        Commands::Stats => run_stats(config, embedder, llm),
    }
}

async fn run_chat(
    config: AsyncEngineConfig,
    embedder: Box<dyn Embedder>,
    llm: Box<dyn LLMProvider>,
) {
    let model_name = llm.model_name().to_string();
    let mut engine = AsyncEngram::load(config, embedder, llm).unwrap_or_else(|e| {
        eprintln!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Engram v0.1.0 | Memory: {} nodes | Model: {}",
        engine.node_count(),
        model_name
    );
    eprintln!("Type 'quit' to exit. Memory is saved on exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        eprint!("You: ");
        stdout.flush().unwrap();

        let mut input = String::new();
        stdin.lock().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        match engine.chat(input).await {
            Ok(response) => {
                eprintln!("\nEngram: {response}\n");
            }
            Err(e) => {
                eprintln!("\nError: {e}\n");
            }
        }
    }

    if let Err(e) = engine.save() {
        eprintln!("Warning: failed to save memory: {e}");
    } else {
        eprintln!("Memory saved ({} nodes).", engine.node_count());
    }
}

async fn run_ingest(
    config: AsyncEngineConfig,
    embedder: Box<dyn Embedder>,
    llm: Box<dyn LLMProvider>,
    path: &std::path::Path,
) {
    let mut engine = AsyncEngram::load(config, embedder, llm).unwrap_or_else(|e| {
        eprintln!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    let before = engine.node_count();

    if path.is_file() {
        ingest_file(&mut engine, path).await;
    } else if path.is_dir() {
        ingest_directory(&mut engine, path).await;
    } else {
        eprintln!("Path does not exist: {}", path.display());
        return;
    }

    let after = engine.node_count();
    eprintln!("Ingested {} new nodes (total: {})", after - before, after);

    if let Err(e) = engine.save() {
        eprintln!("Warning: failed to save memory: {e}");
    }
}

async fn ingest_file(engine: &mut AsyncEngram, path: &std::path::Path) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Failed to read {}: {e}", path.display());
            return;
        }
    };

    let chunks = chunk_content(&content);
    for chunk in &chunks {
        if chunk.trim().is_empty() {
            continue;
        }
        match engine.ingest(chunk).await {
            Ok(_) => {}
            Err(e) => eprintln!("  Failed to ingest chunk: {e}"),
        }
    }
    eprintln!("  {} -> {} chunks", path.display(), chunks.len());
}

async fn ingest_directory(engine: &mut AsyncEngram, path: &std::path::Path) {
    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read directory {}: {e}", path.display());
            return;
        }
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() {
            let name = entry_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            let ext = entry_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let text_extensions = [
                "md", "txt", "rs", "py", "js", "ts", "toml", "yaml", "yml", "json", "html",
                "css",
            ];
            if text_extensions.contains(&ext) {
                ingest_file(engine, &entry_path).await;
            }
        } else if entry_path.is_dir() {
            let name = entry_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            if !name.starts_with('.') && name != "target" && name != "node_modules" {
                Box::pin(ingest_directory(engine, &entry_path)).await;
            }
        }
    }
}

fn chunk_content(content: &str) -> Vec<String> {
    let max_chunk = 500;
    let mut chunks = Vec::new();
    let paragraphs: Vec<&str> = content.split("\n\n").collect();

    let mut current = String::new();
    for para in paragraphs {
        if current.len() + para.len() > max_chunk && !current.is_empty() {
            chunks.push(current.clone());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn run_stats(
    config: AsyncEngineConfig,
    embedder: Box<dyn Embedder>,
    llm: Box<dyn LLMProvider>,
) {
    let engine = AsyncEngram::load(config, embedder, llm).unwrap_or_else(|e| {
        eprintln!("Failed to load engine: {e}");
        std::process::exit(1);
    });
    eprintln!("Engram Memory Stats");
    eprintln!("  Nodes: {}", engine.node_count());
}
