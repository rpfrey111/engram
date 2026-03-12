use engram::async_engine::{AsyncEngineConfig, AsyncEngram};
use engram::index::vector::cosine_similarity;
use engram::provider::cloudflare::{CloudflareEmbedder, CloudflareLLM};
use engram::provider::embedder::Embedder;
use engram::types::enums::RetrievalIntent;

fn cf_creds() -> Option<(String, String)> {
    let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").ok()?;
    let api_token = std::env::var("CLOUDFLARE_API_TOKEN").ok()?;
    Some((account_id, api_token))
}

// --- Dataset ---

struct TopicFacts {
    name: &'static str,
    facts: &'static [&'static str],
}

const TOPICS: &[TopicFacts] = &[
    TopicFacts {
        name: "photosynthesis",
        facts: &[
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy",
            "Chlorophyll is the green pigment in chloroplasts that absorbs light for photosynthesis",
            "The Calvin cycle fixes carbon dioxide into organic molecules in the stroma of chloroplasts",
            "Light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH",
            "Plants use photosystem I and photosystem II to capture photons and drive electron transport chains",
        ],
    },
    TopicFacts {
        name: "black_holes",
        facts: &[
            "A black hole forms when a massive star collapses under its own gravity after exhausting nuclear fuel",
            "The event horizon is the boundary beyond which nothing, not even light, can escape a black hole",
            "Hawking radiation is theoretical thermal radiation predicted to be emitted by black holes due to quantum effects",
            "Supermassive black holes with millions to billions of solar masses exist at the centers of most galaxies",
            "The singularity at the center of a black hole is a point of theoretically infinite density and curvature",
        ],
    },
    TopicFacts {
        name: "music_theory",
        facts: &[
            "A major scale consists of seven notes following the pattern of whole and half steps: W-W-H-W-W-W-H",
            "Chords are built by stacking thirds: a major triad has a root, major third, and perfect fifth",
            "The circle of fifths organizes all twelve musical keys by ascending perfect fifth intervals",
            "Counterpoint is the technique of combining independent melodic lines into a harmonious texture",
            "Time signatures like 4/4 and 3/4 define the meter by specifying beats per measure and beat value",
        ],
    },
];

fn all_facts() -> Vec<(&'static str, &'static str)> {
    TOPICS
        .iter()
        .flat_map(|t| t.facts.iter().map(move |&f| (t.name, f)))
        .collect()
}

// --- Phase 1: Embedding Quality ---

struct Phase1Result {
    intra_mean: f32,
    inter_mean: f32,
    separation_ratio: f32,
    passed: bool,
}

async fn phase1_embedding_quality(embedder: &dyn Embedder) -> Phase1Result {
    let facts = all_facts();
    let mut embeddings: Vec<(&str, Vec<f32>)> = Vec::new();

    for (topic, fact) in &facts {
        let emb = embedder.embed(fact).await.expect("embed failed");
        embeddings.push((topic, emb));
    }

    let mut intra_sims = Vec::new();
    let mut inter_sims = Vec::new();

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            if embeddings[i].0 == embeddings[j].0 {
                intra_sims.push(sim);
            } else {
                inter_sims.push(sim);
            }
        }
    }

    let intra_mean = intra_sims.iter().sum::<f32>() / intra_sims.len() as f32;
    let inter_mean = inter_sims.iter().sum::<f32>() / inter_sims.len() as f32;
    let separation_ratio = intra_mean / inter_mean;
    let passed = separation_ratio > 1.2;

    Phase1Result {
        intra_mean,
        inter_mean,
        separation_ratio,
        passed,
    }
}

// --- Phase 2: Retrieval Precision ---

struct QueryResult {
    query: String,
    #[allow(dead_code)]
    expected_topic: &'static str,
    precision: f32,
    top1_correct: bool,
    confidence: f32,
    coverage: f32,
}

struct Phase2Result {
    queries: Vec<QueryResult>,
    mean_precision: f32,
    mean_top1: f32,
    precision_passed: bool,
    top1_passed: bool,
}

const RETRIEVAL_QUERIES: &[(&str, &str)] = &[
    ("How do plants convert sunlight into energy?", "photosynthesis"),
    ("What role does chlorophyll play in leaves?", "photosynthesis"),
    ("Explain the Calvin cycle and carbon fixation", "photosynthesis"),
    ("What happens when a massive star collapses?", "black_holes"),
    ("Describe the event horizon and its properties", "black_holes"),
    ("How does Hawking radiation work?", "black_holes"),
    ("What is a major scale and how is it constructed?", "music_theory"),
    ("How are chords built from intervals?", "music_theory"),
    ("Explain the circle of fifths in music", "music_theory"),
];

fn fact_topic(content: &str) -> Option<&'static str> {
    for topic in TOPICS {
        for &fact in topic.facts {
            if content.contains(&fact[..fact.len().min(40)]) {
                return Some(topic.name);
            }
        }
    }
    None
}

async fn phase2_retrieval_precision(
    account_id: &str,
    api_token: &str,
) -> Phase2Result {
    let dir = std::env::temp_dir().join("engram_benchmark_phase2");
    let _ = std::fs::remove_dir_all(&dir);

    let config = AsyncEngineConfig {
        data_dir: dir.clone(),
        working_memory_capacity: 10,
        token_budget: 4000,
        ..Default::default()
    };
    let embedder = Box::new(CloudflareEmbedder::new(account_id, api_token));
    let llm = Box::new(CloudflareLLM::new(account_id, api_token));
    let mut engine = AsyncEngram::new(config, embedder, llm);

    // Ingest all facts
    let facts = all_facts();
    for (_topic, fact) in &facts {
        engine.ingest(fact).await.expect("ingest failed");
    }

    let mut queries = Vec::new();

    for &(query_text, expected_topic) in RETRIEVAL_QUERIES {
        let ctx = engine
            .retrieve(query_text, RetrievalIntent::Recall)
            .await
            .expect("retrieve failed");

        let total = ctx.focal_memories.len();
        let correct = ctx
            .focal_memories
            .iter()
            .filter(|m| fact_topic(&m.content) == Some(expected_topic))
            .count();

        let precision = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        };

        let top1_correct = ctx
            .focal_memories
            .first()
            .and_then(|m| fact_topic(&m.content))
            == Some(expected_topic);

        queries.push(QueryResult {
            query: query_text.to_string(),
            expected_topic,
            precision,
            top1_correct,
            confidence: ctx.confidence,
            coverage: ctx.coverage,
        });
    }

    let mean_precision = queries.iter().map(|q| q.precision).sum::<f32>() / queries.len() as f32;
    let mean_top1 =
        queries.iter().filter(|q| q.top1_correct).count() as f32 / queries.len() as f32;

    let _ = std::fs::remove_dir_all(&dir);

    Phase2Result {
        queries,
        mean_precision,
        mean_top1,
        precision_passed: mean_precision >= 0.70,
        top1_passed: mean_top1 >= 0.88,
    }
}

// --- Phase 3: LLM Response Quality ---

struct ChatResult {
    query: String,
    keyword_hits: usize,
    keyword_total: usize,
    contamination: usize,
    score: f32,
}

struct Phase3Result {
    chats: Vec<ChatResult>,
    mean_score: f32,
    passed: bool,
}

struct ChatQuery {
    query: &'static str,
    expected_topic: &'static str,
    keywords: &'static [&'static str],
}

const CHAT_QUERIES: &[ChatQuery] = &[
    ChatQuery {
        query: "Explain how photosynthesis works and what molecules are involved",
        expected_topic: "photosynthesis",
        keywords: &["chlorophyll", "glucose", "oxygen", "carbon dioxide", "light", "chloroplast"],
    },
    ChatQuery {
        query: "What are black holes and what makes them special in physics?",
        expected_topic: "black_holes",
        keywords: &["gravity", "event horizon", "light", "mass", "singularity", "star"],
    },
    ChatQuery {
        query: "How are musical scales and chords constructed?",
        expected_topic: "music_theory",
        keywords: &["scale", "chord", "interval", "third", "note", "key"],
    },
];

// Keywords that indicate cross-topic contamination
const CONTAMINATION_KEYWORDS: &[(&str, &[&str])] = &[
    (
        "photosynthesis",
        &["chlorophyll", "photosystem", "calvin cycle", "thylakoid", "chloroplast"],
    ),
    (
        "black_holes",
        &["event horizon", "hawking radiation", "singularity", "supermassive"],
    ),
    (
        "music_theory",
        &["counterpoint", "circle of fifths", "triad", "time signature", "major scale"],
    ),
];

fn count_keyword_hits(response: &str, keywords: &[&str]) -> usize {
    let lower = response.to_lowercase();
    keywords
        .iter()
        .filter(|kw| lower.contains(&kw.to_lowercase()))
        .count()
}

fn count_contamination(response: &str, correct_topic: &str) -> usize {
    let lower = response.to_lowercase();
    let mut count = 0;
    for (topic, keywords) in CONTAMINATION_KEYWORDS {
        if *topic == correct_topic {
            continue;
        }
        for kw in *keywords {
            if lower.contains(&kw.to_lowercase()) {
                count += 1;
            }
        }
    }
    count
}

async fn phase3_llm_quality(account_id: &str, api_token: &str) -> Phase3Result {
    let dir = std::env::temp_dir().join("engram_benchmark_phase3");
    let _ = std::fs::remove_dir_all(&dir);

    let config = AsyncEngineConfig {
        data_dir: dir.clone(),
        working_memory_capacity: 10,
        token_budget: 4000,
        ..Default::default()
    };
    let embedder = Box::new(CloudflareEmbedder::new(account_id, api_token));
    let llm = Box::new(CloudflareLLM::new(account_id, api_token));
    let mut engine = AsyncEngram::new(config, embedder, llm);

    // Ingest all facts
    let facts = all_facts();
    for (_topic, fact) in &facts {
        engine.ingest(fact).await.expect("ingest failed");
    }

    let mut chats = Vec::new();

    for cq in CHAT_QUERIES {
        let response = engine.chat(cq.query).await.expect("chat failed");

        let hits = count_keyword_hits(&response, cq.keywords);
        let contamination = count_contamination(&response, cq.expected_topic);

        let base_score = if hits >= 2 {
            1.0
        } else if hits == 1 {
            0.5
        } else {
            0.0
        };
        let score = (base_score - contamination as f32 * 0.25).max(0.0);

        chats.push(ChatResult {
            query: cq.query.to_string(),
            keyword_hits: hits,
            keyword_total: cq.keywords.len(),
            contamination,
            score,
        });
    }

    let mean_score = chats.iter().map(|c| c.score).sum::<f32>() / chats.len() as f32;

    let _ = std::fs::remove_dir_all(&dir);

    Phase3Result {
        chats,
        mean_score,
        passed: mean_score >= 0.65,
    }
}

// --- Scorecard ---

fn print_scorecard(p1: &Phase1Result, p2: &Phase2Result, p3: &Phase3Result) {
    let pass_count = [p1.passed, p2.precision_passed, p2.top1_passed, p3.passed]
        .iter()
        .filter(|&&p| p)
        .count();

    let pass_str = |p: bool| if p { "PASS" } else { "FAIL" };

    println!();
    println!("================================================================");
    println!("                  ENGRAM BENCHMARK SCORECARD");
    println!("================================================================");

    println!();
    println!("--- Phase 1: Embedding Quality ---");
    println!("  Intra-topic mean similarity:  {:.4}", p1.intra_mean);
    println!("  Inter-topic mean similarity:  {:.4}", p1.inter_mean);
    println!(
        "  Separation ratio:             {:.4}  {}",
        p1.separation_ratio,
        pass_str(p1.passed)
    );

    println!();
    println!("--- Phase 2: Retrieval Precision ---");
    println!(
        "  {:<40} {:>4}  {:>4}  {:>4}  {:>4}",
        "Query", "Prec", "Top1", "Conf", "Cov"
    );
    println!("  {}", "-".repeat(62));
    for q in &p2.queries {
        let short_query: String = if q.query.len() > 38 {
            format!("{}...", &q.query[..35])
        } else {
            q.query.clone()
        };
        println!(
            "  {:<40} {:>3.0}%  {:>4}  {:>3.0}%  {:>3.0}%",
            short_query,
            q.precision * 100.0,
            if q.top1_correct { "Y" } else { "N" },
            q.confidence * 100.0,
            q.coverage * 100.0,
        );
    }
    println!(
        "  MEAN topic precision:  {:.0}%  {}",
        p2.mean_precision * 100.0,
        pass_str(p2.precision_passed)
    );
    println!(
        "  MEAN top-1 accuracy:   {:.0}%  {}",
        p2.mean_top1 * 100.0,
        pass_str(p2.top1_passed)
    );

    println!();
    println!("--- Phase 3: LLM Response Quality ---");
    for c in &p3.chats {
        let short_query: String = if c.query.len() > 55 {
            format!("{}...", &c.query[..52])
        } else {
            c.query.clone()
        };
        println!("  Query: \"{}\"", short_query);
        println!(
            "    Keywords: {}/{} | Contamination: {} | Score: {:.2}",
            c.keyword_hits, c.keyword_total, c.contamination, c.score
        );
    }
    println!(
        "  MEAN keyword score: {:.2}  {}",
        p3.mean_score,
        pass_str(p3.passed)
    );

    println!();
    println!("================================================================");
    println!("  OVERALL: {}/4 checks passed", pass_count);
    println!("================================================================");
    println!();
}

// --- Test entry point ---

#[tokio::test]
async fn benchmark_engram_system() {
    let Some((account_id, api_token)) = cf_creds() else {
        eprintln!("Skipping benchmark test (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set)");
        return;
    };

    eprintln!("Running Phase 1: Embedding Quality...");
    let embedder = CloudflareEmbedder::new(&account_id, &api_token);
    let p1 = phase1_embedding_quality(&embedder).await;

    eprintln!("Running Phase 2: Retrieval Precision...");
    let p2 = phase2_retrieval_precision(&account_id, &api_token).await;

    eprintln!("Running Phase 3: LLM Response Quality...");
    let p3 = phase3_llm_quality(&account_id, &api_token).await;

    print_scorecard(&p1, &p2, &p3);

    // Assert overall pass
    let pass_count = [p1.passed, p2.precision_passed, p2.top1_passed, p3.passed]
        .iter()
        .filter(|&&p| p)
        .count();
    assert!(
        pass_count >= 3,
        "Benchmark failed: only {pass_count}/4 checks passed"
    );
}
