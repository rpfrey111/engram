use uuid::Uuid;

use super::graph::GraphStore;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Support different-length vectors by zero-padding the shorter one.
    // This is correct for bag-of-words where new dimensions are implicitly zero.
    let len = a.len().max(b.len());
    let get_a = |i: usize| if i < a.len() { a[i] } else { 0.0 };
    let get_b = |i: usize| if i < b.len() { b[i] } else { 0.0 };

    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;
    for i in 0..len {
        let ai = get_a(i);
        let bi = get_b(i);
        dot += ai * bi;
        mag_a += ai * ai;
        mag_b += bi * bi;
    }
    mag_a = mag_a.sqrt();
    mag_b = mag_b.sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Returns vec of (node_id, similarity_score) sorted by score descending
pub fn find_seeds_by_similarity(
    graph: &GraphStore,
    query_embedding: &[f32],
    top_k: usize,
    threshold: f32,
) -> Vec<(Uuid, f32)> {
    let mut scores: Vec<(Uuid, f32)> = graph
        .all_nodes()
        .map(|node| {
            let sim = cosine_similarity(&node.embedding, query_embedding);
            (node.id, sim)
        })
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}
