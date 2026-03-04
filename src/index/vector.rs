use uuid::Uuid;

use super::graph::GraphStore;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

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
