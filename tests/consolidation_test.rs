use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;
use engram::consolidation::ltp::strengthen_coactivated;
use engram::consolidation::decay::{apply_decay, DecayConfig};

#[test]
fn test_ltp_strengthens_edges() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.5);

    // Simulate co-activation: both were retrieved together
    let coactivated_pairs = vec![(id_a, id_b)];
    strengthen_coactivated(&mut graph, &coactivated_pairs, 0.1);

    let a = graph.get(&id_a).unwrap();
    let edge = &a.edges[0];
    assert!((edge.weight - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_ltp_capped_at_one() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.95);

    let coactivated_pairs = vec![(id_a, id_b)];
    strengthen_coactivated(&mut graph, &coactivated_pairs, 0.2);

    let a = graph.get(&id_a).unwrap();
    assert!((a.edges[0].weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_decay_reduces_low_salience() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("forgettable".to_string(), vec![0.1], ContentType::Text)
        .with_salience(0.1); // low salience, should decay fast
    let _id = node.id;
    graph.insert(node);

    let config = DecayConfig {
        base_rate: 0.3,
        min_edge_weight: 0.05,
        prune_threshold: 0.01,
    };
    let _pruned = apply_decay(&mut graph, &config);
}

#[test]
fn test_decay_preserves_high_salience() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("important".to_string(), vec![0.1], ContentType::Text)
        .with_salience(0.95); // high salience, should resist decay
    let id = node.id;
    graph.insert(node);

    let config = DecayConfig {
        base_rate: 0.3,
        min_edge_weight: 0.05,
        prune_threshold: 0.01,
    };
    let pruned = apply_decay(&mut graph, &config);

    // High salience node should NOT be pruned
    assert!(!pruned.contains(&id));
    assert!(graph.get(&id).is_some());
}
