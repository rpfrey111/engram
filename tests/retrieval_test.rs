use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;
use engram::retrieval::activation::{spread, ActivationConfig};
use uuid::Uuid;

fn build_test_graph() -> (GraphStore, Uuid, Uuid, Uuid, Uuid) {
    let mut graph = GraphStore::new();
    // A --0.9--> B --0.8--> C
    // A --0.3--> D
    let a = MemoryNode::new("A".to_string(), vec![1.0, 0.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.8, 0.2], ContentType::Text);
    let c = MemoryNode::new("C".to_string(), vec![0.5, 0.5], ContentType::Text);
    let d = MemoryNode::new("D".to_string(), vec![0.0, 1.0], ContentType::Text);
    let (id_a, id_b, id_c, id_d) = (a.id, b.id, c.id, d.id);

    graph.insert(a);
    graph.insert(b);
    graph.insert(c);
    graph.insert(d);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.9);
    graph.add_edge(id_b, id_c, RelationType::Semantic, 0.8);
    graph.add_edge(id_a, id_d, RelationType::Temporal, 0.3);

    (graph, id_a, id_b, id_c, id_d)
}

#[test]
fn test_spreading_activation_basic() {
    let (graph, id_a, id_b, id_c, id_d) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    // B should have higher activation than D (stronger edge)
    let b_activation = result.get(&id_b).unwrap_or(&0.0);
    let d_activation = result.get(&id_d).unwrap_or(&0.0);
    assert!(b_activation > d_activation);

    // C should have some activation (2 hops from A)
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!(*c_activation > 0.0);

    // C should have less activation than B (further away)
    assert!(c_activation < b_activation);
}

#[test]
fn test_spreading_activation_depth_limit() {
    let (graph, id_a, _, id_c, _) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig { max_depth: 1, ..Default::default() };

    let result = spread(&graph, &seeds, &config);

    // C is 2 hops away, should NOT be activated with max_depth=1
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!((*c_activation).abs() < f32::EPSILON);
}

#[test]
fn test_spreading_activation_threshold() {
    let (graph, id_a, _, _, id_d) = build_test_graph();
    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig {
        min_activation: 0.5,
        ..Default::default()
    };

    let result = spread(&graph, &seeds, &config);

    // D has weak edge (0.3), its activation should be below threshold
    let d_activation = result.get(&id_d).unwrap_or(&0.0);
    assert!((*d_activation).abs() < f32::EPSILON);
}

#[test]
fn test_multi_path_convergence() {
    let mut graph = GraphStore::new();
    // A --0.7--> C
    // B --0.7--> C
    // C should get activation from both paths
    let a = MemoryNode::new("A".to_string(), vec![1.0, 0.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.0, 1.0], ContentType::Text);
    let c = MemoryNode::new("C".to_string(), vec![0.5, 0.5], ContentType::Text);
    let (id_a, id_b, id_c) = (a.id, b.id, c.id);
    graph.insert(a);
    graph.insert(b);
    graph.insert(c);
    graph.add_edge(id_a, id_c, RelationType::Semantic, 0.7);
    graph.add_edge(id_b, id_c, RelationType::Semantic, 0.7);

    let seeds = vec![(id_a, 1.0), (id_b, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    // C reached from two paths, should have higher activation than if from one
    let c_activation = result.get(&id_c).unwrap_or(&0.0);
    assert!(*c_activation > 0.7); // must be more than single-path activation
}

#[test]
fn test_inhibitory_edge() {
    let mut graph = GraphStore::new();
    let a = MemoryNode::new("A".to_string(), vec![1.0], ContentType::Text);
    let b = MemoryNode::new("B".to_string(), vec![0.5], ContentType::Text);
    let (id_a, id_b) = (a.id, b.id);
    graph.insert(a);
    graph.insert(b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, -0.5);

    let seeds = vec![(id_a, 1.0)];
    let config = ActivationConfig::default();

    let result = spread(&graph, &seeds, &config);

    let b_activation = result.get(&id_b).unwrap_or(&0.0);
    assert!(*b_activation < 0.0); // negative activation from inhibitory edge
}
