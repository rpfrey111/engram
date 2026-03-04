use engram::types::enums::*;
use engram::types::memory_node::MemoryNode;
use engram::index::graph::GraphStore;
use engram::index::vector;

#[test]
fn test_graph_insert_and_get() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("hello".to_string(), vec![0.1, 0.2], ContentType::Text);
    let id = node.id;
    graph.insert(node);
    let retrieved = graph.get(&id).unwrap();
    assert_eq!(retrieved.content, "hello");
}

#[test]
fn test_graph_get_nonexistent() {
    let graph = GraphStore::new();
    let id = uuid::Uuid::new_v4();
    assert!(graph.get(&id).is_none());
}

#[test]
fn test_graph_add_edge_between_nodes() {
    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let node_b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let id_a = node_a.id;
    let id_b = node_b.id;
    graph.insert(node_a);
    graph.insert(node_b);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.8);

    let a = graph.get(&id_a).unwrap();
    assert_eq!(a.edges.len(), 1);
    assert_eq!(a.edges[0].target_id, id_b);
}

#[test]
fn test_graph_neighbors() {
    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text);
    let node_b = MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text);
    let node_c = MemoryNode::new("C".to_string(), vec![0.3], ContentType::Text);
    let id_a = node_a.id;
    let id_b = node_b.id;
    let id_c = node_c.id;
    graph.insert(node_a);
    graph.insert(node_b);
    graph.insert(node_c);

    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.9);
    graph.add_edge(id_a, id_c, RelationType::Temporal, 0.5);

    let neighbors = graph.neighbors(&id_a);
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_graph_remove_node() {
    let mut graph = GraphStore::new();
    let node = MemoryNode::new("remove me".to_string(), vec![0.1], ContentType::Text);
    let id = node.id;
    graph.insert(node);
    assert!(graph.get(&id).is_some());
    graph.remove(&id);
    assert!(graph.get(&id).is_none());
}

#[test]
fn test_graph_node_count() {
    let mut graph = GraphStore::new();
    assert_eq!(graph.len(), 0);
    graph.insert(MemoryNode::new("A".to_string(), vec![0.1], ContentType::Text));
    graph.insert(MemoryNode::new("B".to_string(), vec![0.2], ContentType::Text));
    assert_eq!(graph.len(), 2);
}

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = vector::cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_find_seeds_by_similarity() {
    let mut graph = GraphStore::new();
    let close = MemoryNode::new("close".to_string(), vec![0.9, 0.1, 0.0], ContentType::Text);
    let far = MemoryNode::new("far".to_string(), vec![0.0, 0.0, 1.0], ContentType::Text);
    let medium = MemoryNode::new("medium".to_string(), vec![0.6, 0.4, 0.1], ContentType::Text);
    let close_id = close.id;
    graph.insert(close);
    graph.insert(far);
    graph.insert(medium);

    let query = vec![1.0, 0.0, 0.0];
    let seeds = vector::find_seeds_by_similarity(&graph, &query, 2, 0.0);
    assert_eq!(seeds.len(), 2);
    assert_eq!(seeds[0].0, close_id); // closest first
}

#[test]
fn test_find_seeds_with_threshold() {
    let mut graph = GraphStore::new();
    graph.insert(MemoryNode::new("close".to_string(), vec![0.95, 0.05], ContentType::Text));
    graph.insert(MemoryNode::new("far".to_string(), vec![0.0, 1.0], ContentType::Text));

    let query = vec![1.0, 0.0];
    let seeds = vector::find_seeds_by_similarity(&graph, &query, 10, 0.5);
    assert_eq!(seeds.len(), 1); // only the close one passes threshold
}
