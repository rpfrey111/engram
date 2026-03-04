use engram::index::graph::GraphStore;
use engram::types::enums::{ContentType, RelationType};
use engram::types::memory_node::MemoryNode;
use std::path::PathBuf;

#[test]
fn test_graph_save_and_load() {
    let dir = std::env::temp_dir().join("engram_test_persistence");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("graph.json");

    let mut graph = GraphStore::new();
    let node_a = MemoryNode::new("fact A".to_string(), vec![0.1, 0.2], ContentType::Fact);
    let node_b = MemoryNode::new("fact B".to_string(), vec![0.3, 0.4], ContentType::Fact);
    let id_a = node_a.id;
    let id_b = node_b.id;
    graph.insert(node_a);
    graph.insert(node_b);
    graph.add_edge(id_a, id_b, RelationType::Semantic, 0.8);

    graph.save(&path).unwrap();
    assert!(path.exists());

    let loaded = GraphStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    let loaded_a = loaded.get(&id_a).unwrap();
    assert_eq!(loaded_a.content, "fact A");
    assert_eq!(loaded_a.edges.len(), 1);
    assert_eq!(loaded_a.edges[0].target_id, id_b);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_graph_load_nonexistent_returns_empty() {
    let path = PathBuf::from("/tmp/engram_nonexistent_graph.json");
    let graph = GraphStore::load(&path).unwrap();
    assert_eq!(graph.len(), 0);
}
