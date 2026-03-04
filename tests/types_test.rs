use engram::types::enums::*;
use engram::types::edge::Edge;
use engram::types::memory_node::MemoryNode;
use engram::types::cue::RetrievalCue;
use engram::types::constellation::MemoryConstellation;
use uuid::Uuid;

#[test]
fn test_content_type_serialization() {
    let ct = ContentType::Conversation;
    let json = serde_json::to_string(&ct).unwrap();
    let deserialized: ContentType = serde_json::from_str(&json).unwrap();
    assert_eq!(ct, deserialized);
}

#[test]
fn test_relation_type_all_variants() {
    let types = vec![
        RelationType::Semantic,
        RelationType::Temporal,
        RelationType::Causal,
        RelationType::Contextual,
        RelationType::Hierarchical,
    ];
    assert_eq!(types.len(), 5);
}

#[test]
fn test_abstraction_level_ordering() {
    assert!((AbstractionLevel::Raw as u8) < (AbstractionLevel::Chunk as u8));
    assert!((AbstractionLevel::Chunk as u8) < (AbstractionLevel::Summary as u8));
    assert!((AbstractionLevel::Summary as u8) < (AbstractionLevel::Schema as u8));
}

#[test]
fn test_retrieval_intent_variants() {
    let intents = vec![
        RetrievalIntent::Recall,
        RetrievalIntent::Recognize,
        RetrievalIntent::Explore,
        RetrievalIntent::Verify,
    ];
    assert_eq!(intents.len(), 4);
}

#[test]
fn test_edge_creation() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Semantic, 0.8);
    assert_eq!(edge.target_id, target);
    assert_eq!(edge.relation_type, RelationType::Semantic);
    assert!((edge.weight - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_edge_weight_clamped() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Causal, 1.5);
    assert!(edge.weight <= 1.0);

    let edge2 = Edge::new(target, RelationType::Causal, -0.5);
    assert!(edge2.weight >= -1.0);
}

#[test]
fn test_edge_strengthen() {
    let target = Uuid::new_v4();
    let mut edge = Edge::new(target, RelationType::Semantic, 0.5);
    edge.strengthen(0.1);
    assert!((edge.weight - 0.6).abs() < f32::EPSILON);

    // Should not exceed 1.0
    edge.strengthen(0.9);
    assert!((edge.weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_edge_weaken() {
    let target = Uuid::new_v4();
    let mut edge = Edge::new(target, RelationType::Temporal, 0.5);
    edge.weaken(0.2);
    assert!((edge.weight - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_inhibitory_edge() {
    let target = Uuid::new_v4();
    let edge = Edge::new(target, RelationType::Semantic, -0.5);
    assert!(edge.is_inhibitory());
}

#[test]
fn test_memory_node_creation() {
    let node = MemoryNode::new(
        "test content".to_string(),
        vec![0.1, 0.2, 0.3],
        ContentType::Text,
    );
    assert_eq!(node.content, "test content");
    assert_eq!(node.content_type, ContentType::Text);
    assert_eq!(node.abstraction_level, AbstractionLevel::Raw);
    assert_eq!(node.access_count, 0);
    assert!(node.edges.is_empty());
}

#[test]
fn test_memory_node_add_edge() {
    let mut node = MemoryNode::new(
        "source".to_string(),
        vec![0.1],
        ContentType::Text,
    );
    let target_id = Uuid::new_v4();
    node.add_edge(target_id, RelationType::Semantic, 0.9);
    assert_eq!(node.edges.len(), 1);
    assert_eq!(node.edges[0].target_id, target_id);
}

#[test]
fn test_memory_node_record_access() {
    let mut node = MemoryNode::new(
        "test".to_string(),
        vec![0.1],
        ContentType::Fact,
    );
    assert_eq!(node.access_count, 0);
    node.record_access();
    assert_eq!(node.access_count, 1);
    node.record_access();
    assert_eq!(node.access_count, 2);
}

#[test]
fn test_memory_node_salience_default() {
    let node = MemoryNode::new(
        "test".to_string(),
        vec![0.1],
        ContentType::Text,
    );
    assert!((node.salience - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_memory_node_serialization() {
    let node = MemoryNode::new(
        "test content".to_string(),
        vec![0.1, 0.2],
        ContentType::Event,
    );
    let json = serde_json::to_string(&node).unwrap();
    let deserialized: MemoryNode = serde_json::from_str(&json).unwrap();
    assert_eq!(node.id, deserialized.id);
    assert_eq!(node.content, deserialized.content);
}

#[test]
fn test_retrieval_cue_creation() {
    let cue = RetrievalCue::new(
        vec![0.1, 0.2, 0.3],
        RetrievalIntent::Recall,
    );
    assert_eq!(cue.intent, RetrievalIntent::Recall);
    assert!(cue.entities.is_empty());
    assert!((cue.salience_floor - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_retrieval_cue_builder() {
    let cue = RetrievalCue::new(vec![0.1], RetrievalIntent::Explore)
        .with_entities(vec!["Acme Corp".to_string()])
        .with_salience_floor(0.3);
    assert_eq!(cue.entities.len(), 1);
    assert!((cue.salience_floor - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_constellation_empty() {
    let constellation = MemoryConstellation::empty();
    assert!(constellation.focal_nodes.is_empty());
    assert!(constellation.context_nodes.is_empty());
    assert!((constellation.confidence - 0.0).abs() < f32::EPSILON);
    assert!((constellation.coverage - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_constellation_with_focal_node() {
    let node = MemoryNode::new("test".to_string(), vec![0.1], ContentType::Fact);
    let mut constellation = MemoryConstellation::empty();
    constellation.add_focal(node, 0.9);
    assert_eq!(constellation.focal_nodes.len(), 1);
    assert!((constellation.focal_nodes[0].activation - 0.9).abs() < f32::EPSILON);
}
