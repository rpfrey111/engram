use engram::types::enums::*;
use engram::types::edge::Edge;
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
