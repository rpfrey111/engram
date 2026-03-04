use engram::types::enums::*;

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
