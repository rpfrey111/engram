use engram::sensory::buffer::SensoryBuffer;
use engram::types::enums::ContentType;

#[test]
fn test_detect_content_type_code() {
    assert_eq!(
        SensoryBuffer::detect_content_type("fn main() { println!(\"hello\"); }"),
        ContentType::Code,
    );
}

#[test]
fn test_detect_content_type_conversation() {
    assert_eq!(
        SensoryBuffer::detect_content_type("User: Hello\nAssistant: Hi there!"),
        ContentType::Conversation,
    );
}

#[test]
fn test_detect_content_type_fact() {
    assert_eq!(
        SensoryBuffer::detect_content_type("The capital of France is Paris."),
        ContentType::Fact,
    );
}

#[test]
fn test_extract_entities() {
    let entities = SensoryBuffer::extract_entities("Ryan Frey works at Anthropic in San Francisco.");
    assert!(entities.contains(&"Frey".to_string()));
    assert!(entities.contains(&"Anthropic".to_string()));
}

#[test]
fn test_score_novelty_empty() {
    let novelty = SensoryBuffer::score_novelty(&[0.1, 0.2, 0.3], &[]);
    assert!((novelty - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_score_novelty_duplicate() {
    let existing = vec![vec![0.1, 0.2, 0.3]];
    let novelty = SensoryBuffer::score_novelty(&[0.1, 0.2, 0.3], &existing);
    assert!(novelty < 0.01);
}
