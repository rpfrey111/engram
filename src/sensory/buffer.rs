use crate::index::vector::cosine_similarity;
use crate::types::enums::ContentType;

pub struct SensoryBuffer;

impl SensoryBuffer {
    /// Detect content type from text patterns (Craik & Lockhart deep processing).
    pub fn detect_content_type(text: &str) -> ContentType {
        let code_signals = [
            "fn ", "def ", "class ", "import ", "func ", "var ", "let ", "const ", "return ",
            "pub ", "struct ", "impl ", "async ", "=>", "->", "println!", "console.log",
        ];
        let code_score: usize = code_signals.iter().filter(|s| text.contains(**s)).count();
        if code_score >= 2 || (code_score >= 1 && (text.contains('{') || text.contains('}'))) {
            return ContentType::Code;
        }

        let conversation_signals = [
            "User:",
            "Assistant:",
            "Human:",
            "AI:",
            "Q:",
            "A:",
            "you said",
            "I think",
            "tell me",
        ];
        let conv_score: usize = conversation_signals
            .iter()
            .filter(|s| text.contains(**s))
            .count();
        if conv_score >= 2 {
            return ContentType::Conversation;
        }

        let event_signals = [
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "meeting",
            "happened",
            "occurred",
            "event",
            "scheduled",
        ];
        let event_score: usize = event_signals
            .iter()
            .filter(|s| text.to_lowercase().contains(*s))
            .count();
        if event_score >= 2 {
            return ContentType::Event;
        }

        if text.len() < 200
            && (text.contains(" is ") || text.contains(" are ") || text.contains(" was ")
                || text.contains(" has "))
        {
            return ContentType::Fact;
        }

        ContentType::Text
    }

    /// Extract entities from text (capitalized words that aren't sentence starters).
    pub fn extract_entities(text: &str) -> Vec<String> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.is_empty() {
                continue;
            }
            let first_char = clean.chars().next().unwrap();
            let is_sentence_start = i == 0 || (i > 0 && words[i - 1].ends_with('.'));
            if first_char.is_uppercase() && !is_sentence_start && clean.len() > 1 {
                entities.push(clean);
            }
        }
        entities.sort();
        entities.dedup();
        entities
    }

    /// Score novelty of an embedding against existing embeddings.
    /// 1.0 = completely novel, 0.0 = exact duplicate.
    pub fn score_novelty(embedding: &[f32], existing: &[Vec<f32>]) -> f32 {
        if existing.is_empty() {
            return 1.0;
        }
        let max_similarity = existing
            .iter()
            .map(|e| cosine_similarity(embedding, e))
            .fold(0.0f32, |acc, s| acc.max(s));
        (1.0 - max_similarity).clamp(0.0, 1.0)
    }
}
