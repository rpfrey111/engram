use engram::router::salience::{SalienceScore, RetrievalStrategy, route};

#[test]
fn test_salience_scoring() {
    let score = SalienceScore {
        novelty: 0.9,
        urgency: 0.8,
        emotional: 0.3,
        entity_weight: 0.5,
    };
    let composite = score.composite();
    assert!(composite > 0.5);
    assert!(composite <= 1.0);
}

#[test]
fn test_route_high_salience_low_coverage() {
    let strategy = route(0.8, 0.1);
    assert!(matches!(strategy, RetrievalStrategy::Deep { .. }));
}

#[test]
fn test_route_high_salience_partial_coverage() {
    let strategy = route(0.8, 0.5);
    assert!(matches!(strategy, RetrievalStrategy::Targeted { .. }));
}

#[test]
fn test_route_any_salience_high_coverage() {
    let strategy = route(0.9, 0.9);
    assert!(matches!(strategy, RetrievalStrategy::Skip));
}

#[test]
fn test_route_low_salience_low_coverage() {
    let strategy = route(0.2, 0.2);
    assert!(matches!(strategy, RetrievalStrategy::Standard { .. }));
}

#[test]
fn test_route_moderate() {
    let strategy = route(0.5, 0.5);
    assert!(matches!(strategy, RetrievalStrategy::Light { .. }));
}
