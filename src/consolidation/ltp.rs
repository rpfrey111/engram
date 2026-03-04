use uuid::Uuid;
use crate::index::graph::GraphStore;

pub fn strengthen_coactivated(
    graph: &mut GraphStore,
    coactivated_pairs: &[(Uuid, Uuid)],
    amount: f32,
) {
    for (source_id, target_id) in coactivated_pairs {
        if let Some(source) = graph.get_mut(source_id) {
            for edge in &mut source.edges {
                if edge.target_id == *target_id {
                    edge.strengthen(amount);
                }
            }
        }
    }
}
