use crate::types::{Label, LabeledRow, RejectReason};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// K-fold cross-validation outcome
// ---------------------------------------------------------------------------

pub struct KFoldOutcome {
    pub accuracy: f64,
    pub auc: f64,
    pub recall_by_reason: HashMap<RejectReason, f64>,
    pub feature_weights: Vec<(String, f64)>,
    pub suggested_high_cutoff: Option<f64>,
}

// ---------------------------------------------------------------------------
// Per-fold reject-reason accumulator
// ---------------------------------------------------------------------------

/// For each reject reason, count `(tp_reject, actual_reject)` — i.e. how many
/// of this fold's rows were rejected for this reason, and how many of those
/// the model also predicted as reject (prob < 0.5). Per-reason **recall on
/// the reject class** is `tp / actual` summed across folds.
pub fn fold_reject_counts(
    fold_preds: &[(usize, f64)], // (labeled_idx, prob)
    labeled: &[LabeledRow],
) -> HashMap<RejectReason, (usize, usize)> {
    let mut counts: HashMap<RejectReason, (usize, usize)> = HashMap::new();
    for &(idx, prob) in fold_preds {
        if let Label::Reject(reason) = labeled[idx].label {
            let e = counts.entry(reason).or_insert((0, 0));
            e.1 += 1;
            if prob < 0.5 {
                e.0 += 1;
            }
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// AUC via 200-trace-pair sampling
// ---------------------------------------------------------------------------

/// AUC via 200-trace-pair sampling: take the top 200 positive and 200 negative
/// predictions (by descending score), then count how often a positive > negative.
/// Returns 0.5 on degenerate inputs.
pub fn fast_auc(pooled: &[(f64, bool)]) -> f64 {
    let mut pos: Vec<f64> = pooled.iter().filter(|(_, y)| *y).map(|(p, _)| *p).collect();
    let mut neg: Vec<f64> = pooled
        .iter()
        .filter(|(_, y)| !*y)
        .map(|(p, _)| *p)
        .collect();
    pos.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    neg.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    pos.truncate(200);
    neg.truncate(200);

    let mut n_pairs: u64 = 0;
    let mut n_correct: f64 = 0.0;
    for &pp in &pos {
        for &nn in &neg {
            n_pairs += 1;
            if pp > nn {
                n_correct += 1.0;
            } else if pp == nn {
                n_correct += 0.5;
            }
        }
    }
    if n_pairs == 0 {
        0.5
    } else {
        n_correct / n_pairs as f64
    }
}

// ---------------------------------------------------------------------------
// Cutoff suggestion
// ---------------------------------------------------------------------------

/// Suggest a `p_combined` threshold that puts roughly `target_recall` of the
/// validation rejects into the High tier (i.e. `p_combined < threshold`).
/// Returns `None` when there are no reject samples to learn from.
pub fn suggest_high_cutoff(predictions: &[(f64, bool)], target_recall: f64) -> Option<f64> {
    let mut reject_probs: Vec<f64> = predictions
        .iter()
        .filter(|(_, is_rej)| *is_rej)
        .map(|(p, _)| *p)
        .collect();
    if reject_probs.is_empty() {
        return None;
    }
    reject_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target = target_recall.clamp(0.0, 1.0);
    let pos = (target * reject_probs.len() as f64).ceil() as usize;
    let idx = pos.clamp(1, reject_probs.len()) - 1;
    Some(reject_probs[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_high_cutoff_empty() {
        assert!(suggest_high_cutoff(&[], 0.90).is_none());
    }

    #[test]
    fn test_suggest_high_cutoff() {
        let preds = vec![
            (0.1, true),
            (0.2, true),
            (0.3, true),
            (0.4, true),
            (0.5, true),
            (0.6, true),
            (0.7, true),
            (0.8, true),
            (0.9, true),
            (0.95, true),
            (0.1, false),
            (0.2, false),
        ];
        let cutoff = suggest_high_cutoff(&preds, 0.90);
        assert!(cutoff.is_some());
        // 10 rejects, target_recall=0.9 → pos=ceil(9)=9 → idx=8 (value=0.9).
        // This ensures ≥90% of rejects (9 of 10) land below the threshold.
        assert!((cutoff.unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_suggest_high_cutoff_clamp() {
        let preds = vec![(0.1, true), (0.2, true), (0.3, true)];
        let cutoff = suggest_high_cutoff(&preds, 1.0);
        assert!(cutoff.is_some());
        // 3 rejects, 100th percentile clamped to index 2 (0-based), value = 0.3
        assert!((cutoff.unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_fast_auc_perfect() {
        // All positives rank above all negatives → AUC = 1.0.
        let pairs = vec![(0.9, true), (0.8, true), (0.3, false), (0.2, false)];
        assert!((fast_auc(&pairs) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fast_auc_known() {
        // Positives: 0.8, 0.6  | Negatives: 0.7, 0.5
        // (0.8>0.7=1) (0.8>0.5=1) (0.6>0.7=0) (0.6>0.5=1) = 3/4 = 0.75
        let pairs = vec![(0.8, true), (0.6, true), (0.7, false), (0.5, false)];
        assert!((fast_auc(&pairs) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_fast_auc_empty() {
        assert!((fast_auc(&[]) - 0.5).abs() < 1e-9);
    }
}
