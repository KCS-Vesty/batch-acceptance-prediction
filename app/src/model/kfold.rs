//! K-fold cross-validation orchestrator for logistic regression.
//!
//! Splits labeled data into folds, trains per-fold LR models, pools
//! predictions, and accumulates accuracy, AUC, per-reason recall, and
//! feature weights.

use crate::types::LR_FEATURE_NAMES;
use crate::validation::{fast_auc, fold_reject_counts, suggest_high_cutoff, KFoldOutcome};
use super::logistic_regression::LogisticRegression;
use std::collections::HashMap;

pub fn run_lr_kfold_cv(
    labeled: &[crate::types::LabeledRow],
    all_features: &[Vec<f64>],
    labeled_indices: &[usize],
    k: usize,
    epochs: usize,
    lr: f64,
) -> KFoldOutcome {
    // k-fold CV: split labeled_indices into k folds, train on k-1, validate on 1.
    let n = labeled_indices.len();
    if n == 0 {
        return KFoldOutcome {
            accuracy: 0.5,
            auc: 0.5,
            recall_by_reason: HashMap::new(),
            feature_weights: Vec::new(),
            suggested_high_cutoff: None,
        };
    }

    let fold_size = n / k;
    let mut total_correct = 0;
    let mut total_auc_pairs: Vec<(f64, bool)> = Vec::new();
    let mut all_recalls: HashMap<crate::types::RejectReason, (usize, usize)> = HashMap::new();
    let mut avg_weights = vec![0.0; 14];
    let mut fold_count = 0usize;
    // Pool of (predicted_prob, is_reject) across all folds — used to suggest
    // a `p_combined` cutoff that catches the target reject-recall fraction.
    let mut pooled_for_threshold: Vec<(f64, bool)> = Vec::new();

    for fold in 0..k {
        // `[val_start, val_end)` is the held-out validation slice; everything
        // outside that range is the training set for this fold. The last fold
        // absorbs the remainder when `n` doesn't divide evenly by `k`.
        let val_start = fold * fold_size;
        let val_end = if fold == k - 1 {
            n
        } else {
            val_start + fold_size
        };
        if val_end == val_start {
            continue;
        }

        let mut train_labels: Vec<bool> = Vec::new();
        let mut train_feats: Vec<Vec<f64>> = Vec::new();
        let mut val_data: Vec<(usize, bool)> = Vec::new(); // (original_idx, label)

        for (offset, &idx) in labeled_indices.iter().enumerate() {
            if offset >= val_start && offset < val_end {
                val_data.push((idx, labeled[idx].label == crate::types::Label::Accept));
            } else {
                train_labels.push(labeled[idx].label == crate::types::Label::Accept);
                train_feats.push(all_features[idx].clone());
            }
        }

        if train_feats.len() < 5 {
            continue;
        }

        // Train LR.
        let mut lr_model = LogisticRegression::new();
        lr_model.train(&train_labels, &train_feats, lr, epochs);

        // Accumulate weights for averaging.
        fold_count += 1;
        for (avg, w) in avg_weights.iter_mut().zip(lr_model.weights.iter()) {
            *avg += w;
        }

        // Validate fold.
        let mut fold_pairs: Vec<(usize, f64)> = Vec::with_capacity(val_data.len());
        for (val_idx, true_label) in &val_data {
            let prob = lr_model.predict_proba(&all_features[*val_idx]);
            let pred = prob >= 0.5;
            if pred == *true_label {
                total_correct += 1;
            }
            total_auc_pairs.push((prob, *true_label));
            fold_pairs.push((*val_idx, prob));
        }
        let fold_counts = fold_reject_counts(&fold_pairs, labeled);
        for (reason, (tp, actual)) in fold_counts {
            let e = all_recalls.entry(reason).or_insert((0, 0));
            e.0 += tp;
            e.1 += actual;
        }

        // Pool (prob, is_reject) for the cutoff suggestion.
        for (idx, prob) in &fold_pairs {
            pooled_for_threshold.push((*prob, labeled[*idx].label.is_reject()));
        }
    }

    // Compute accuracy.
    let total_preds = total_auc_pairs.len();
    let accuracy = if total_preds > 0 {
        total_correct as f64 / total_preds as f64
    } else {
        0.5
    };

    let auc = fast_auc(&total_auc_pairs);

    // Per-reason recall = pooled TP / pooled actual across folds.
    let mut final_recalls = HashMap::new();
    for (reason, (tp_sum, actual_sum)) in all_recalls {
        final_recalls.insert(
            reason,
            if actual_sum > 0 {
                tp_sum as f64 / actual_sum as f64
            } else {
                0.0
            },
        );
    }

    // Average weights over the folds that actually contributed (skipping the
    // guard at line 218 reduces fold_count so we divide by the real count).
    if fold_count > 0 {
        for w in &mut avg_weights {
            *w /= fold_count as f64;
        }
    }
    let mut avg_pairs: Vec<(String, f64)> = LR_FEATURE_NAMES
        .iter()
        .zip(avg_weights.iter())
        .map(|(&name, &w)| (name.to_string(), w))
        .collect();
    avg_pairs.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let suggested_high_cutoff = suggest_high_cutoff(&pooled_for_threshold, 0.90);

    KFoldOutcome {
        accuracy,
        auc,
        recall_by_reason: final_recalls,
        feature_weights: avg_pairs,
        suggested_high_cutoff,
    }
}
