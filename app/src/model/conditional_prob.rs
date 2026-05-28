//! Bayesian prior model for per-(batch, subject) acceptance rate.
//!
//! Uses additive smoothing (alpha-weighted) with a global acceptance rate
//! fallback. Provides the `p_categorical` feature (index 13 in the LR vector).

use std::collections::HashMap;

/// A simple Bayesian prior model for the per-(batch, subject) acceptance rate.
///
/// Uses additive smoothing (alpha-weighted) with a global acceptance rate
/// fallback. This gives the `p_categorical` feature — index 13 in the
/// 14-element feature vector.
#[derive(Debug)]
pub struct ConditionalProbModel {
    pub alpha: f64,
    pub global_rate: f64,
    cells: HashMap<(i32, String), (u32, u32)>,
}

impl ConditionalProbModel {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            global_rate: 0.943,
            cells: HashMap::new(),
        }
    }

    pub fn fit(&mut self, labeled: &[crate::types::LabeledRow]) {
        let n_acc = labeled
            .iter()
            .filter(|r| r.label == crate::types::Label::Accept)
            .count();
        self.global_rate = if labeled.is_empty() {
            0.943
        } else {
            n_acc as f64 / labeled.len() as f64
        };
        for r in labeled {
            let k = (r.meta.batch, r.meta.subject.clone());
            let e = self.cells.entry(k).or_insert((0, 0));
            e.1 += 1;
            if r.label == crate::types::Label::Accept {
                e.0 += 1;
            }
        }
    }

    pub fn predict_proba(&self, batch: i32, subject: &str) -> f64 {
        let key = (batch, subject.to_string());
        let (n_acc, n_tot) = self.cells.get(&key).copied().unwrap_or((0, 0));
        (n_acc as f64 + self.alpha * self.global_rate) / (n_tot as f64 + self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Label, LabeledRow, Meta};

    fn make_labeled_row(batch: i32, subject: &str, label: Label) -> LabeledRow {
        LabeledRow {
            meta: Meta {
                stem: std::path::PathBuf::from(format!("{}_{}", batch, subject)),
                grade: "grade-1".into(),
                subject: subject.into(),
                doc_type: "homework".into(),
                batch,
            },
            label,
        }
    }

    #[test]
    fn test_conditional_prob_model_perfect_predictor() {
        // 10 accepts for batch 1 / math, 0 rejects
        let rows: Vec<LabeledRow> = (0..10)
            .map(|_| make_labeled_row(1, "math", Label::Accept))
            .collect();
        let mut model = ConditionalProbModel::new(3.0);
        model.fit(&rows);
        let p = model.predict_proba(1, "math");
        // (10 + 3*0.943) / (10 + 3) ≈ 0.988
        assert!(p > 0.95);
        assert!(p <= 1.0);
    }

    #[test]
    fn test_conditional_prob_model_unseen_cell() {
        let rows = vec![make_labeled_row(1, "math", Label::Accept)];
        let mut model = ConditionalProbModel::new(3.0);
        model.fit(&rows);
        // Unseen (batch, subject) → falls back to global_rate
        let p = model.predict_proba(99, "unknown");
        // (0 + 3*global_rate) / (0 + 3) = global_rate
        assert!((p - model.global_rate).abs() < 1e-9);
    }

    #[test]
    fn test_conditional_prob_model_smoothing() {
        // All rejects for a cell → should still be pulled toward global_rate
        let rows: Vec<LabeledRow> = (0..5)
            .map(|_| {
                make_labeled_row(
                    1,
                    "math",
                    Label::Reject(crate::types::RejectReason::Generic),
                )
            })
            .collect();
        let mut model = ConditionalProbModel::new(3.0);
        model.fit(&rows);
        let p = model.predict_proba(1, "math");
        // global_rate = 0/5 = 0.0, so (0 + 3*0.0)/(5+3) = 0.0
        assert!((p - 0.0).abs() < 1e-9);
        // A cell with accepts should give higher prob
        let rows2 = vec![
            make_labeled_row(2, "science", Label::Accept),
            make_labeled_row(2, "science", Label::Accept),
            make_labeled_row(
                2,
                "science",
                Label::Reject(crate::types::RejectReason::Generic),
            ),
        ];
        let mut model2 = ConditionalProbModel::new(3.0);
        model2.fit(&rows2);
        let p2 = model2.predict_proba(2, "science");
        // global_rate = 2/3, cell = (2 + 3*2/3)/(3+3) = 4/6 ≈ 0.667
        assert!(p2 > 0.5);
        assert!(p2 < 1.0);
    }

    #[test]
    fn test_conditional_prob_model_empty_fit() {
        let mut model = ConditionalProbModel::new(3.0);
        model.fit(&[]);
        assert_eq!(model.global_rate, 0.943);
        let p = model.predict_proba(1, "math");
        // (0 + 3*0.943) / (0 + 3) = 0.943
        assert!((p - 0.943).abs() < 1e-9);
    }
}
