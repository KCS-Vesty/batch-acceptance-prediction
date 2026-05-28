//! 14-feature logistic regression with L2 regularisation, z-score
//! standardisation, and per-class re-weighting for imbalanced data.

/// 14-feature logistic regression trained via batch gradient descent with
/// L2 regularisation, per-column z-score standardisation, and per-class
/// re-weighting to handle the ~17× accept/reject imbalance.
#[derive(Debug)]
pub struct LogisticRegression {
    pub(crate) weights: Vec<f64>,
    bias: f64,
    /// Per-feature mean computed from the training set; used to standardise
    /// both training and prediction-time inputs.
    mu: Vec<f64>,
    /// Per-feature standard deviation (≥ 1e-6 to avoid division by zero).
    sigma: Vec<f64>,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            weights: vec![0.0; 14],
            bias: 0.0,
            mu: vec![0.0; 14],
            sigma: vec![1.0; 14],
        }
    }

    pub(crate) fn sigmoid(z: f64) -> f64 {
        if z > 20.0 {
            1.0
        } else if z < -20.0 {
            0.0
        } else {
            1.0 / (1.0 + (-z).exp())
        }
    }

    /// Train on `labels` (true = accept) and `features` (14-dim per row).
    /// Computes mu/sigma from `features`, standardises, then runs L2 batch
    /// gradient descent with class re-weighting on the minority class.
    pub fn train(&mut self, labels: &[bool], features: &[Vec<f64>], lr: f64, epochs: usize) {
        let n = labels.len();
        if n == 0 || features.is_empty() {
            return;
        }
        let dim = self.weights.len();

        // ── Standardisation parameters ──────────────────────────────────────
        for (j, mu_j) in self.mu.iter_mut().enumerate() {
            let s: f64 = features.iter().map(|x| x[j]).sum();
            *mu_j = s / n as f64;
        }
        for (j, sigma_j) in self.sigma.iter_mut().enumerate() {
            let mu = self.mu[j];
            let sq: f64 = features.iter().map(|x| (x[j] - mu).powi(2)).sum();
            *sigma_j = (sq / n as f64).sqrt().max(1e-6);
        }
        let x_std: Vec<Vec<f64>> = features
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.mu.iter().zip(self.sigma.iter()))
                    .map(|(x, (mu, sigma))| (x - mu) / sigma)
                    .collect()
            })
            .collect();

        // ── Class weights: balance the loss across imbalanced classes ──────
        let n_pos = labels.iter().filter(|&&l| l).count();
        let n_neg = n - n_pos;
        let (w_pos, w_neg) = if n_pos == 0 || n_neg == 0 {
            (1.0, 1.0)
        } else if n_pos < n_neg {
            (n_neg as f64 / n_pos as f64, 1.0)
        } else {
            (1.0, n_pos as f64 / n_neg as f64)
        };

        // ── Batch gradient descent ──────────────────────────────────────────
        let lambda = L2_LAMBDA;
        for _epoch in 0..epochs {
            let mut grad_w = vec![0.0; dim];
            let mut grad_b = 0.0;
            let mut total_w = 0.0;

            for (&label, x_row) in labels.iter().zip(x_std.iter()) {
                let y = if label { 1.0 } else { 0.0 };
                let w = if label { w_pos } else { w_neg };

                let mut z = self.bias;
                for (wj, xj) in self.weights.iter().zip(x_row.iter()) {
                    z += wj * xj;
                }
                let err = Self::sigmoid(z) - y;

                for (gw, xj) in grad_w.iter_mut().zip(x_row.iter()) {
                    *gw += w * err * xj;
                }
                grad_b += w * err;
                total_w += w;
            }

            let inv = 1.0 / total_w.max(1.0);
            for (w, gw) in self.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * (gw * inv + 2.0 * lambda * *w);
            }
            self.bias -= lr * grad_b * inv;
        }
    }

    /// Predict P(accept) for a single 14-feature vector. The input is
    /// standardised using the mu/sigma captured at training time.
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        if features.len() != self.weights.len() {
            return 0.5;
        }
        let mut z = self.bias;
        for (((w, mu), sigma), x) in self
            .weights
            .iter()
            .zip(&self.mu)
            .zip(&self.sigma)
            .zip(features)
        {
            z += w * (x - mu) / sigma;
        }
        Self::sigmoid(z)
    }
}

/// L2 regularisation constant for logistic regression.
/// Kept small (0.01) so the model can still fit the data while preventing
/// extreme weight magnitudes on small training sets.
pub const L2_LAMBDA: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        // sigmoid(0.0) == 0.5 exactly
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-9);
        // sigmoid(20.0) is very close to 1.0; allow generous tolerance for float precision
        assert!(LogisticRegression::sigmoid(20.0) > 0.999999);
        assert!(LogisticRegression::sigmoid(20.0) <= 1.0);
        // sigmoid(-20.0) is very close to 0.0
        assert!(LogisticRegression::sigmoid(-20.0) < 1e-6);
        assert!(LogisticRegression::sigmoid(-20.0) >= 0.0);
        // Monotonicity check
        assert!(LogisticRegression::sigmoid(-1.0) < LogisticRegression::sigmoid(1.0));
    }

    // ── LogisticRegression training convergence ─────────────────────────────

    #[test]
    fn test_lr_training_converges_on_separable_data() {
        // Create perfectly separable data: accept if feature[0] > 0.5
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..20 {
            features.push(vec![
                0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]);
            labels.push(true);
            features.push(vec![
                0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]);
            labels.push(false);
        }
        let mut lr = LogisticRegression::new();
        lr.train(&labels, &features, 0.1, 500);
        // After training, predictions should be confident
        let p_accept = lr.predict_proba(&features[0]);
        let p_reject = lr.predict_proba(&features[1]);
        assert!(
            p_accept > 0.9,
            "accept prob should be high, got {}",
            p_accept
        );
        assert!(
            p_reject < 0.1,
            "reject prob should be low, got {}",
            p_reject
        );
    }

    #[test]
    fn test_lr_training_empty_data() {
        let mut lr = LogisticRegression::new();
        lr.train(&[], &[], 0.1, 100);
        // Weights should remain at initial values
        assert!(lr.weights.iter().all(|&w| w == 0.0));
        assert_eq!(lr.bias, 0.0);
    }

    #[test]
    fn test_lr_predict_proba_wrong_dim() {
        let lr = LogisticRegression::new();
        // Wrong number of features → returns 0.5
        assert!((lr.predict_proba(&[0.5, 0.5]) - 0.5).abs() < 1e-9);
    }
}
