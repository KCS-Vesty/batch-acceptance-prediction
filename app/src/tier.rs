use crate::types::RiskTier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierMethod {
    /// Fixed absolute thresholds on `p_combined`.
    Absolute,
    /// Bottom `pct_high` % of `p_combined` is High, top `pct_low` % is Low,
    /// rest is Medium. Always produces differentiation regardless of base rate.
    Percentile,
}

impl std::fmt::Display for TierMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TierMethod::Absolute => write!(f, "Absolute"),
            TierMethod::Percentile => write!(f, "Percentile"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TierConfig {
    pub method: TierMethod,
    /// Absolute mode: p_combined >= this is Low. Default 0.97.
    pub abs_low: f64,
    /// Absolute mode: p_combined < this is High. Default 0.92.
    pub abs_high: f64,
    /// Percentile mode: bottom this % is High. Default 20.
    pub pct_high: f64,
    /// Percentile mode: top this % is Low. Default 50.
    pub pct_low: f64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            method: TierMethod::Percentile,
            abs_low: 0.97,
            abs_high: 0.92,
            pct_high: 20.0,
            pct_low: 50.0,
        }
    }
}

/// Numeric cutoffs derived from `TierConfig` and (for Percentile mode) the
/// sorted distribution of `p_combined`.
#[derive(Debug, Clone, Copy)]
pub struct Cutoffs {
    /// `p_combined < high_cutoff` ⇒ High risk.
    pub high_cutoff: f64,
    /// `p_combined >= low_cutoff` ⇒ Low risk. Otherwise Medium.
    pub low_cutoff: f64,
}

impl Cutoffs {
    pub const FALLBACK: Cutoffs = Cutoffs {
        high_cutoff: 0.92,
        low_cutoff: 0.97,
    };
}

impl TierConfig {
    /// `sorted_p_combined` must be ascending. If empty (e.g. before analysis),
    /// returns the absolute defaults so the UI still has something sensible.
    pub fn resolve(&self, sorted_p_combined: &[f64]) -> Cutoffs {
        match self.method {
            TierMethod::Absolute => Cutoffs {
                high_cutoff: self.abs_high,
                low_cutoff: self.abs_low,
            },
            TierMethod::Percentile => {
                if sorted_p_combined.is_empty() {
                    return Cutoffs::FALLBACK;
                }
                let n = sorted_p_combined.len();
                // High cutoff = value at percentile `pct_high` (bottom slice).
                // anything STRICTLY below this value is High.
                let high_idx = ((self.pct_high.clamp(0.0, 100.0) / 100.0)
                    * n as f64)
                    .round() as usize;
                let high_idx = high_idx.min(n.saturating_sub(1));
                let high_val = sorted_p_combined[high_idx];
                // Low cutoff = value at percentile `100 - pct_low` (so top `pct_low`%
                // of values are >= this).
                let low_idx = (((100.0 - self.pct_low.clamp(0.0, 100.0)) / 100.0)
                    * n as f64)
                    .round() as usize;
                let low_idx = low_idx.min(n.saturating_sub(1));
                let low_val = sorted_p_combined[low_idx];
                // Guarantee high_cutoff <= low_cutoff so the Medium band is non-empty
                // when pct_high + pct_low <= 100. If the user set overlapping percentiles
                // (e.g. pct_high=60, pct_low=60) the swap silently fixes the ordering;
                // the side-panel warning covers this visually.
                let (high_cutoff, low_cutoff) = if high_val <= low_val {
                    (high_val, low_val)
                } else {
                    (low_val, high_val)
                };
                Cutoffs {
                    high_cutoff,
                    low_cutoff,
                }
            }
        }
    }
}

#[inline]
pub fn classify(p_combined: f64, c: Cutoffs) -> RiskTier {
    if p_combined >= c.low_cutoff {
        RiskTier::Low
    } else if p_combined < c.high_cutoff {
        RiskTier::High
    } else {
        RiskTier::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_low() {
        let c = Cutoffs { high_cutoff: 0.3, low_cutoff: 0.7 };
        assert_eq!(classify(0.7, c), RiskTier::Low);
        assert_eq!(classify(0.99, c), RiskTier::Low);
        assert_eq!(classify(1.0, c), RiskTier::Low);
    }

    #[test]
    fn test_classify_high() {
        let c = Cutoffs { high_cutoff: 0.3, low_cutoff: 0.7 };
        assert_eq!(classify(0.0, c), RiskTier::High);
        assert_eq!(classify(0.29, c), RiskTier::High);
        // Exactly at high_cutoff → Medium (strict less-than)
        assert_ne!(classify(0.3, c), RiskTier::High);
    }

    #[test]
    fn test_classify_medium() {
        let c = Cutoffs { high_cutoff: 0.3, low_cutoff: 0.7 };
        assert_eq!(classify(0.3, c), RiskTier::Medium);
        assert_eq!(classify(0.5, c), RiskTier::Medium);
        assert_eq!(classify(0.69, c), RiskTier::Medium);
    }

    #[test]
    fn test_classify_boundary_exact() {
        // At the exact low_cutoff → Low (>=)
        let c = Cutoffs { high_cutoff: 0.3, low_cutoff: 0.7 };
        assert_eq!(classify(0.7, c), RiskTier::Low);
        // Just below low_cutoff → Medium
        assert_eq!(classify(0.699, c), RiskTier::Medium);
    }

    #[test]
    fn test_tier_config_default() {
        let cfg = TierConfig::default();
        assert_eq!(cfg.method, TierMethod::Percentile);
        assert_eq!(cfg.abs_low, 0.97);
        assert_eq!(cfg.abs_high, 0.92);
        assert_eq!(cfg.pct_high, 20.0);
        assert_eq!(cfg.pct_low, 50.0);
    }

    #[test]
    fn test_resolve_absolute() {
        let cfg = TierConfig {
            method: TierMethod::Absolute,
            abs_low: 0.8,
            abs_high: 0.2,
            ..Default::default()
        };
        let c = cfg.resolve(&[0.1, 0.5, 0.9]);
        assert_eq!(c.high_cutoff, 0.2);
        assert_eq!(c.low_cutoff, 0.8);
    }

    #[test]
    fn test_resolve_percentile() {
        // 10 values: [0.0, 0.1, 0.2, ..., 0.9]
        let sorted: Vec<f64> = (0..10).map(|i| i as f64 / 10.0).collect();
        let cfg = TierConfig {
            method: TierMethod::Percentile,
            pct_high: 20.0,
            pct_low: 50.0,
            ..Default::default()
        };
        let c = cfg.resolve(&sorted);
        // Bottom 20% → index 2 → value 0.2
        assert!((c.high_cutoff - 0.2).abs() < 1e-9);
        // Top 50% → index 5 → value 0.5
        assert!((c.low_cutoff - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_percentile_empty() {
        let cfg = TierConfig {
            method: TierMethod::Percentile,
            ..Default::default()
        };
        let c = cfg.resolve(&[]);
        assert_eq!(c.high_cutoff, Cutoffs::FALLBACK.high_cutoff);
        assert_eq!(c.low_cutoff, Cutoffs::FALLBACK.low_cutoff);
    }

    #[test]
    fn test_resolve_percentile_single_element() {
        let cfg = TierConfig {
            method: TierMethod::Percentile,
            pct_high: 20.0,
            pct_low: 50.0,
            ..Default::default()
        };
        let c = cfg.resolve(&[0.5]);
        // Both cutoffs resolve to the single element
        assert!((c.high_cutoff - 0.5).abs() < 1e-9);
        assert!((c.low_cutoff - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_resolve_percentile_clamps_pct() {
        let sorted: Vec<f64> = (0..10).map(|i| i as f64 / 10.0).collect();
        let cfg = TierConfig {
            method: TierMethod::Percentile,
            pct_high: 150.0, // clamped to 100
            pct_low: 150.0,  // clamped to 100
            ..Default::default()
        };
        let c = cfg.resolve(&sorted);
        // pct_high=100 → high_idx = min(9, 9) = 9 → value 0.9
        // pct_low=100 → low_idx = min(9, 0) = 0 → value 0.0
        // high_val (0.9) > low_val (0.0), so they swap:
        // high_cutoff = 0.0, low_cutoff = 0.9
        assert!((c.high_cutoff - 0.0).abs() < 1e-9);
        assert!((c.low_cutoff - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_fallback_cutoffs() {
        assert_eq!(Cutoffs::FALLBACK.high_cutoff, 0.92);
        assert_eq!(Cutoffs::FALLBACK.low_cutoff, 0.97);
    }
}
