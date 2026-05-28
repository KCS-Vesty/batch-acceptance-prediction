//! Core domain types shared across the application.
//!
//! Centralising these here eliminates the circular-ish dependency where
//! `pipeline.rs` had to own the type definitions just because the `run`
//! orchestrator returns them, while every other module imported them from
//! `crate::pipeline`. Now `pipeline.rs` can focus solely on orchestration.

pub const IMAGE_SIZE: f64 = 640.0;

/// Feature names for the 14-element LR vector.
/// Indices 0-6: JSON geometry. Indices 7-12: image-derived (see
/// `crate::image_features::aggregate_arrays`). Index 13: prior from
/// `model::ConditionalProbModel`.
pub const LR_FEATURE_NAMES: [&str; 14] = [
    "n_shapes",
    "coverage",
    "avg_area",
    "std_area",
    "avg_width",
    "avg_height",
    "dir_std",
    "cutoff_mean",
    "cutoff_max",
    "isolation_mean",
    "misalign_mean",
    "misalign_max",
    "ink_contrast_mean",
    "p_categorical",
];

// ---------------------------------------------------------------------------
// File index
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub stem: std::path::PathBuf,
    pub json: Option<std::path::PathBuf>,
    pub jpg: Option<std::path::PathBuf>,
    pub txt: Option<std::path::PathBuf>,
}

// ---------------------------------------------------------------------------
// Metadata & labels
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Meta {
    pub stem: std::path::PathBuf,
    pub grade: String,
    pub subject: String,
    pub doc_type: String,
    pub batch: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectReason {
    Whitespace,
    Rotation,
    Structure,
    CutOff,
    Generic,
    Manual,
}

impl RejectReason {
    /// Stable display order so per-reason chips don't reshuffle every run.
    /// Ordering matches reviewer mental model (most→least specific).
    #[inline]
    pub fn display_order(self) -> u8 {
        match self {
            RejectReason::Whitespace => 0,
            RejectReason::Rotation => 1,
            RejectReason::CutOff => 2,
            RejectReason::Structure => 3,
            RejectReason::Generic => 4,
            RejectReason::Manual => 5,
        }
    }
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::Whitespace => write!(f, "Whitespace"),
            RejectReason::Rotation => write!(f, "Rotation"),
            RejectReason::CutOff => write!(f, "Cutoff"),
            RejectReason::Structure => write!(f, "Structure"),
            RejectReason::Generic => write!(f, "Generic"),
            RejectReason::Manual => write!(f, "Manual"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Accept,
    Reject(RejectReason),
}

impl Label {
    pub fn is_reject(&self) -> bool {
        matches!(self, Label::Reject(_))
    }
}

#[derive(Debug, Clone)]
pub struct LabeledRow {
    pub meta: Meta,
    pub label: Label,
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// Column that the predictions table is currently sorted by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortCol {
    Risk,
    Batch,
    Subject,
    Grade,
    DocType,
    PCategorical,
    PLogReg,
    #[default]
    PCombined,
    Filename,
}

// ---------------------------------------------------------------------------
// Risk tier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskTier {
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------------------
// Prediction & results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Prediction {
    pub filename: String,
    pub batch: i32,
    pub grade: String,
    pub subject: String,
    pub doc_type: String,
    pub p_categorical: f64,
    pub p_logreg: Option<f64>,
    pub p_combined: f64,
    pub json_path: std::path::PathBuf,
    pub jpg_path: Option<std::path::PathBuf>,
    pub txt_path: Option<std::path::PathBuf>,
    /// True when the most recent status line in the `.txt` review log is a
    /// `manual-*` override written from the in-app editor.
    pub manually_overridden: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Additive-smoothing alpha for the conditional prior model.
    pub alpha: f64,
    /// Learning rate for final logistic regression training.
    pub lr_rate: f64,
    /// Gradient-descent epochs for final LR training.
    pub lr_epochs: usize,
    /// Number of cross-validation folds.
    pub cv_folds: usize,
    /// Gradient-descent epochs per fold during CV.
    pub cv_epochs: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            alpha: 3.0,
            lr_rate: 0.1,
            lr_epochs: 300,
            cv_folds: 5,
            cv_epochs: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub accuracy: f64,
    pub auc: f64,
    pub recall_by_reason: std::collections::HashMap<RejectReason, f64>,
    /// (feature_name, weight) pairs sorted by absolute |weight| descending.
    pub feature_weights: Vec<(String, f64)>,
    /// `p_combined` threshold for ~90% reject recall. None when no rejects.
    pub suggested_high_cutoff: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub predictions: Vec<Prediction>,
    pub sorted_p_combined: Vec<f64>,
    pub validation: ValidationResult,
    pub n_labeled: usize,
    pub n_accept: usize,
    pub n_reject: usize,
    pub n_ambiguous: usize,
    pub n_unprocessed: usize,
    pub lr_trained_on: usize,
    pub lr_scored: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reject_reason_display() {
        assert_eq!(RejectReason::Whitespace.to_string(), "Whitespace");
        assert_eq!(RejectReason::Rotation.to_string(), "Rotation");
        assert_eq!(RejectReason::CutOff.to_string(), "Cutoff");
        assert_eq!(RejectReason::Structure.to_string(), "Structure");
        assert_eq!(RejectReason::Generic.to_string(), "Generic");
        assert_eq!(RejectReason::Manual.to_string(), "Manual");
    }

    #[test]
    fn test_reject_reason_display_order() {
        use RejectReason::*;
        let reasons = [Whitespace, Rotation, CutOff, Structure, Generic, Manual];
        for w in reasons.windows(2) {
            assert!(w[0].display_order() < w[1].display_order());
        }
    }
}
