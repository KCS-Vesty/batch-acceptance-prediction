//! End-to-end pipeline integration test with synthetic data.
//!
//! Creates a temporary dataset with labeled and unprocessed files, runs the
//! full pipeline (file index → parsing → features → model → scoring), and
//! verifies the output structure and basic sanity invariants.
//!
//! The synthetic data is designed to produce differentiable feature patterns
//! between accept and reject classes so the logistic regression converges to
//! something beyond its initial state.

use crate::pipeline;
use crate::types::*;
use std::path::Path;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a LabelMe JSON string with shapes that encode either a clean
/// geometry (accept) or a messy one (reject) so the LR picks up signal.
fn mk_json(is_accept: bool) -> String {
    if is_accept {
        r#"{
  "version": "5.0",
  "flags": {},
  "imagePath": "x.jpg",
  "imageHeight": 200,
  "imageWidth": 200,
  "shapes": [
    {
      "label": "text",
      "shape_type": "rectangle",
      "points": [[10.0, 10.0], [180.0, 60.0], [180.0, 60.0], [10.0, 60.0]],
      "direction": 0.0,
      "group_id": null
    }
  ]
}"#
    } else {
        r#"{
  "version": "5.0",
  "flags": {},
  "imagePath": "x.jpg",
  "imageHeight": 200,
  "imageWidth": 200,
  "shapes": [
    {
      "label": "text",
      "shape_type": "rectangle",
      "points": [[5.0, 5.0], [80.0, 30.0], [80.0, 30.0], [5.0, 30.0]],
      "direction": 0.8,
      "group_id": null
    },
    {
      "label": "text",
      "shape_type": "rectangle",
      "points": [[100.0, 40.0], [195.0, 80.0], [195.0, 80.0], [100.0, 80.0]],
      "direction": 0.3,
      "group_id": null
    },
    {
      "label": "text",
      "shape_type": "rectangle",
      "points": [[20.0, 100.0], [120.0, 140.0], [120.0, 140.0], [20.0, 140.0]],
      "direction": 0.0,
      "group_id": null
    }
  ]
}"#
    }
    .to_string()
}

/// Create a small white JPEG with a black rectangular border so Sobel
/// gradients are non-zero when image features are computed.
fn mk_jpg(path: &Path) {
    let w = 200u32;
    let h = 200u32;
    let mut img = image::RgbImage::new(w, h);
    // White background
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, image::Rgb([255, 255, 255]));
        }
    }
    // Black border — creates strong horizontal & vertical edges
    for x in 0..w {
        img.put_pixel(x, 0, image::Rgb([0, 0, 0]));
        img.put_pixel(x, h - 1, image::Rgb([0, 0, 0]));
    }
    for y in 0..h {
        img.put_pixel(0, y, image::Rgb([0, 0, 0]));
        img.put_pixel(w - 1, y, image::Rgb([0, 0, 0]));
    }
    img.save(path).expect("saving test JPEG");
}

/// Write a review-log .txt with a single status line.
fn mk_txt(path: &Path, status: &str) {
    let content = format!("15/05/2026 14:00:00-{}", status);
    std::fs::write(path, content).expect("writing test .txt");
}

// ---------------------------------------------------------------------------
// Builder for synthetic datasets
// ---------------------------------------------------------------------------

struct DatasetSpec {
    /// How many labeled accept files.
    n_accept: usize,
    /// How many labeled reject files.
    n_reject: usize,
    /// How many unprocessed (no .txt) files.
    n_unprocessed: usize,
    /// Batch folder name — must match the batch_re pattern.
    batch_folder_prefix: &'static str,
    /// Batch number for conditional prior.
    batch_num: i32,
}

/// Populate a temp directory with synthetic data per `spec`.
fn create_dataset(root: &Path, spec: &DatasetSpec) {
    let batch_dir = root.join(format!(
        "{}_{}",
        spec.batch_folder_prefix, spec.batch_num
    ));
    std::fs::create_dir_all(&batch_dir).expect("creating batch dir");

    let subjects = [
        "economics", "history", "science", "math", "english",
    ];
    let doctypes = [
        "homework", "quiz", "exam", "worksheet", "essay",
    ];
    let reject_statuses = [
        "reject-whitespace",
        "reject-rotation",
        "reject-structure",
        "cut off",
        "reject",
    ];

    let mut idx = 0usize;

    // Labeled accept files
    for _ in 0..spec.n_accept {
        let subj = subjects[idx % subjects.len()];
        let doc = doctypes[(idx / 2) % doctypes.len()];
        let stem = format!(
            "padded_china_junior-high-school_grade-1_{}_{}_20250804180723{:04}_train_para1",
            subj, doc, idx + 10
        );
        write_file_trio(&batch_dir, &stem, true, Some("accept"), subj, doc);
        idx += 1;
    }

    // Labeled reject files
    for i in 0..spec.n_reject {
        let subj = subjects[idx % subjects.len()];
        let doc = doctypes[(idx / 2) % doctypes.len()];
        let stem = format!(
            "padded_china_junior-high-school_grade-1_{}_{}_20250804180723{:04}_train_para1",
            subj, doc, idx + 10
        );
        let status = reject_statuses[i % reject_statuses.len()];
        write_file_trio(&batch_dir, &stem, false, Some(status), subj, doc);
        idx += 1;
    }

    // Unprocessed files (no .txt)
    for _ in 0..spec.n_unprocessed {
        let subj = subjects[(idx + 2) % subjects.len()];
        let doc = doctypes[(idx + 3) % doctypes.len()];
        let stem = format!(
            "padded_china_junior-high-school_grade-1_{}_{}_20250804180724{:04}_train_para1",
            subj, doc, idx
        );
        write_file_trio(&batch_dir, &stem, true, None::<&str>, subj, doc);
        idx += 1;
    }
}

/// Write .json + .jpg (+ optional .txt) for one file stem.
fn write_file_trio(
    dir: &Path,
    stem: &str,
    is_accept: bool,
    txt_status: Option<&str>,
    _subject: &str,
    _doctype: &str,
) {
    // JSON annotation
    let json_content = mk_json(is_accept);
    std::fs::write(dir.join(format!("{}.json", stem)), json_content)
        .expect("writing test JSON");

    // JPEG image
    mk_jpg(&dir.join(format!("{}.jpg", stem)));

    // Optional review log
    if let Some(status) = txt_status {
        mk_txt(&dir.join(format!("{}.txt", stem)), status);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_synthetic_dataset() {
    let temp_dir = std::env::temp_dir().join("bap_integration_test");
    // Clean slate
    let _ = std::fs::remove_dir_all(&temp_dir);

    let spec = DatasetSpec {
        n_accept: 5,
        n_reject: 5,
        n_unprocessed: 3,
        batch_folder_prefix: "POC_P2_20000",
        batch_num: 99,
    };
    create_dataset(&temp_dir, &spec);

    // Capture all log messages the pipeline emits.
    let mut log_lines: Vec<String> = Vec::new();

    let cfg = PipelineConfig {
        alpha: 3.0,
        lr_rate: 0.1,
        lr_epochs: 300,
        cv_folds: 3,
        cv_epochs: 200,
    };

    let result = pipeline::run(&temp_dir, &cfg, |msg| {
        log_lines.push(msg.to_string())
    })
    .expect("pipeline::run should succeed");

    // ── Basic counts ──────────────────────────────────────────────────────
    assert_eq!(
        result.n_labeled, 10,
        "all .json + .txt pairs should be parsed as labeled"
    );
    assert_eq!(result.n_accept, 5, "5 accept labels");
    assert_eq!(result.n_reject, 5, "5 reject labels");
    assert_eq!(
        result.n_unprocessed, 3,
        "all .json-only files should count as unprocessed"
    );
    assert_eq!(
        result.n_ambiguous, 0,
        "no files with bad .txt content"
    );

    // ── Predictions ───────────────────────────────────────────────────────
    assert_eq!(
        result.predictions.len(),
        3,
        "one prediction per unprocessed file"
    );
    assert_eq!(
        result.lr_scored,
        3,
        "all unprocessed files should be LR-scored"
    );

    for (i, pred) in result.predictions.iter().enumerate() {
        assert!(
            !pred.filename.is_empty(),
            "prediction {} should have a filename",
            i
        );
        assert_eq!(pred.batch, 99, "batch from folder name");
        assert!(
            pred.p_categorical > 0.0,
            "p_categorical > 0 for prediction {} — got {}",
            i,
            pred.p_categorical
        );
        assert!(
            pred.p_categorical <= 1.0,
            "p_categorical <= 1 for prediction {}",
            i
        );
        assert!(
            pred.p_combined > 0.0,
            "p_combined > 0 for prediction {}",
            i
        );
        assert!(
            pred.p_combined <= 1.0,
            "p_combined <= 1 for prediction {}",
            i
        );
        // p_logreg is Some when LR scoring succeeded
        assert!(
            pred.p_logreg.is_some(),
            "p_logreg should be Some for prediction {}",
            i
        );
        assert!(
            !pred.manually_overridden,
            "no manual override in synthetic data"
        );
    }

    // ── Sorted p_combined ─────────────────────────────────────────────────
    assert_eq!(result.sorted_p_combined.len(), 3);
    assert!(
        result.sorted_p_combined.windows(2).all(|w| w[0] <= w[1]),
        "sorted_p_combined should be ascending"
    );

    // ── Validation ────────────────────────────────────────────────────────
    assert!(
        result.validation.auc >= 0.0 && result.validation.auc <= 1.0,
        "AUC should be in [0, 1]"
    );
    assert!(
        result.validation.accuracy >= 0.0 && result.validation.accuracy <= 1.0,
        "accuracy should be in [0, 1]"
    );
    // With 10 labeled rows × 3 CV folds, at least some folds should train.
    assert!(
        result.lr_trained_on >= 5,
        "at least 5 labeled rows used for LR training"
    );

    // Feature weights should be present (14 features).
    assert_eq!(
        result.validation.feature_weights.len(),
        14,
        "all 14 feature weights should be reported"
    );
    let top_feature = &result.validation.feature_weights[0];
    assert!(
        !top_feature.0.is_empty(),
        "top feature should have a name"
    );

    // ── Log messages ──────────────────────────────────────────────────────
    let log_text = log_lines.join(" ");
    assert!(
        log_text.contains("Scanning dataset"),
        "log should contain pipeline progress messages"
    );
    assert!(
        log_text.contains("Labeled:"),
        "log should show labeled counts"
    );

    // ── Cleanup ───────────────────────────────────────────────────────────
    std::fs::remove_dir_all(&temp_dir).expect("cleanup temp dir");
}

// ── Edge case: no labeled files ──────────────────────────────────────────

#[test]
fn pipeline_requires_labeled_data() {
    let temp_dir = std::env::temp_dir().join("bap_integration_test_no_labeled");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Only unprocessed files, nothing labeled.
    let spec = DatasetSpec {
        n_accept: 0,
        n_reject: 0,
        n_unprocessed: 2,
        batch_folder_prefix: "POC_P2_20000",
        batch_num: 98,
    };
    create_dataset(&temp_dir, &spec);

    let cfg = PipelineConfig::default();
    let result = pipeline::run(&temp_dir, &cfg, |_| {});

    assert!(
        result.is_err(),
        "pipeline should error when no labeled files exist"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No labeled"),
        "error message should mention 'No labeled': '{}'",
        err
    );

    std::fs::remove_dir_all(&temp_dir).expect("cleanup temp dir");
}

// ── Edge case: no unprocessed files ──────────────────────────────────────

#[test]
fn pipeline_requires_unprocessed_data() {
    let temp_dir = std::env::temp_dir().join("bap_integration_test_no_unproc");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let spec = DatasetSpec {
        n_accept: 3,
        n_reject: 2,
        n_unprocessed: 0,
        batch_folder_prefix: "POC_P2_20000",
        batch_num: 97,
    };
    create_dataset(&temp_dir, &spec);

    let cfg = PipelineConfig::default();
    let result = pipeline::run(&temp_dir, &cfg, |_| {});

    assert!(
        result.is_err(),
        "pipeline should error when no unprocessed files exist"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No unprocessed"),
        "error message should mention 'No unprocessed': '{}'",
        err
    );

    std::fs::remove_dir_all(&temp_dir).expect("cleanup temp dir");
}

// ── Edge case: manual-override txt ───────────────────────────────────────

#[test]
fn pipeline_detects_manual_overrides() {
    let temp_dir = std::env::temp_dir().join("bap_integration_test_manual");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create one labeled file with manual-* status
    let batch_dir = temp_dir.join("POC_P2_20000_96");
    std::fs::create_dir_all(&batch_dir).expect("creating batch dir");

    let stem = "padded_china_junior-high-school_grade-1_math_homework_20250804180723001_train_para1";

    // JSON + JPG
    std::fs::write(
        batch_dir.join(format!("{}.json", stem)),
        mk_json(true),
    )
    .expect("writing JSON");
    mk_jpg(&batch_dir.join(format!("{}.jpg", stem)));

    // Status: accept initially, then manual-reject on a new line.
    let txt_content =
        "15/05/2026 10:00:00-accept\n15/05/2026 11:00:00-manual-reject-whitespace\n";
    std::fs::write(batch_dir.join(format!("{}.txt", stem)), txt_content)
        .expect("writing .txt");

    // Also one unprocessed file so the pipeline has something to predict.
    let stem2 = "padded_china_junior-high-school_grade-1_english_essay_20250804180723002_train_para1";
    std::fs::write(
        batch_dir.join(format!("{}.json", stem2)),
        mk_json(true),
    )
    .expect("writing JSON");
    mk_jpg(&batch_dir.join(format!("{}.jpg", stem2)));

    let cfg = PipelineConfig::default();
    let result = pipeline::run(&temp_dir, &cfg, |_| {}).expect("pipeline should succeed");

    // The labeled file was accepted then manually-overridden to reject.
    // The prediction for the unprocessed file should NOT be flagged.
    assert_eq!(result.predictions.len(), 1);
    assert!(
        !result.predictions[0].manually_overridden,
        "unprocessed file should not be manually overridden"
    );

    std::fs::remove_dir_all(&temp_dir).expect("cleanup temp dir");
}

// ── PipelineConfig hyperparameter round-trip ─────────────────────────────

#[test]
fn pipeline_config_default_values_are_sensible() {
    let cfg = PipelineConfig::default();
    assert_eq!(cfg.alpha, 3.0, "smoothing alpha");
    assert_eq!(cfg.lr_rate, 0.1, "learning rate");
    assert_eq!(cfg.lr_epochs, 300, "LR epochs");
    assert_eq!(cfg.cv_folds, 5, "CV folds");
    assert_eq!(cfg.cv_epochs, 200, "CV epochs");
}
