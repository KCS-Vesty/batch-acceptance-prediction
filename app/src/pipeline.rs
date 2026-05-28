use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::features::build_feature_vector;
use crate::types::*;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Concatenate the three feature groups into the canonical 14-element feature
/// vector used by logistic regression: indices 0-6 = JSON geometry, 7-12 =
/// image-derived, 13 = conditional prior.
// (Re-exported from `crate::features` so both training and scoring paths
//  resolve the same function. The import above keeps the call sites clean.)

// ---------------------------------------------------------------------------
// STEP 1: file index
// ---------------------------------------------------------------------------

pub fn build_file_index(root: &Path) -> HashMap<PathBuf, FileEntry> {
    let mut index: HashMap<PathBuf, FileEntry> = HashMap::new();
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.into_path();
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        let dot = match fname.rfind('.') {
            Some(i) => i,
            None => continue,
        };
        let ext = fname[dot + 1..].to_lowercase();
        if !matches!(ext.as_str(), "json" | "jpg" | "txt") {
            continue;
        }
        let dir = match path.parent() {
            Some(p) => p,
            None => continue,
        };
        let key = dir.join(&fname[..dot]);
        let e = index.entry(key).or_insert_with_key(|k| FileEntry {
            stem: k.clone(),
            json: None,
            jpg: None,
            txt: None,
        });
        match ext.as_str() {
            "json" => e.json = Some(path),
            "jpg" => e.jpg = Some(path),
            "txt" => e.txt = Some(path),
            _ => {}
        }
    }
    index
}

// ---------------------------------------------------------------------------
// Aligned labeled rows — type-safe index coupling
// ---------------------------------------------------------------------------

/// A labeled row paired with its computed features.
/// The `json7` field is `Some` when annotation parsing succeeded;
/// `img6` is `Some` when image feature extraction succeeded.
pub struct LabeledRowWithFeatures<'a> {
    pub row: &'a LabeledRow,
    pub json7: Option<[f64; 7]>,
    pub img6: Option<[f64; 6]>,
}

/// Zip `labeled` with `rows_labeled` by positional index, producing aligned
/// pairs. Panics in debug if lengths mismatch (they come from the same stem list).
pub fn align_labeled_features<'a>(
    labeled: &'a [LabeledRow],
    rows_labeled: &'a [crate::features::RowFeatures],
) -> Vec<LabeledRowWithFeatures<'a>> {
    debug_assert_eq!(
        labeled.len(),
        rows_labeled.len(),
        "labeled and rows_labeled must have the same length"
    );
    labeled
        .iter()
        .zip(rows_labeled.iter())
        .map(|(row, feat)| LabeledRowWithFeatures {
            row,
            json7: feat.json7,
            img6: feat.img6,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Prediction builder
// ---------------------------------------------------------------------------

/// Build a `Prediction` from an unprocessed `Meta`, its LR score, and the
/// file index. Encapsulates the 10-field construction so the pipeline
/// function reads as a high-level recipe.
fn build_prediction(
    meta: Meta,
    p_logreg: Option<f64>,
    p_cat: f64,
    file_index: &HashMap<PathBuf, FileEntry>,
    manual_override_set: &HashSet<PathBuf>,
) -> Prediction {
    let p_combined = p_logreg.unwrap_or(p_cat);
    let entry = file_index.get(&meta.stem);
    let json_path = entry.and_then(|e| e.json.clone()).unwrap_or_default();
    let jpg_path = entry.and_then(|e| e.jpg.clone());
    let txt_path = entry.and_then(|e| e.txt.clone());
    let manually_overridden = txt_path
        .as_ref()
        .map(|t| manual_override_set.contains(t))
        .unwrap_or(false);
    let filename = meta
        .stem
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    Prediction {
        filename,
        batch: meta.batch,
        grade: meta.grade,
        subject: meta.subject,
        doc_type: meta.doc_type,
        p_categorical: p_cat,
        p_logreg,
        p_combined,
        json_path,
        jpg_path,
        txt_path,
        manually_overridden,
    }
}

// ---------------------------------------------------------------------------
// Pipeline phases
// ---------------------------------------------------------------------------

/// Output of the scan-and-classify phase.
struct ClassificationResult {
    labeled: Vec<LabeledRow>,
    unprocessed: Vec<Meta>,
    n_ambiguous: usize,
    n_accept: usize,
    n_reject: usize,
    manual_override_set: HashSet<PathBuf>,
}

/// Walk every `.json`-backed stem in the file index, parse the filename and
/// review log, and split into labeled / unprocessed / ambiguous buckets.
fn scan_and_classify(file_index: &HashMap<PathBuf, FileEntry>) -> ClassificationResult {
    let mut labeled = Vec::new();
    let mut unprocessed = Vec::new();
    let mut n_ambiguous = 0usize;
    let mut manual_override_set = HashSet::new();

    for entry in file_index.values() {
        if entry.json.is_none() {
            continue;
        }
        let meta = match crate::parsing::parse_stem(&entry.stem) {
            Some(m) => m,
            None => continue,
        };
        match &entry.txt {
            Some(t) => match crate::parsing::parse_txt(t) {
                Some((label, is_manual)) => {
                    if is_manual {
                        manual_override_set.insert(t.clone());
                    }
                    labeled.push(LabeledRow { meta, label });
                }
                None => n_ambiguous += 1,
            },
            None => unprocessed.push(meta),
        }
    }

    let n_accept = labeled.iter().filter(|r| r.label == Label::Accept).count();
    let n_reject = labeled.len() - n_accept;

    ClassificationResult {
        labeled,
        unprocessed,
        n_ambiguous,
        n_accept,
        n_reject,
        manual_override_set,
    }
}

/// Build the 14-feature vectors for all labeled rows and collect the indices
/// of rows that have at least JSON features (required for LR training).
fn build_labeled_features(
    aligned: &[LabeledRowWithFeatures<'_>],
    cond: &crate::model::ConditionalProbModel,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let all_features: Vec<Vec<f64>> = aligned
        .iter()
        .map(|a| {
            let p_cat = cond.predict_proba(a.row.meta.batch, &a.row.meta.subject);
            let json7 = a.json7.unwrap_or([0.0; 7]);
            let img6 = a.img6.unwrap_or([0.0; 6]);
            build_feature_vector(json7, img6, p_cat)
        })
        .collect();

    let labeled_indices: Vec<usize> = aligned
        .iter()
        .enumerate()
        .filter(|(_, a)| a.json7.is_some())
        .map(|(i, _)| i)
        .collect();

    (all_features, labeled_indices)
}

/// Train the final logistic regression model on all labeled rows with
/// complete features.
fn train_final_lr(
    labeled: &[LabeledRow],
    all_features: &[Vec<f64>],
    labeled_indices: &[usize],
    cfg: &PipelineConfig,
) -> crate::model::LogisticRegression {
    let final_labels: Vec<bool> = labeled_indices
        .iter()
        .map(|&i| labeled[i].label == Label::Accept)
        .collect();
    let final_features: Vec<Vec<f64>> = labeled_indices
        .iter()
        .map(|&i| all_features[i].clone())
        .collect();

    let mut lr_model = crate::model::LogisticRegression::new();
    lr_model.train(&final_labels, &final_features, cfg.lr_rate, cfg.lr_epochs);
    lr_model
}

/// Score all unprocessed files in parallel using the trained LR model.
/// Returns a map from stem path to P(accept).
fn score_unprocessed_with_lr(
    unprocessed: &[Meta],
    rows_unproc: &[crate::features::RowFeatures],
    cond: &crate::model::ConditionalProbModel,
    lr_model: &crate::model::LogisticRegression,
) -> HashMap<PathBuf, f64> {
    unprocessed
        .par_iter()
        .enumerate()
        .filter_map(|(i, r)| {
            let row = rows_unproc.get(i)?;
            let json7 = row.json7?;
            let img6 = row.img6?;
            let p_cat = cond.predict_proba(r.batch, &r.subject);
            let prob = lr_model.predict_proba(&build_feature_vector(json7, img6, p_cat));
            Some((r.stem.clone(), prob))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MAIN PIPELINE
// ---------------------------------------------------------------------------

pub fn run<F: FnMut(&str)>(root: &Path, cfg: &PipelineConfig, mut log: F) -> Result<PipelineResult> {
    // ── Phase 1: scan and classify ──────────────────────────────────────
    log(&format!("Scanning dataset: {}", root.display()));
    let file_index = build_file_index(root);
    log(&format!("  Found {} unique file stems", file_index.len()));

    let class = scan_and_classify(&file_index);

    log(&format!(
        "  Labeled: {}  (accept={}, reject={})",
        class.labeled.len(),
        class.n_accept,
        class.n_reject
    ));
    log(&format!("  Ambiguous: {}", class.n_ambiguous));
    log(&format!("  Unprocessed: {}", class.unprocessed.len()));

    if class.labeled.is_empty() {
        return Err(anyhow!("No labeled files found. Cannot train model."));
    }
    let n_unprocessed = class.unprocessed.len();
    if class.unprocessed.is_empty() {
        return Err(anyhow!("No unprocessed files found. Nothing to predict."));
    }

    // ── Phase 2: train conditional model + load features ────────────────
    log("Training conditional probability model (batch x subject)...");
    let mut cond = crate::model::ConditionalProbModel::new(cfg.alpha);
    cond.fit(&class.labeled);

    let mut img_cache = crate::img_feat_cache::load(root).unwrap_or_default();

    log("Loading annotations and computing JSON + image features...");
    let labeled_stems: Vec<&Path> = class.labeled.iter().map(|r| r.meta.stem.as_path()).collect();
    let unproc_stems: Vec<&Path> = class.unprocessed.iter().map(|r| r.stem.as_path()).collect();
    let (rows_labeled, n_load_fail_labeled) =
        crate::features::load_row_features(&labeled_stems, &file_index, &mut img_cache);
    let (rows_unproc, n_load_fail_unproc) =
        crate::features::load_row_features(&unproc_stems, &file_index, &mut img_cache);

    if n_load_fail_labeled + n_load_fail_unproc > 0 {
        log(&format!(
            "  Annotation load failures: {} labeled, {} unprocessed (rows excluded from image features)",
            n_load_fail_labeled, n_load_fail_unproc,
        ));
    }

    // ── Phase 3: build features ─────────────────────────────────────────
    let aligned = align_labeled_features(&class.labeled, &rows_labeled);
    let (all_features, labeled_indices) = build_labeled_features(&aligned, &cond);

    let lr_trained_on = labeled_indices.len();
    log(&format!(
        "  LR training set: {} labeled files with complete features",
        lr_trained_on
    ));

    // ── Phase 4: cross-validate + train final LR ────────────────────────
    log(&format!(
        "Running {}-fold cross-validation for LR...",
        cfg.cv_folds
    ));
    let kfold = crate::model::run_lr_kfold_cv(
        &class.labeled,
        &all_features,
        &labeled_indices,
        cfg.cv_folds,
        cfg.cv_epochs,
        cfg.lr_rate,
    );

    log("Training final LR model on all labeled data...");
    let lr_model = train_final_lr(&class.labeled, &all_features, &labeled_indices, cfg);

    log(&format!(
        "  Validation - accuracy: {:.3}  AUC: {:.3}",
        kfold.accuracy, kfold.auc
    ));
    if let Some(t) = kfold.suggested_high_cutoff {
        log(&format!(
            "  Suggested High cutoff (90% reject recall): {:.3}",
            t
        ));
    }

    // ── Phase 5: score unprocessed ──────────────────────────────────────
    log(&format!(
        "Scoring {} unprocessed files with LR...",
        class.unprocessed.len()
    ));
    let lr_scores =
        score_unprocessed_with_lr(&class.unprocessed, &rows_unproc, &cond, &lr_model);

    let _ = img_cache.save(root);

    let lr_scored = lr_scores.len();
    log(&format!(
        "  LR scored {}/{} files",
        lr_scored, n_unprocessed
    ));

    // ── Phase 6: generate predictions ───────────────────────────────────
    log("Generating predictions...");
    let predictions: Vec<Prediction> = class
        .unprocessed
        .into_iter()
        .map(|r| {
            let p_cat = cond.predict_proba(r.batch, &r.subject);
            let p_logreg = lr_scores.get(&r.stem).copied();
            build_prediction(r, p_logreg, p_cat, &file_index, &class.manual_override_set)
        })
        .collect();

    let mut sorted_p_combined: Vec<f64> = predictions.iter().map(|p| p.p_combined).collect();
    sorted_p_combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    log(&format!("Done. {} predictions.", predictions.len()));

    Ok(PipelineResult {
        predictions,
        sorted_p_combined,
        validation: ValidationResult {
            accuracy: kfold.accuracy,
            auc: kfold.auc,
            recall_by_reason: kfold.recall_by_reason,
            feature_weights: kfold.feature_weights,
            suggested_high_cutoff: kfold.suggested_high_cutoff,
        },
        n_labeled: class.labeled.len(),
        n_accept: class.n_accept,
        n_reject: class.n_reject,
        n_ambiguous: class.n_ambiguous,
        n_unprocessed,
        lr_trained_on,
        lr_scored,
    })
}
