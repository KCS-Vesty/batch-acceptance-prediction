# Batch Acceptance Prediction — Rust desktop app

Native Rust (eframe/egui) desktop app. Analyses `POC_P2_*` batch folders, scores
annotation quality with logistic regression on JSON + image features, and lets
you correct individual annotations in-app.

## Build

Requires Rust 1.74+ (install via <https://rustup.rs>).

```powershell
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
cd app
cargo build --release
```

`cargo build --release` produces `target/release/batch-acceptance-app.exe`. Single
file — copy anywhere; no DLLs.

## Run

```
.\target\release\batch-acceptance-app.exe
```

## Workflow

1. **Switch folder...** in the top bar — pick a directory containing `POC_P2_*`
   subfolders. You can switch parent folders at any time.
2. **Run analysis** in the side panel. The first run on a dataset populates the
   image-feature cache (slow); subsequent runs are fast.
3. Sort / filter the predictions table. Each row shows risk tier, batch,
   subject, grade, doc type, `p_cat`, `p_lr`, `p_comb`, and filename (with a
   `(manual)` badge when the `.txt` ends in a `manual-*` line).
4. **Double-click any row** → annotation editor:
   - Translate / corner-resize / rotate / add / delete shapes.
   - Edit `label` and `shape_type`.
   - **Ctrl-Z** undoes the last destructive change (one step).
   - **Save annotation** writes back to the JSON, preserving every field outside
     `shapes`.
   - **Mark accept / Mark reject ▾** appends `dd/mm/yyyy hh:mm:ss-manual-…` to
     the `.txt` sidecar (creates it if absent). The manual-override badge
     appears in the table immediately.
5. **Extract risky files** or **Send for correction** in the side panel: pick a
   destination; matching triplets (`.json` + `.jpg` + `.txt` when they exist)
   are copied or moved. Collision-safe naming (`_1`, `_2`, …) avoids overwrites.

## Risk tier modes

Side panel toggles between two `p_combined` → tier resolvers:

- **Percentile** — bottom X% = High, top Y% = Low. Always differentiates
  regardless of model output range.
- **Absolute** — user-set absolute `p_combined` cutoffs. After each run a
  suggested cutoff (90% reject recall, derived from k-fold CV) appears here
  with an **Apply** button.

## Model

14 features → hand-rolled L2 batch-GD logistic regression with z-score
standardisation and ~17× minority-class re-weighting:

- **7 JSON geometry features**: `n_shapes`, `coverage`, `avg_area`, `std_area`,
  `avg_width`, `avg_height`, `dir_std`.
- **6 image aggregates** (per-shape Sobel-gradient analysis on the JPEG):
  `[cutoff_mean, cutoff_max, isolation_mean, misalign_mean, misalign_max, ink_contrast_mean]`.
  Mean + max captures both "typical" and "worst" shape on the page.
- **`p_categorical`** — Laplace-smoothed `P(accept | batch, subject)` prior
  (alpha = side-panel slider, default 3.0).

K=5 cross-validation produces accuracy, AUC, per-`RejectReason` recall, top-5
weights, and a suggested 90%-reject-recall threshold.

## Cache

`<dataset_root>/.qa_img_cache.bin` (bincode). Key: JPEG path. Value:
`(jpg_mtime, json_mtime, feats: Vec<[f64; 4]>)`. Both mtimes must match for a
hit. A `CACHE_VERSION` prefix invalidates old caches when the per-shape feature
layout changes.

## File naming conventions

The app derives metadata from filename and folder structure:

- **Batch number** — read from the ancestor folder: `POC_P2_20000_<NN>` → batch `NN`.
- **Subject / grade / doc type** — parsed from the filename stem using the
  `padded_china_(junior|senior)-high-school_grade-<N>_<subject>_<doc-type>_…` pattern.
- **Unmatched filenames** are silently skipped (logged at trace level).

## Annotation format

The editor loads and saves LabelMe JSON. Top-level fields (`version`,
`imageWidth`, `imageHeight`, `imagePath`, `flags`, …) are preserved on save via
the `AnnotationFile::raw` map. Shape-level extra fields (`kie_linking`,
`group_id`, `difficult`, `score`, `description`, …) are round-tripped via
`#[serde(flatten)]` on `Shape::extra`.

## Architecture

```
src/
  main.rs              eframe entry point
  app.rs               App state, App::open_editor, async pipeline thread
  pipeline.rs          file indexing, parse_txt/parse_stem, LogisticRegression, KFoldOutcome
  tier.rs              TierConfig (Percentile/Absolute), Cutoffs, classify()
  annotation.rs        LabelMe JSON load/save; Shape with #[serde(flatten)] extras
  image_features.rs     cutoff / isolation / misalign / ink_contrast per shape
  img_feat_cache.rs     version-prefixed bincode cache, jpg + json mtime keyed
  review_log.rs        append_status (manual-* lines in the existing format)
  extract.rs           triplet-aware copy/move with collision-safe naming
  ui/
    top_bar.rs         folder bar + Switch folder
    side_panel.rs      Alpha, tier method, Run, Extract, Send for correction
    central.rs         KPI strip + status + filter + 2×2 chart grid + log
    charts.rs          risk distribution, accept-vs-reject, by-folder, by-batch
    table_panel.rs     bottom-panel host; delegates double-click to App::open_editor
    table.rs           sortable predictions table; TableAction carries double-click idx
    editor_window.rs   annotation editor (canvas + sidebar + status panel)
    log_view.rs        analysis-log display
    mod.rs             colour palette and format helpers
```

## How it differs from the Python script

- No CSV/HTML output — results live in the app.
- Logistic regression with standardisation + class weights replaces KNN.
- Image features (Sobel-gradient based) target the reviewer criteria: content
  cutoff, surrounding interference, text-line alignment.
- Parallelised via `rayon`; cached image features.
- In-app annotation editor + manual status override.