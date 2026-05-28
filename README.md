# Batch Acceptance Prediction

Desktop app that scores annotation quality across annotation batch folders using
logistic regression on JSON geometry features + Sobel-gradient image features.
Built in Rust with an egui/eframe GUI.

## Features Overview

_After running analysis on a dataset folder:_

- **Dashboard** — model chips (accuracy, AUC, per-reason recall, top-5 feature
  weights), KPI strip (files in view, expected accept count, tier breakdown),
  risk distribution chart, accept-vs-reject chart, by-folder and by-batch
  acceptance-rate charts.
- **Predictions table** — sortable by risk tier, batch, subject, grade, doc
  type, p_cat, p_lr, p_comb, or filename. Double-click any row to open the
  annotation editor.
- **Annotation editor** — JPEG image canvas with shape overlays. Translate,
  corner-resize, rotate, add, or delete shapes. Ctrl-Z undo. Save annotation
  (writes LabelMe JSON) and Mark accept / Mark reject (appends to review-log
  `.txt`). `(manual)` badge appears immediately in the table.
- **Extract** — copy or move `.json` + `.jpg` + `.txt` triplets for selected
  risk tiers. Collision-safe naming (`_1`, `_2`, …).

## Techniques

| Technique | Where | Purpose |
|-----------|-------|---------|
| **Logistic regression** (hand-rolled L2 batch GD) | `model/logistic_regression.rs` | 14-feature → P(accept) with ~17× minority-class re-weighting |
| **Z-score standardisation** | `model/logistic_regression.rs` | Per-feature (mu, sigma) computed at train time, applied at predict time |
| **Sobel gradient magnitude** (via `imageproc`) | `image_features.rs` | Per-shape cutoff, isolation, misalignment, ink-contrast features |
| **Structure tensor** | `image_features.rs` | Dominant text-line orientation inside each bounding box for misalignment |
| **K-fold cross-validation** (k=5) | `model/kfold.rs` | Accuracy, AUC, per-reason recall, suggested 90%-reject-recall cutoff |
| **AUC via top-200 trace-pair sampling** | `validation.rs` | Fast approximation — top 200 positive + 200 negative predictions |
| **Laplace-smoothed Bayesian prior** | `model/conditional_prob.rs` | P(accept \| batch, subject) with additive smoothing (alpha tunable) |
| **Parallel feature loading** (rayon) | `features.rs`, `pipeline.rs` | JSON annotation parsing + image feature extraction parallelised |
| **Bincode cache** (mtime-keyed) | `img_feat_cache.rs` | Version-prefixed `.qa_img_cache.bin` — skips recomputation on rerun |
| **LabelMe JSON round-tripping** | `annotation.rs` | `#[serde(flatten)]` preserves unknown fields on save |
| **Point-in-polygon hit testing** | `ui/editor/hit_test.rs` | Even-odd ray casting for shape selection; handle-priority hit resolution |
| **Aspect-preserving image fit** | `ui/editor_window.rs` | Canvas scales JPEG to the smaller dimension with centring |

## Quick Start

**Requirements:** Rust 1.74+ ([rustup.rs](https://rustup.rs))

```powershell
# Build (Windows PowerShell)
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
cd app
cargo build --release

# Run
.\target\release\batch-acceptance-app.exe
```

On first run the image-feature cache is cold (~200 MB); subsequent runs use the
cached results and are significantly faster.

## Workflow

1. **Pick a folder** — top bar → "Switch folder..." → select a parent directory
   containing annotation batch subfolders.
2. **Run analysis** — side panel → ▶ Run analysis. Progress streams into the
   log. The pipeline scans `.json` + `.txt` pairs (labeled) and `.json`-only
   files (unprocessed).
3. **Explore results** — sort and filter the predictions table. Charts update
   live with the current filter.
4. **Edit annotations** — double-click any table row to open the annotation
   editor. Save changes back to the JSON; mark accept/reject to append to the
   `.txt` review log.
5. **Extract** — tick risk tiers, pick a destination, and copy/move triplets.

## Architecture

```
app/src/
  main.rs                    eframe entry point (thin)
  lib.rs                     library crate root
  app.rs                     App state, async pipeline thread, extract runner
  pipeline.rs                Pipeline orchestrator + scan/classify/score phases
  types.rs                   Shared domain types (Meta, Prediction, PipelineConfig, …)
  model/
    mod.rs                   Re-exports
    logistic_regression.rs   LR with L2 batch GD, z-score std, class re-weighting
    kfold.rs                 K-fold CV orchestrator
    conditional_prob.rs      Laplace-smoothed P(accept | batch, subject) prior
  features.rs                JSON geometry features + RowFeatures + build_feature_vector
  parsing.rs                 Filename stem parsing, review-log .txt parsing
  validation.rs              AUC, fold_reject_counts, cutoff suggestion
  tier.rs                    TierConfig (Percentile / Absolute), Cutoffs, classify()
  image_features.rs          Per-shape Sobel-gradient analysis (cutoff/isolation/misalign/ink)
  img_feat_cache.rs          Version-prefixed bincode cache (jpg+json mtime keyed)
  annotation.rs              LabelMe JSON load/save; Shape with #[serde(flatten)] extras
  extract.rs                 Triplet-aware copy/move with collision-safe naming
  review_log.rs              Append D/M/Y HH:MM:SS-<status> to .txt review log
  integration_tests.rs       Full pipeline test with synthetic data (temp dirs)
  ui/
    mod.rs                   Colour palette, tier helpers, pass_filter, format_int
    top_bar.rs               Folder bar + status pill + Switch folder button
    side_panel.rs            Alpha slider, tier method, Run, Extract, Send for correction
    central.rs               Dashboard: KPI strip, model chips, filters, 2x2 chart grid
    charts.rs                Risk distribution, accept-vs-reject, by-folder, by-batch
    table.rs                 Sortable predictions table (egui_extras TableBuilder)
    table_panel.rs           Bottom panel host; delegates sort + double-click to App
    editor_window.rs         Annotation editor (canvas + sidebar + status panel)
    onboarding.rs            Empty-state UI shown before first analysis run
    log_view.rs              Analysis-log display
    editor/
      mod.rs                 Re-exports EditorState
      state.rs               EditorState, DragOp enum, open_for, snapshot_shapes, new_rect
      hit_test.rs            Point-in-polygon, handle hit-testing, image_center
      render.rs              Shape outline + corner/rotation handle painting
      drag.rs                Translate / resize / rotate / draw interaction
      sidebar.rs             Shape list, label/shape_type editor, status panel
```

## How It Works

### Pipeline

1. **Scan** all `.json`, `.jpg`, `.txt` files. Index by stem path.
2. **Classify** each stem: `.json` + `.txt` → labeled; `.json`-only → unprocessed;
   `.txt` with unparseable status → ambiguous (skipped).
3. **Train** the Laplace-smoothed conditional prior `P(accept | batch, subject)`.
4. **Load features** — parse annotations (parallel rayon), extract JSON geometry
   features (7 dims), load image features through the mtime cache (6 aggregate dims).
5. **Build** 14-element feature vectors: `[json7 | img6 | p_categorical]`.
6. **Cross-validate** (k=5): train per-fold LR, pool predictions, compute accuracy,
   AUC (top-200 sampling), per-reject-reason recall, feature weights, and a
   suggested 90%-reject-recall `p_combined` cutoff.
7. **Train final LR** on all labeled rows.
8. **Score** all unprocessed files in parallel.
9. **Classify** each prediction into High / Medium / Low risk.

### Image Features (per shape, per JPEG)

| Feature | Signal | How |
|---------|--------|-----|
| **cutoff** | Content crossing the box edge (clipping) | Mean Sobel-gradient magnitude in a 2px interior border band, normalised by image-wide gradient baseline |
| **isolation** | Noisy surroundings outside the box | Mean gradient magnitude in a 6px exterior ring, same normalisation |
| **misalign** | Text orientation vs box direction | Structure-tensor dominant orientation vs shape `direction`; 0 = aligned, 1 = perpendicular |
| **ink_contrast** | Actual content inside a clean box | `(mean_outside_ring − mean_inside) / 255`, clamped to [−1, 1] |

Aggregated to 6 dims per file: `[cutoff_mean, cutoff_max, isolation_mean, misalign_mean, misalign_max, ink_contrast_mean]`.
Mean captures the "typical" shape; max captures the "worst" shape (one badly clipped box is often the reject signal).

### Risk Classification

Two modes (side panel toggle):

**Percentile** — bottom X% of `p_combined` = High, top Y% = Low, rest = Medium.
Always produces differentiation regardless of model output range.

**Absolute** — user-set `p_combined` cutoffs. After each run, a **Suggested
cutoff** (90% reject recall from k-fold CV) appears with an Apply button.

### Review Log Format

Each `.txt` file is an append-only review log:

```
dd/mm/yyyy hh:mm:ss-accept
dd/mm/yyyy hh:mm:ss-reject-whitespace
dd/mm/yyyy hh:mm:ss-manual-accept
dd/mm/yyyy hh:mm:ss-manual-reject-rotation
```

The **last valid line** determines the file's status. The parser skips
date-prefixed comment lines (e.g. `2024-01-01 note`) by checking for `:` in
the timestamp portion. `manual-*` lines flag the `(manual)` badge in the table.

## Tests

```powershell
cd app
cargo test
```

88 unit + integration tests covering: LR training convergence, sigmoid bounds,
conditional probability model, k-fold CV fold splitting, feature extraction,
stem + review-log parsing, tier classification, AUC computation, cutoff
suggestion, extract transfer + collision renaming, annotation JSON round-trip,
editor state, hit-test geometry, and a full end-to-end pipeline run on
synthetic data.

## Cache

`<dataset_root>/.qa_img_cache.bin` — bincode. Keyed by JPEG path. Value stores
`(jpg_mtime, json_mtime, per_shape_features)`. Both mtimes must match for a hit
(editing a JSON in the annotation editor bumps its mtime, invalidating that
file's cache entry). A `CACHE_VERSION` prefix invalidates the entire cache when
the per-shape feature layout changes.

## License

[MIT](LICENSE)
