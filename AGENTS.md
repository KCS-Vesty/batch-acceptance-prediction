# Agents.md

## Project Overview

**`app/`** — Rust desktop app (eframe/egui). **Primary implementation.** Native
GUI with in-app results table, per-file annotation editor, and extract feature.
See [app/README.md](app/README.md) for build/run details.

**Legacy Python reference** — moved to `../batch-acceptance-legacy-python/`.
See that directory's `LEGACY_README.md` for historical documentation.

## Running the Rust app

```powershell
$env:PATH += ";$env:USERPROFILE\.cargo\bin"   # cargo not on PATH by default
cd app
cargo run --release
# or run the built exe:
./target/release/batch-acceptance-app.exe
```

Tests: `cargo test` (88 unit + integration tests across `parsing.rs`, `features.rs`, `image_features.rs`, `annotation.rs`, `img_feat_cache.rs`, `model.rs`, `validation.rs`, `extract.rs`, `tier.rs`, `review_log.rs`, `app.rs`, `ui/`, and `integration_tests.rs`).

## Modules & Logic (Rust app)

| Module | Key items | Purpose |
|--------|-----------|---------|
| `types.rs` | Domain types — `FileEntry`, `Meta`, `RejectReason`, `Label`, `LabeledRow`, `RiskTier`, `Prediction`, `PipelineConfig` (5 hyperparameters), `ValidationResult`, `PipelineResult`, `IMAGE_SIZE`, `LR_FEATURE_NAMES` | Single source of truth for all shared domain types. Imported by every other module via `crate::types`. |
|| `pipeline.rs` | `run()`, `build_file_index()`, `align_labeled_features()`, `LabeledRowWithFeatures` | Orchestrates the full pipeline. Scans dataset, delegates to parsing/features/model, returns `PipelineResult`. `align_labeled_features` couples labeled rows with their computed features by index. Entry from `main.rs`. |
|| `parsing.rs` | `parse_stem()`, `parse_txt()` | Filename → `Meta`. Batch number is read from the ancestor folder (`POC_P2_20000_NN`), not the filename. Reads `.txt` in reverse, returns `(Label, is_manual)`. Skips lines that lack a second `-` (guards against mis-parsing date-prefixed lines like `dd/mm/yyyy`). `is_manual = true` when the last status is `manual-*`. |
|| `features.rs` | `load_row_features()`, `RowFeatures` | Single-pass: parse annotation (parallel), compute the 7 JSON features, aggregate the 6 image features through the cache. Used by both labeled and unprocessed populations. |
|| `model.rs` | `LogisticRegression`, `ConditionalProbModel`, `KFoldOutcome`, `run_lr_kfold_cv()`, `fast_auc()` | 14 features (7 JSON + 6 image aggregates + `p_categorical`). Z-score standardised, ~17× minority-class weighted, L2 batch GD. `ConditionalProbModel` is a Bayesian prior for per-(batch, subject) acceptance. k=5 CV. AUC via `fast_auc()` — top-200 trace-pair sampling. |
|| `validation.rs` | `suggest_high_cutoff()`, `fast_auc()`, `fold_reject_counts()`, `KFoldOutcome` | Given validation predictions, finds `p_combined` threshold for ~90% reject recall. AUC via top-200 trace-pair sampling. Per-fold reject-reason accumulator. |
|| `tier.rs` | `TierConfig`, `classify()`, `Cutoffs`, `TierMethod` | Percentile mode (bottom X% = High, top Y% = Low) or Absolute mode (user-set `p_combined` cutoffs). |
|| `annotation.rs` | `load()`, `save()`, `AnnotationFile`, `Shape` | Parse and re-serialise LabelMe JSON. Unknown shape fields preserved via `#[serde(flatten)]`; top-level fields (imageWidth, version, flags, …) preserved on save. |
|| `image_features.rs` | `extract_for_file()`, `aggregate_arrays()`, `Rect` | Per-shape `cutoff`, `isolation`, `misalign`, `ink_contrast` from Sobel gradients. Aggregated to 6 dims (mean + max for cutoff/misalign). Inner-box and exterior-ring helpers take `Rect` rather than raw u32 coordinate quartets. |
|| `img_feat_cache.rs` | `load()`, `save()`, `get_or_insert()` | Bincode cache at `<dataset_root>/.qa_img_cache.bin`. Version-prefixed; keyed by jpg + json mtime. |
|| `review_log.rs` | `append_status()` | Appends `dd/mm/yyyy hh:mm:ss-<status>` to the `.txt` (creates if missing). |
|| `extract.rs` | `extract()`, `transfer_one()`, `ExtractOptions`, `ExtractResult` | Triplet-aware copy/move with collision-safe naming (`_1`, `_2`, …). The `.json` is the anchor — if its filename can't be derived the row is skipped; sidecars run through the same `transfer_one` helper. |
|| `app.rs` | `App`, `AnalysisState`, `Sentinel`, `run_extract()`, `RunExtractOpts` | Top-level application state (`App`) with shared analysis state (`AnalysisState`) bundling result, tier config, and filter/sort settings. `run_extract` is the shared body of `do_extract` / `do_export_correction`. `Sentinel<T>` replaces `Option<PipelineResult>` with a self-documenting type. |
|| `ui/mod.rs` | `passes_filter()`, `tier_color()`, `tier_short()`, `tier_long()`, `format_int()` | Shared widgets/helpers. `passes_filter` is the single source of truth for tier+subject filtering used by the dashboard and the table. |
| `ui/central.rs` | `dashboard_panel()` | Main dashboard view — model chips, KPI strip, filter row. 378 lines. |
| `ui/charts.rs` | various | Risk distribution, accept-vs-reject, by-folder/by-batch acceptance rate charts. |
| `ui/editor_window.rs` | `render_window()`, `image_canvas()` | Annotation editor top-level layout. 192 lines (orchestrator). |
| `ui/editor/mod.rs` | Re-exports `EditorState` | Entry point for the `ui/editor/` sub-module. |
| `ui/editor/drag.rs` | `DragState`, `update_drag()`, `commit_drag()` | Rectangle corner drag, shape translation, rotation handle drag. |
| `ui/editor/hit_test.rs` | `hit_test()`, `image_center()`, `bbox_bounds()` | Point-in-polygon, bounding-box proximity, rotation handle hit detection. |
| `ui/editor/render.rs` | `draw_shapes()` | Pure paint-only shape outlines, corner handles, rotation handles. |
| `ui/editor/sidebar.rs` | `shape_sidebar()`, `status_panel()`, `append_override()` | Shape property editor, label field, save/accept/reject buttons, manual override logic. 287 lines. |
| `ui/editor/state.rs` | `EditorState` | Annotation editor mutable state — selected shape, drag, undo snapshot, texture handle. |
| `ui/side_panel.rs` | `side_panel()` | Folder selection, tier mode config, run button, extract controls. |
| `ui/table.rs` | `SortCol`, `sort_predictions()`, `data_table()` | Sortable predictions table with double-click → editor, manual-flag badge. |
| `ui/table_panel.rs` | `table_panel()` | Wraps `table.rs` with filter-aware index mapping. |
| `ui/top_bar.rs` | `top_bar()` | Menu bar and app-level actions. |
| `ui/log_view.rs` | `log_view()` | Console log output display. |

## UI Features (Rust app)

- **Switch folder** — pick any parent directory containing `POC_P2_*`
  subfolders at any time.
- **Risk tier method** — Percentile or Absolute. Absolute mode shows a
  suggested cutoff (90% reject recall from k-fold CV) with an Apply button.
- **Predictions table** — sortable; double-click a row to open the annotation
  editor; `(manual)` badge on rows with `manual-*` status in the `.txt`.
- **Annotation editor** — translate / corner-resize / rotate / add / delete /
  relabel shapes. Drag empty canvas → new rectangle (dashed preview while
  dragging). **Ctrl-Z** for one-step undo. Polygon point-in-polygon hit-test.
  **Save annotation** writes the JSON; **Mark accept / Mark reject ▾** appends
  a `manual-*` line to the `.txt`.
- **Extract risky files / Send for correction** — copy or move triplets
  (`.json` + `.jpg` + `.txt`) to a chosen folder.
- **Charts** — risk distribution, accept-vs-reject, by-folder acceptance rate
  (folder = immediate parent dir, e.g. `POC_P2_20000_16`), by-batch rate.
- **Status line** — N predictions, accuracy, AUC, per-`RejectReason` recall,
  top-5 feature weights.

## Key Contracts

- **Training data**: `.json` + `.txt` = labeled; `.json` only = unlabeled.
- **Feature extraction**: shapes require `{label, points, shape_type}`; unknown
  fields preserved via `#[serde(flatten)]`.
- **Model validation**: k=5 cross-validation; AUC over top 200 pos/neg pairs.
- **Prediction pipeline**: `p_combined` = LR output → `tier::classify(p_combined, cutoffs)`.
  Cutoffs come from the user-selected `TierConfig` mode.
- **Triplet integrity**: Extract always travels `.json` + `.jpg` + `.txt`
  together when sidecars exist. Checkboxes control inclusion, not splitting.
- **Annotation editor save**: patches `shapes` into the JSON while preserving
  every other top-level field. Unknown shape fields round-trip via the
  `Shape.extra` map.
- **Manual override flow**: editor append → `.txt` ends in `manual-*` line →
  next `parse_txt` flags `is_manual` → `Prediction.manually_overridden` set →
  table badge appears. In-memory flag is also flipped immediately so the badge
  updates without a re-run.

## Gotchas

- **Cargo PATH**: always prepend `$env:PATH += ";$env:USERPROFILE\.cargo\bin"`
  before `cargo` on this Windows host.
- **Never commit, push, or alter git state** unless the user explicitly asks
  — this is not a git repo today.
- **Cache invalidation**: `CACHE_VERSION` in `img_feat_cache.rs` is bumped
  whenever the per-shape feature layout in `image_features.rs` changes;
  mismatched files are silently treated as empty so old caches don't corrupt
  new runs. JSON edits via the editor bump the json mtime, which invalidates
  just that file's cache entry.
- **Release builds fail if the .exe is running** (Windows file lock). Close
  the running app first.
- **`parse_txt` line format**: `<prefix>-<status>` per line. Real review logs
  use `dd/mm/yyyy hh:mm:ss-<status>`. The parser skips lines without a second
  `-` after the timestamp prefix, so date-prefixed lines cannot corrupt status
  extraction.
- **Data location**: at the time of writing only `POC_P2_20000_16` is
  populated; other batch dirs are placeholders.
