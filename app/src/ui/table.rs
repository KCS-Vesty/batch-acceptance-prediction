use super::{tier_color, tier_short};
use crate::types::{Prediction, SortCol};
use crate::tier::{classify, Cutoffs};
use eframe::egui;
use egui_extras::{Column, TableBuilder};

/// Signals the table sends back to its caller after one frame.
pub struct TableAction {
    /// `Some((column, ascending))` if the user clicked a header to change sort.
    pub sort_change: Option<(SortCol, bool)>,
    /// `Some(prediction_index)` if the user double-clicked a row (to open the
    /// annotation editor on that file).
    pub open_editor_idx: Option<usize>,
}

pub fn show(
    ui: &mut egui::Ui,
    predictions: &[Prediction],
    indices: &[usize],
    cutoffs: Cutoffs,
    sort_col: SortCol,
    sort_asc: bool,
) -> TableAction {
    let mut sorted = indices.to_vec();
    sort_indices(&mut sorted, predictions, cutoffs, sort_col, sort_asc);

    let cols: [(SortCol, &str, f32); 9] = [
        (SortCol::Risk, "Risk", 50.0),
        (SortCol::Batch, "Batch", 50.0),
        (SortCol::Subject, "Subject", 90.0),
        (SortCol::Grade, "Grade", 70.0),
        (SortCol::DocType, "Doc type", 90.0),
        (SortCol::PCategorical, "p_cat", 60.0),
        (SortCol::PLogReg, "p_lr", 60.0),
        (SortCol::PCombined, "p_comb", 60.0),
        (SortCol::Filename, "File", 240.0),
    ];

    let mut builder = TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .min_scrolled_height(0.0);
    for (i, (_, _, min_w)) in cols.iter().enumerate() {
        builder = if i + 1 == cols.len() {
            builder.column(Column::remainder().at_least(*min_w))
        } else {
            builder.column(Column::initial(*min_w).at_least(*min_w).clip(true))
        };
    }

    let mut action = TableAction {
        sort_change: None,
        open_editor_idx: None,
    };

    builder
        .header(24.0, |mut header| {
            for (col, label, _) in &cols {
                header.col(|ui| {
                    let active = sort_col == *col;
                    let arrow = if active {
                        if sort_asc { " ▲" } else { " ▼" }
                    } else {
                        ""
                    };
                    let text = egui::RichText::new(format!("{}{}", label, arrow)).strong();
                    let mut resp = ui.button(text);
                    if let Some(tip) = col_tooltip(*col) {
                        resp = resp.on_hover_text(tip);
                    }
                    if resp.clicked() {
                        action.sort_change = Some(if active {
                            (sort_col, !sort_asc)
                        } else {
                            (*col, true)
                        });
                    }
                });
            }
        })
        .body(|body| {
            body.rows(22.0, sorted.len(), |mut row| {
                let global_idx = sorted[row.index()];
                let p = &predictions[global_idx];
                row.col(|ui| {
                    let t = classify(p.p_combined, cutoffs);
                    ui.colored_label(tier_color(t), tier_short(t));
                });
                row.col(|ui| {
                    ui.label(format!("{:02}", p.batch));
                });
                row.col(|ui| {
                    ui.label(&p.subject);
                });
                row.col(|ui| {
                    ui.label(&p.grade);
                });
                row.col(|ui| {
                    ui.label(&p.doc_type);
                });
                row.col(|ui| {
                    ui.monospace(format!("{:.3}", p.p_categorical));
                });
                row.col(|ui| {
                    ui.monospace(
                        p.p_logreg
                            .map(|v| format!("{:.3}", v))
                            .unwrap_or_else(|| "-".into()),
                    );
                });
                row.col(|ui| {
                    ui.monospace(format!("{:.3}", p.p_combined));
                });
                row.col(|ui| {
                    let resp = ui
                        .add(egui::Label::new(&p.filename).truncate().sense(egui::Sense::click()))
                        .on_hover_text("Double-click to open the annotation editor");
                    if resp.double_clicked() {
                        action.open_editor_idx = Some(global_idx);
                    }
                    if p.manually_overridden {
                        ui.colored_label(super::C_INDIGO, "(manual)");
                    }
                });
            });
        });

    action
}

fn sort_indices(
    indices: &mut [usize],
    preds: &[Prediction],
    cutoffs: Cutoffs,
    col: SortCol,
    asc: bool,
) {
    indices.sort_by(|&a, &b| {
        let pa = &preds[a];
        let pb = &preds[b];
        let ord = match col {
            SortCol::Risk => tier_rank(pa, cutoffs).cmp(&tier_rank(pb, cutoffs)),
            SortCol::Batch => pa.batch.cmp(&pb.batch),
            SortCol::Subject => pa.subject.cmp(&pb.subject),
            SortCol::Grade => pa.grade.cmp(&pb.grade),
            SortCol::DocType => pa.doc_type.cmp(&pb.doc_type),
            SortCol::PCategorical => pa
                .p_categorical
                .partial_cmp(&pb.p_categorical)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortCol::PLogReg => pa
                .p_logreg
                .unwrap_or(f64::NAN)
                .partial_cmp(&pb.p_logreg.unwrap_or(f64::NAN))
                .unwrap_or(std::cmp::Ordering::Equal),
            SortCol::PCombined => pa
                .p_combined
                .partial_cmp(&pb.p_combined)
                .unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Filename => pa.filename.cmp(&pb.filename),
        };
        if asc { ord } else { ord.reverse() }
    });
}

fn tier_rank(p: &Prediction, cutoffs: Cutoffs) -> u8 {
    match classify(p.p_combined, cutoffs) {
        crate::types::RiskTier::High => 0,
        crate::types::RiskTier::Medium => 1,
        crate::types::RiskTier::Low => 2,
    }
}

/// Hover tooltips for table headers. Returning None means "no tooltip" (the
/// header's purpose is obvious from its label).
fn col_tooltip(c: SortCol) -> Option<&'static str> {
    match c {
        SortCol::Risk => Some("Computed from p_comb and the tier cutoffs in the side panel."),
        SortCol::PCategorical => Some(
            "Laplace-smoothed P(accept | batch, subject) prior. Doesn't look at the image.",
        ),
        SortCol::PLogReg => Some(
            "Logistic regression output combining 7 JSON-geometry + 6 image features + the categorical prior.",
        ),
        SortCol::PCombined => Some(
            "Final probability used for the risk tier. Uses LR when available, falls back to categorical prior when image features are missing.",
        ),
        SortCol::Filename => Some("Double-click a row to open the annotation editor."),
        _ => None,
    }
}
