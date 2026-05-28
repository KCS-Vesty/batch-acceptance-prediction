//! Sidebar panels for the annotation editor.
//!
//! Contains the shape list / property editor (`shape_sidebar`), the status
//! panel with Save / Mark accept / Mark reject (`status_panel`), and helpers
//! for writing manual override entries (`append_override`).

use eframe::egui;
use super::state::{EditorState, new_rect, snapshot_shapes};
use crate::annotation::{self, AnnotationFile};
use crate::app::App;
use crate::review_log;
use crate::ui::{C_INDIGO, C_MUTED, tier_color, tier_long};

/// Reject reason options shown in the "Mark reject ▾" submenu.
const REJECT_REASONS: &[(&str, &str)] = &[
    ("Whitespace", "whitespace"),
    ("Rotation", "rotation"),
    ("Cutoff", "cutoff"),
    ("Structure", "structure"),
    ("Generic", "generic"),
];

// ---------------------------------------------------------------------------
// Shape list sidebar
// ---------------------------------------------------------------------------

pub(crate) fn shape_sidebar(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.heading("Shapes");
    if state.annotation.is_none() {
        ui.label("No annotation loaded.");
        return;
    }

    // Per-row selection list.
    let shape_count = state.annotation.as_ref().map(|a| a.shapes.len()).unwrap_or(0);
    egui::ScrollArea::vertical()
        .max_height(240.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for i in 0..shape_count {
                let (is_sel, label) = match state.annotation.as_ref() {
                    Some(a) => (
                        state.selected_shape == Some(i),
                        format!("[{}] {}", i, a.shapes[i].label),
                    ),
                    None => break,
                };
                if ui.selectable_label(is_sel, label).clicked() {
                    if let Some(a) = state.annotation.as_ref() {
                        state.label_buffer = a.shapes[i].label.clone();
                        state.shape_type_buffer = a.shapes[i].shape_type.clone();
                    }
                    state.selected_shape = Some(i);
                }
            }
        });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui.button("Add box").clicked() {
            snapshot_shapes(state);
            let shape = new_rect("object", 280.0, 280.0, 80.0, 80.0);
            if state.annotation.is_none() {
                state.annotation = Some(AnnotationFile {
                    raw: serde_json::Map::new(),
                    shapes: Vec::new(),
                });
            }
            if let Some(a) = &mut state.annotation {
                a.shapes.push(shape);
                state.selected_shape = Some(a.shapes.len() - 1);
                state.label_buffer = "object".to_string();
                state.shape_type_buffer = "rectangle".to_string();
                state.dirty = true;
            }
        }
        let del_enabled = state.selected_shape.is_some();
        if ui
            .add_enabled(del_enabled, egui::Button::new("Delete"))
            .clicked()
        {
            snapshot_shapes(state);
            if let Some(idx) = state.selected_shape.take() {
                if let Some(a) = &mut state.annotation {
                    if idx < a.shapes.len() {
                        a.shapes.remove(idx);
                        state.dirty = true;
                    }
                }
            }
        }
    });

    ui.add_space(8.0);
    ui.separator();
    ui.label("Selected shape:");

    // Label field — only writes back if the user actually typed something.
    if state.selected_shape.is_some() {
        let label_resp = ui.add(
            egui::TextEdit::singleline(&mut state.label_buffer).hint_text("Label"),
        );
        if label_resp.changed() {
            if let (Some(idx), Some(a)) = (state.selected_shape, state.annotation.as_mut()) {
                if idx < a.shapes.len() && a.shapes[idx].label != state.label_buffer {
                    a.shapes[idx].label = state.label_buffer.clone();
                    state.dirty = true;
                }
            }
        }

        ui.label("Shape type:");
        let mut new_type = state.shape_type_buffer.clone();
        egui::ComboBox::from_id_source("shape_type_combo")
            .selected_text(&new_type)
            .show_ui(ui, |ui| {
                for t in &["rectangle", "polygon", "line", "point", "rotation"] {
                    ui.selectable_value(&mut new_type, (*t).to_string(), *t);
                }
            });
        if new_type != state.shape_type_buffer {
            state.shape_type_buffer = new_type.clone();
            if let (Some(idx), Some(a)) = (state.selected_shape, state.annotation.as_mut()) {
                if idx < a.shapes.len() && a.shapes[idx].shape_type != new_type {
                    a.shapes[idx].shape_type = new_type;
                    state.dirty = true;
                }
            }
        }
    }

    ui.add_space(6.0);
    ui.small("Drag a box to move, drag corners to resize,");
    ui.small("drag the red circle to rotate. Delete to remove.");
}

// ---------------------------------------------------------------------------
// Status panel (Save / Mark accept / Mark reject)
// ---------------------------------------------------------------------------

pub(crate) fn status_panel(ui: &mut egui::Ui, state: &mut EditorState, app: &mut App) {
    // ── Why was this file flagged? ─────────────────────────────────────────
    // Pull the prediction context up so the reviewer sees the model's verdict
    // alongside the edit controls, not just an opaque filename.
    if let (Some(idx), Some(r)) = (state.current, app.analysis.result.as_ref()) {
        if let Some(p) = r.predictions.get(idx) {
            let cutoffs = app.analysis.cutoffs();
            let tier = crate::tier::classify(p.p_combined, cutoffs);
            egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Risk:").color(C_MUTED));
                        ui.label(
                            egui::RichText::new(tier_long(tier))
                                .color(tier_color(tier))
                                .strong(),
                        );
                        ui.separator();
                        ui.label(egui::RichText::new("p_comb:").color(C_MUTED));
                        ui.monospace(format!("{:.3}", p.p_combined));
                        if let Some(lr) = p.p_logreg {
                            ui.separator();
                            ui.label(egui::RichText::new("p_lr:").color(C_MUTED));
                            ui.monospace(format!("{:.3}", lr));
                        }
                    });
                    ui.add_space(2.0);
                    ui.small(format!(
                        "Batch {:02} · {} · {} · {}",
                        p.batch, p.subject, p.grade, p.doc_type
                    ));
                    if p.manually_overridden {
                        ui.add_space(2.0);
                        ui.colored_label(
                            C_INDIGO,
                            "This file already has a manual override.",
                        );
                    }
                });
            ui.add_space(8.0);
        }
    }

    ui.heading("Status");

    let save_enabled = state.annotation.is_some() && state.dirty;
    if ui
        .add_enabled(save_enabled, egui::Button::new("Save annotation"))
        .clicked()
    {
        if let (Some(idx), Some(ann)) = (state.current, state.annotation.as_ref()) {
            if let Some(r) = app.analysis.result.as_ref() {
                let path = r.predictions[idx].json_path.clone();
                match annotation::save(&path, ann) {
                    Ok(()) => {
                        state.dirty = false;
                        state.undo_snapshot = None;
                        state.last_status_msg = Some(format!("Saved {}", path.display()));
                    }
                    Err(e) => {
                        state.last_status_msg = Some(format!("Save failed: {}", e));
                    }
                }
            }
        }
    }

    ui.add_space(8.0);
    ui.separator();
    ui.label("Override review status:");
    ui.horizontal(|ui| {
        if ui.button("Mark accept").clicked() {
            append_override(state, app, "manual-accept");
        }
        ui.menu_button("Mark reject ▾", |ui| {
            for (label, reason) in REJECT_REASONS {
                if ui.button(*label).clicked() {
                    ui.close_menu();
                    let token = format!("manual-reject-{}", reason);
                    append_override(state, app, &token);
                }
            }
        });
    });

    ui.add_space(8.0);
    ui.separator();
    if let Some(msg) = &state.last_status_msg {
        ui.colored_label(C_MUTED, msg);
    }

    // File info.
    if let (Some(idx), Some(r)) = (state.current, app.analysis.result.as_ref()) {
        let p = &r.predictions[idx];
        ui.add_space(6.0);
        ui.label(egui::RichText::new(&p.filename).strong());
        ui.small(format!("JSON: {}", short_path(&p.json_path)));
        if let Some(jpg) = &p.jpg_path {
            ui.small(format!("JPG: {}", short_path(jpg)));
        }
        if let Some(txt) = &p.txt_path {
            ui.small(format!("TXT: {}", short_path(txt)));
        }
    }
}

fn append_override(state: &mut EditorState, app: &mut App, token: &str) {
    let Some(idx) = state.current else { return };
    let Some(r) = app.analysis.result.as_mut() else { return };
    if idx >= r.predictions.len() {
        return;
    }
    // Derive a `.txt` path even when the row didn't originally have one —
    // sibling of the `.json` with the `.txt` extension. `append_status`
    // creates the file if missing.
    let txt_path = {
        let pred = &r.predictions[idx];
        pred.txt_path.clone().unwrap_or_else(|| {
            let mut p = pred.json_path.clone();
            p.set_extension("txt");
            p
        })
    };
    match review_log::append_status(&txt_path, token) {
        Ok(()) => {
            state.last_status_msg =
                Some(format!("Wrote {} to {}", token, short_path(&txt_path)));
            // Make the badge appear immediately without a rerun.
            r.predictions[idx].manually_overridden = true;
            r.predictions[idx].txt_path = Some(txt_path);
        }
        Err(e) => {
            state.last_status_msg = Some(format!("Override write failed: {}", e));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn short_path(p: &std::path::Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}
