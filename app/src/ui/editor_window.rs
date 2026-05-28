//! Floating annotation editor window.
//!
//! Opens on double-click of a predictions table row. Shows the JPEG image
//! with shape overlays; right-hand sidebar for shape list, add/delete,
//! label/type editing, and status buttons (Save / Mark accept / Mark reject).

use super::editor::drag;
use super::editor::render::draw_shapes;
use super::editor::sidebar;
use super::editor::state::{DragOp, EditorState};
use crate::app::App;
use eframe::egui;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// The logical image size (pixels). Shapes are stored in this coordinate space.
/// Uses the canonical IMAGE_SIZE from the shared types module.
const IMAGE_SIZE: f64 = crate::types::IMAGE_SIZE;

const IMAGE_W: f32 = IMAGE_SIZE as f32;
const IMAGE_H: f32 = IMAGE_SIZE as f32;

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

pub fn show(app: &mut App, ctx: &egui::Context) {
    if !app.editor_state.open {
        return;
    }
    // Move state out of `app` for the duration of the frame so we can mutate
    // both freely. Put it back at the end.
    let mut state = std::mem::take(&mut app.editor_state);
    render_window(ctx, &mut state, app);
    app.editor_state = state;
}

fn render_window(ctx: &egui::Context, state: &mut EditorState, app: &mut App) {
    let title = state
        .current
        .and_then(|idx| {
            app.analysis
                .result
                .as_ref()
                .map(|r| r.predictions[idx].filename.clone())
        })
        .unwrap_or_else(|| "Annotation Editor".to_string());

    let mut keep_open = true;
    egui::Window::new(format!("Edit: {}", title))
        .open(&mut keep_open)
        .resizable(true)
        .default_width(1100.0)
        .default_height(740.0)
        .show(ctx, |ui| {
            if state.dirty {
                ui.colored_label(super::C_AMBER, "Unsaved changes");
                ui.separator();
            }
            ui.columns(2, |cols| {
                let canvas_col = &mut cols[0];
                canvas_col.heading("Canvas");
                image_canvas(canvas_col, state, app);

                let sidebar_col = &mut cols[1];
                sidebar::shape_sidebar(sidebar_col, state);
                sidebar_col.add_space(8.0);
                sidebar_col.separator();
                sidebar::status_panel(sidebar_col, state, app);
            });
        });

    if !keep_open {
        // User closed via window's X button.
        state.open = false;
        state.current = None;
        state.annotation = None;
        state.texture = None;
        state.selected_shape = None;
        state.drag = None;
    }
}

// ---------------------------------------------------------------------------
// Canvas (image + shape overlay + interactions)
// ---------------------------------------------------------------------------

fn image_canvas(ui: &mut egui::Ui, state: &mut EditorState, app: &mut App) {
    // ── Lazily load the JPEG into a texture ─────────────────────────────────
    let jpg_path = state.current.and_then(|idx| {
        app.analysis
            .result
            .as_ref()
            .and_then(|r| r.predictions.get(idx))
            .and_then(|p| p.jpg_path.clone())
    });

    if state.texture.is_none() {
        if let Some(jp) = &jpg_path {
            match image::open(jp) {
                Ok(img) => {
                    let rgb = img.into_rgb8();
                    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                    let pixels: Vec<egui::Color32> = rgb
                        .into_raw()
                        .chunks(3)
                        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
                        .collect();
                    let color_img = egui::ColorImage { size: [w, h], pixels };
                    state.texture = Some(ui.ctx().load_texture(
                        "annotation_image",
                        color_img,
                        egui::TextureOptions::default(),
                    ));
                }
                Err(e) => {
                    ui.colored_label(super::C_RED, format!("Failed to load JPEG: {}", e));
                }
            }
        }
    }

    // ── Allocate the canvas rect ────────────────────────────────────────────
    let desired = egui::vec2(ui.available_width(), IMAGE_SIZE as f32);
    let (resp, painter) = ui.allocate_painter(desired, egui::Sense::click_and_drag());

    // Aspect-preserving fit: scale to the smaller dimension.
    let scale = (resp.rect.width() / IMAGE_W).min(resp.rect.height() / IMAGE_H);
    let drawn_w = IMAGE_W * scale;
    let drawn_h = IMAGE_H * scale;
    let offset = egui::Pos2::new(
        resp.rect.left() + (resp.rect.width() - drawn_w) * 0.5,
        resp.rect.top() + (resp.rect.height() - drawn_h) * 0.5,
    );
    let img_rect = egui::Rect::from_min_size(offset, egui::vec2(drawn_w, drawn_h));

    let to_screen = |x: f64, y: f64| -> egui::Pos2 {
        egui::Pos2::new(offset.x + (x as f32) * scale, offset.y + (y as f32) * scale)
    };
    let to_image = |sp: egui::Pos2| -> (f64, f64) {
        (
            ((sp.x - offset.x) / scale) as f64,
            ((sp.y - offset.y) / scale) as f64,
        )
    };

    // ── Draw image as background ────────────────────────────────────────────
    if let Some(tex) = &state.texture {
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        let mut mesh = egui::Mesh::with_texture(tex.into());
        mesh.add_rect_with_uv(img_rect, uv, egui::Color32::WHITE);
        painter.add(mesh);
    } else {
        painter.rect_filled(img_rect, 0.0, egui::Color32::from_gray(40));
    }

    // ── Draw shapes ─────────────────────────────────────────────────────────
    if let Some(ann) = &state.annotation {
        draw_shapes(&painter, &ann.shapes, state.selected_shape, scale, &to_screen);
    }

    // ── Preview rectangle while drawing ─────────────────────────────────────
    if let Some(DragOp::Draw { p0 }) = &state.drag {
        if let Some(sp) = ui.ctx().pointer_interact_pos() {
            let (mx, my) = to_image(sp);
            let x0 = p0.0.min(mx);
            let y0 = p0.1.min(my);
            let x1 = p0.0.max(mx);
            let y1 = p0.1.max(my);
            let preview = egui::Rect::from_two_pos(to_screen(x0, y0), to_screen(x1, y1));
            painter.rect_stroke(
                preview,
                0.0,
                egui::Stroke::new(1.5, super::C_BLUE.gamma_multiply(0.9)),
            );
        }
    }

    // ── Ctrl-Z: undo last destructive change ────────────────────────────────
    let undo_pressed =
        ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z));
    if undo_pressed {
        if let Some(snap) = state.undo_snapshot.take() {
            if let Some(ann) = &mut state.annotation {
                ann.shapes = snap;
                state.dirty = true;
                state.selected_shape = None;
                state.last_status_msg = Some("Undid last change".to_string());
            }
        }
    }

    // ── Process interaction ─────────────────────────────────────────────────
    drag::handle_interaction(ui, &resp, state, scale, &to_image);
}
