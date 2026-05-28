//! Drag interaction handling for the annotation editor canvas.
//!
//! Provides `handle_interaction`, which is called each frame from the canvas
//! to process mouse drag-initiate, drag-apply, click, and keystroke events.
//! Also handles the Delete/Backspace key for removing the selected shape.

use eframe::egui;

use super::hit_test::{hit_test, image_center, HitTarget};
use super::state::{snapshot_shapes, DragOp, EditorState};

/// Process mouse / pointer interaction for one frame.
///
/// * `resp` — the `Response` from `ui.allocate_painter(…, click_and_drag)`
/// * `to_image` — maps a screen-space `Pos2` to image-space `(x, y)`.
pub(crate) fn handle_interaction(
    ui: &egui::Ui,
    resp: &egui::Response,
    state: &mut EditorState,
    scale: f32,
    to_image: &dyn Fn(egui::Pos2) -> (f64, f64),
) {
    // Snapshot once per frame for any change that *could* mutate shapes —
    // drag start (translate/resize/rotate/draw) or a Delete keystroke.
    // Cheap (clone of a few Vec entries) and means Ctrl-Z always works.
    let delete_pressed =
        ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
    if resp.drag_started() || delete_pressed {
        snapshot_shapes(state);
    }

    let shapes = match &mut state.annotation {
        Some(a) => &mut a.shapes,
        None => return,
    };

    let pointer_pos = ui.ctx().pointer_interact_pos();

    // ── Initiate a drag on press ────────────────────────────────────────────
    if resp.drag_started() {
        if let Some(sp) = pointer_pos {
            let (mx, my) = to_image(sp);
            let hit = hit_test(shapes, mx, my, scale, state.selected_shape);
            state.drag = match hit {
                HitTarget::Body(i) => {
                    state.selected_shape = Some(i);
                    state.label_buffer = shapes[i].label.clone();
                    state.shape_type_buffer = shapes[i].shape_type.clone();
                    Some(DragOp::Translate {
                        start_mouse: sp,
                        start_points: shapes[i].points.clone(),
                    })
                }
                HitTarget::Corner { shape, corner } => {
                    state.selected_shape = Some(shape);
                    Some(DragOp::Resize {
                        corner,
                        start_mouse: sp,
                        start_point: shapes[shape].points[corner as usize],
                    })
                }
                HitTarget::Rotate { shape } => {
                    let center = image_center(&shapes[shape]).unwrap_or((0.0, 0.0));
                    Some(DragOp::Rotate {
                        start_mouse: sp,
                        start_points: shapes[shape].points.clone(),
                        start_angle: shapes[shape].direction.unwrap_or(0.0),
                        center,
                    })
                }
                HitTarget::None => {
                    // Click on empty canvas: start drawing a new rect.
                    state.selected_shape = None;
                    Some(DragOp::Draw { p0: (mx, my) })
                }
            };
        }
    }

    // ── Apply an in-progress drag ───────────────────────────────────────────
    if resp.dragged() {
        if let (Some(sp), Some(op)) = (pointer_pos, state.drag.clone()) {
            let (mx, my) = to_image(sp);
            match op {
                DragOp::Translate {
                    start_mouse,
                    start_points,
                } => {
                    let (sx, sy) = to_image(start_mouse);
                    let dx = mx - sx;
                    let dy = my - sy;
                    if let Some(idx) = state.selected_shape {
                        if let Some(shape) = shapes.get_mut(idx) {
                            for (pi, pt) in shape.points.iter_mut().enumerate() {
                                if pi < start_points.len() {
                                    pt.0 = start_points[pi].0 + dx;
                                    pt.1 = start_points[pi].1 + dy;
                                }
                            }
                            state.dirty = true;
                        }
                    }
                }
                DragOp::Resize {
                    corner,
                    start_mouse,
                    start_point,
                } => {
                    let (sx, sy) = to_image(start_mouse);
                    if let Some(idx) = state.selected_shape {
                        if let Some(shape) = shapes.get_mut(idx) {
                            let c = corner as usize;
                            if c < shape.points.len() {
                                shape.points[c].0 = start_point.0 + (mx - sx);
                                shape.points[c].1 = start_point.1 + (my - sy);
                                state.dirty = true;
                            }
                        }
                    }
                }
                DragOp::Rotate {
                    start_mouse,
                    start_points,
                    start_angle,
                    center,
                } => {
                    let (sx, sy) = to_image(start_mouse);
                    let theta0 = (sy - center.1).atan2(sx - center.0);
                    let theta1 = (my - center.1).atan2(mx - center.0);
                    let delta = theta1 - theta0;
                    let new_angle = start_angle + delta;
                    let cos_a = delta.cos();
                    let sin_a = delta.sin();
                    if let Some(idx) = state.selected_shape {
                        if let Some(shape) = shapes.get_mut(idx) {
                            for (pi, pt) in shape.points.iter_mut().enumerate() {
                                if pi < start_points.len() {
                                    let dx = start_points[pi].0 - center.0;
                                    let dy = start_points[pi].1 - center.1;
                                    pt.0 = center.0 + cos_a * dx - sin_a * dy;
                                    pt.1 = center.1 + sin_a * dx + cos_a * dy;
                                }
                            }
                            shape.direction = Some(new_angle);
                            state.dirty = true;
                        }
                    }
                }
                DragOp::Draw { .. } => {
                    // Drawing previews on release; no interim update.
                }
            }
        }
    }

    // ── Commit drag on release ──────────────────────────────────────────────
    if resp.drag_stopped() {
        if let (Some(sp), Some(DragOp::Draw { p0 })) = (pointer_pos, state.drag.clone()) {
            let (mx, my) = to_image(sp);
            let x0 = p0.0.min(mx);
            let y0 = p0.1.min(my);
            let x1 = p0.0.max(mx);
            let y1 = p0.1.max(my);
            if (x1 - x0) > 4.0 && (y1 - y0) > 4.0 {
                let rect = crate::ui::editor::state::new_rect("object", x0, y0, x1 - x0, y1 - y0);
                shapes.push(rect);
                state.selected_shape = Some(shapes.len() - 1);
                state.label_buffer = "object".to_string();
                state.shape_type_buffer = "rectangle".to_string();
                state.dirty = true;
            }
        }
        state.drag = None;
    }

    // ── Single-click selection (no drag) ────────────────────────────────────
    if resp.clicked() && state.drag.is_none() {
        if let Some(sp) = pointer_pos {
            let (mx, my) = to_image(sp);
            let hit = hit_test(shapes, mx, my, scale, state.selected_shape);
            match hit {
                HitTarget::Body(i) => {
                    state.selected_shape = Some(i);
                    state.label_buffer = shapes[i].label.clone();
                    state.shape_type_buffer = shapes[i].shape_type.clone();
                }
                HitTarget::None => state.selected_shape = None,
                _ => {} // corner/rotate already selected
            }
        }
    }

    // ── Delete selected shape with Delete/Backspace ─────────────────────────
    if delete_pressed {
        if let Some(idx) = state.selected_shape.take() {
            if idx < shapes.len() {
                shapes.remove(idx);
                state.dirty = true;
            }
        }
    }
}
