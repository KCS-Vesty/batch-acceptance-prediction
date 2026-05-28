//! Shape rendering for the annotation editor.
//!
//! Draws shape outlines, corner handles, and rotation handles onto an egui
//! canvas. Pure paint-only logic — no state mutation.

use crate::annotation::Shape;
use crate::ui::{C_BLUE, C_GREEN, C_RED};
use super::hit_test::{image_center, ROTATE_HANDLE_OFFSET};
use eframe::egui;

/// Radius of the rotation handle circle (image-space pixels).
pub(crate) const ROTATE_HANDLE_R: f32 = 8.0;

/// Radius of corner resize handles (image-space pixels).
pub(crate) const CORNER_HANDLE_R: f32 = 6.0;

/// Whether the shape type forms a closed polygon (last point connects back to first).
fn is_closed_shape(shape_type: &str) -> bool {
    matches!(shape_type, "rectangle" | "polygon" | "rotation")
}

/// Paint labelled image regions and shape overlays for the annotation canvas.
pub(crate) fn draw_shapes<F>(
    painter: &egui::Painter,
    shapes: &[Shape],
    selected: Option<usize>,
    scale: f32,
    to_screen: &F,
) where
    F: Fn(f64, f64) -> egui::Pos2,
{
    for (i, shape) in shapes.iter().enumerate() {
        let pts: Vec<egui::Pos2> = shape
            .points
            .iter()
            .map(|(x, y)| to_screen(*x, *y))
            .collect();
        if pts.len() < 2 {
            continue;
        }

        let is_sel = selected == Some(i);
        let color = if is_sel { C_BLUE } else { C_GREEN };
        let stroke = egui::Stroke::new(if is_sel { 2.5 } else { 1.5 }, color);

        if is_closed_shape(&shape.shape_type) {
            for j in 0..pts.len() {
                let a = pts[j];
                let b = pts[(j + 1) % pts.len()];
                painter.line_segment([a, b], stroke);
            }
        } else {
            for j in 0..(pts.len() - 1) {
                painter.line_segment([pts[j], pts[j + 1]], stroke);
            }
        }

        // Handles for selected shape.
        if is_sel {
            let handle_r = (CORNER_HANDLE_R * scale).clamp(4.0, 10.0);
            for pt in &pts {
                painter.circle_stroke(*pt, handle_r, egui::Stroke::new(1.5, C_BLUE));
            }
            // Rotation handle: in image-space, above the box centre, then
            // converted to screen space exactly once.
            if let Some((cx, cy)) = image_center(shape) {
                let angle = shape.direction.unwrap_or(0.0);
                let r = ROTATE_HANDLE_OFFSET as f64;
                let hx = cx + r * angle.sin();
                let hy = cy - r * angle.cos();
                let h_screen = to_screen(hx, hy);
                let c_screen = to_screen(cx, cy);
                painter.line_segment(
                    [c_screen, h_screen],
                    egui::Stroke::new(1.0, C_RED),
                );
                let rot_r = (ROTATE_HANDLE_R * scale).clamp(5.0, 12.0);
                painter.circle_stroke(
                    h_screen,
                    rot_r,
                    egui::Stroke::new(1.5, C_RED),
                );
            }
        }
    }
}
