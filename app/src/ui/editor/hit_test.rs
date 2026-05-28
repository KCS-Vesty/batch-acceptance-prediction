//! Hit-testing engine for the annotation editor.
//!
//! Translates screen-space mouse coordinates into logical hit targets
//! (shape body, corner handle, rotation handle) in image coordinates.

use crate::annotation::Shape;

/// Result of a hit-test: what the mouse is over at a given image-space point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HitTarget {
    /// Nothing — empty canvas space.
    None,
    /// Mouse is inside the body of shape `usize`.
    Body(usize),
    /// Mouse is on corner `u8` of shape `usize`.
    Corner { shape: usize, corner: u8 },
    /// Mouse is on the rotation handle of shape `usize`.
    Rotate { shape: usize },
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Bounding-box centre of the shape's points. Returns `(x, y)` in image space.
pub(crate) fn image_center(shape: &Shape) -> Option<(f64, f64)> {
    if shape.points.is_empty() {
        return None;
    }
    let n = shape.points.len() as f64;
    let cx = shape.points.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = shape.points.iter().map(|p| p.1).sum::<f64>() / n;
    Some((cx, cy))
}

fn dist2(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

/// Point-in-polygon for shapes with ≥3 points (even-odd ray casting).
/// Lines and points fall back to a generous AABB hit so they're still
/// selectable.
fn point_in_shape(mx: f64, my: f64, shape: &Shape) -> bool {
    let pts = &shape.points;
    if pts.len() >= 3 {
        let n = pts.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = pts[i];
            let (xj, yj) = pts[j];
            if (yi > my) != (yj > my)
                && mx < (xj - xi) * (my - yi) / (yj - yi + 1e-12) + xi
            {
                inside = !inside;
            }
            j = i;
        }
        return inside;
    }
    if pts.is_empty() {
        return false;
    }
    // ≤ 2 points: line segment or a single point — give it a small grab pad.
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for &(x, y) in pts {
        if x < xmin { xmin = x; }
        if x > xmax { xmax = x; }
        if y < ymin { ymin = y; }
        if y > ymax { ymax = y; }
    }
    let pad = 4.0;
    mx >= xmin - pad && mx <= xmax + pad && my >= ymin - pad && my <= ymax + pad
}

/// Hit-test in image coordinates. `selected` is only consulted to enable
/// corner/rotation handle picking — they only exist on the selected shape.
pub(crate) fn hit_test(
    shapes: &[Shape],
    mx: f64,
    my: f64,
    scale: f32,
    selected: Option<usize>,
) -> HitTarget {
    let handle_hit_radius = (12.0 / scale) as f64;

    // Handles on the selected shape take priority over body hits.
    if let Some(sel) = selected {
        if let Some(shape) = shapes.get(sel) {
            // Rotation handle.
            if let Some((cx, cy)) = image_center(shape) {
                let angle = shape.direction.unwrap_or(0.0);
                let r = ROTATE_HANDLE_OFFSET as f64;
                let hx = cx + r * angle.sin();
                let hy = cy - r * angle.cos();
                if dist2(mx, my, hx, hy) < handle_hit_radius.powi(2) {
                    return HitTarget::Rotate { shape: sel };
                }
            }
            // Corner handles.
            for (ci, (px, py)) in shape.points.iter().enumerate() {
                if dist2(mx, my, *px, *py) < handle_hit_radius.powi(2) {
                    return HitTarget::Corner {
                        shape: sel,
                        corner: ci as u8,
                    };
                }
            }
        }
    }

    // Body hits (front-to-back: last drawn = topmost).
    for (i, shape) in shapes.iter().enumerate().rev() {
        if point_in_shape(mx, my, shape) {
            return HitTarget::Body(i);
        }
    }

    HitTarget::None
}

/// Distance offset of the rotation handle from the box centre (image-space pixels).
pub(crate) const ROTATE_HANDLE_OFFSET: f32 = 40.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::Shape;

    fn make_shape(points: Vec<(f64, f64)>) -> Shape {
        Shape {
            label: "test".into(),
            shape_type: "rectangle".into(),
            points,
            direction: None,
            extra: serde_json::Map::new(),
        }
    }

    fn make_rect(x: f64, y: f64, w: f64, h: f64) -> Shape {
        make_shape(vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    }

    #[test]
    fn test_image_center_square() {
        let s = make_rect(0.0, 0.0, 100.0, 100.0);
        let (cx, cy) = image_center(&s).unwrap();
        assert!((cx - 50.0).abs() < 1e-9);
        assert!((cy - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_image_center_empty() {
        let s = make_shape(vec![]);
        assert!(image_center(&s).is_none());
    }

    #[test]
    fn test_image_center_single_point() {
        let s = make_shape(vec![(42.0, 58.0)]);
        let (cx, cy) = image_center(&s).unwrap();
        assert!((cx - 42.0).abs() < 1e-9);
        assert!((cy - 58.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_in_shape_rect_inside() {
        let s = make_rect(10.0, 10.0, 80.0, 60.0);
        assert!(point_in_shape(50.0, 40.0, &s));
    }

    #[test]
    fn test_point_in_shape_rect_outside() {
        let s = make_rect(10.0, 10.0, 80.0, 60.0);
        assert!(!point_in_shape(0.0, 0.0, &s));
        assert!(!point_in_shape(200.0, 200.0, &s));
    }

    #[test]
    fn test_point_in_shape_rect_on_edge() {
        // Points exactly on the edge may be inside or outside depending on
        // the ray-casting implementation — just verify it doesn't panic.
        let s = make_rect(0.0, 0.0, 100.0, 100.0);
        let _ = point_in_shape(0.0, 50.0, &s);
        let _ = point_in_shape(100.0, 50.0, &s);
    }

    #[test]
    fn test_point_in_shape_empty() {
        let s = make_shape(vec![]);
        assert!(!point_in_shape(0.0, 0.0, &s));
    }

    #[test]
    fn test_point_in_shape_single_point() {
        let s = make_shape(vec![(10.0, 20.0)]);
        // Within grab pad (4px)
        assert!(point_in_shape(10.0, 20.0, &s));
        assert!(point_in_shape(12.0, 22.0, &s));
        // Outside grab pad
        assert!(!point_in_shape(20.0, 30.0, &s));
    }

    #[test]
    fn test_point_in_shape_line() {
        let s = make_shape(vec![(0.0, 0.0), (100.0, 0.0)]);
        // Within grab pad of the line
        assert!(point_in_shape(50.0, 2.0, &s));
        // Far from the line
        assert!(!point_in_shape(50.0, 50.0, &s));
    }

    #[test]
    fn test_hit_test_body_priority_front_to_back() {
        let shapes = vec![
            make_rect(0.0, 0.0, 100.0, 100.0),   // index 0 (back)
            make_rect(10.0, 10.0, 80.0, 80.0),    // index 1 (front)
        ];
        // Point inside both → frontmost (index 1) wins
        let hit = hit_test(&shapes, 50.0, 50.0, 1.0, None);
        assert_eq!(hit, HitTarget::Body(1));
    }

    #[test]
    fn test_hit_test_none() {
        let shapes = vec![make_rect(0.0, 0.0, 10.0, 10.0)];
        let hit = hit_test(&shapes, 200.0, 200.0, 1.0, None);
        assert_eq!(hit, HitTarget::None);
    }

    #[test]
    fn test_hit_test_corner_on_selected() {
        let shapes = vec![make_rect(0.0, 0.0, 100.0, 100.0)];
        // Corner 0 is at (0, 0). With scale=1, handle_hit_radius = 12.
        let hit = hit_test(&shapes, 2.0, 2.0, 1.0, Some(0));
        assert_eq!(hit, HitTarget::Corner { shape: 0, corner: 0 });
    }

    #[test]
    fn test_hit_test_corner_not_on_unselected() {
        let shapes = vec![make_rect(0.0, 0.0, 100.0, 100.0)];
        // Without selection, corner handles are not active → body hit
        let hit = hit_test(&shapes, 2.0, 2.0, 1.0, None);
        assert_eq!(hit, HitTarget::Body(0));
    }

    #[test]
    fn test_hit_test_rotate_handle() {
        let mut s = make_rect(0.0, 0.0, 100.0, 100.0);
        s.direction = Some(0.0); // angle = 0 → handle at center + (0, -40)
        let shapes = vec![s];
        let (cx, cy) = image_center(&shapes[0]).unwrap();
        // Rotation handle is at (cx + 40*sin(0), cy - 40*cos(0)) = (cx, cy - 40)
        let hy = cy - ROTATE_HANDLE_OFFSET as f64;
        let hit = hit_test(&shapes, cx, hy, 1.0, Some(0));
        assert_eq!(hit, HitTarget::Rotate { shape: 0 });
    }

    #[test]
    fn test_hit_test_handles_priority_over_body() {
        let shapes = vec![make_rect(0.0, 0.0, 100.0, 100.0)];
        // Point at corner 0 (0,0) with selection → corner hit, not body
        let hit = hit_test(&shapes, 0.0, 0.0, 1.0, Some(0));
        assert!(matches!(hit, HitTarget::Corner { .. }));
    }

    #[test]
    fn test_hit_test_empty_shapes() {
        let shapes: Vec<Shape> = vec![];
        let hit = hit_test(&shapes, 50.0, 50.0, 1.0, None);
        assert_eq!(hit, HitTarget::None);
    }

    #[test]
    fn test_dist2() {
        assert!((dist2(0.0, 0.0, 3.0, 4.0) - 25.0).abs() < 1e-9);
        assert!((dist2(1.0, 1.0, 1.0, 1.0)).abs() < 1e-9);
    }
}