//! Editor state types and construction helpers.
//!
//! The `EditorState` struct holds all mutable UI state for the annotation
//! editor window. `DragOp` describes an in-flight interaction.

use std::path::Path;

use eframe::egui;

use crate::annotation::{self, AnnotationFile, Shape};

// ---------------------------------------------------------------------------
// Drag operation
// ---------------------------------------------------------------------------

/// Drag operation in progress. `start_*` fields snapshot the pre-drag state so
/// each frame computes the new position from a stable origin (no rounding drift).
#[derive(Debug, Clone)]
pub enum DragOp {
    Translate {
        start_mouse: egui::Pos2,
        start_points: Vec<(f64, f64)>,
    },
    Resize {
        corner: u8,
        start_mouse: egui::Pos2,
        start_point: (f64, f64),
    },
    Rotate {
        start_mouse: egui::Pos2,
        start_points: Vec<(f64, f64)>,
        start_angle: f64,
        center: (f64, f64),
    },
    Draw {
        p0: (f64, f64),
    },
}

// ---------------------------------------------------------------------------
// Editor state
// ---------------------------------------------------------------------------

/// Persistent state for the floating annotation editor window.
///
/// Lives on `App::editor_state`. `std::mem::take`d out of `App` for
/// the duration of each frame so the UI can freely mutate both `state`
/// and `app` without borrow checker conflicts.
pub struct EditorState {
    pub open: bool,
    pub current: Option<usize>,
    pub annotation: Option<AnnotationFile>,
    pub texture: Option<egui::TextureHandle>,
    pub selected_shape: Option<usize>,
    pub drag: Option<DragOp>,
    pub label_buffer: String,
    pub shape_type_buffer: String,
    pub dirty: bool,
    /// Last status message (e.g. "Saved", "Marked accept"). Cleared on next open.
    pub last_status_msg: Option<String>,
    /// One-step undo snapshot taken just before each destructive change.
    /// Ctrl-Z restores it.
    pub undo_snapshot: Option<Vec<Shape>>,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            open: false,
            current: None,
            annotation: None,
            texture: None,
            selected_shape: None,
            drag: None,
            label_buffer: String::new(),
            shape_type_buffer: "rotation".to_string(),
            dirty: false,
            last_status_msg: None,
            undo_snapshot: None,
        }
    }
}

impl EditorState {
    /// Open the editor for prediction row `idx` whose JSON lives at `json_path`.
    /// Image texture is uploaded lazily on first paint.
    pub fn open_for(&mut self, idx: usize, json_path: &Path) {
        self.current = Some(idx);
        // Clear before load so a successful open shows a clean panel; a failed
        // load below replaces it with the error message.
        self.last_status_msg = None;
        match annotation::load(json_path) {
            Ok(ann) => self.annotation = Some(ann),
            Err(e) => {
                self.annotation = None;
                self.last_status_msg = Some(format!("Failed to load annotation: {}", e));
            }
        }
        self.texture = None;
        self.selected_shape = None;
        self.drag = None;
        self.label_buffer.clear();
        self.shape_type_buffer = "rotation".to_string();
        self.dirty = false;
        self.undo_snapshot = None;
        self.open = true;
    }
}

/// Capture a single-step undo snapshot of the current shapes. Overwrites
/// any prior snapshot — one-level undo only.
pub(crate) fn snapshot_shapes(state: &mut EditorState) {
    if let Some(ann) = &state.annotation {
        state.undo_snapshot = Some(ann.shapes.clone());
    }
}

/// Construct a fresh axis-aligned rectangle at `(x, y)` with given size.
pub(crate) fn new_rect(label: &str, x: f64, y: f64, w: f64, h: f64) -> Shape {
    Shape {
        label: label.to_string(),
        shape_type: "rectangle".to_string(),
        points: vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        direction: None,
        extra: serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::AnnotationFile;

    fn make_test_annotation() -> AnnotationFile {
        AnnotationFile {
            raw: serde_json::Map::new(),
            shapes: vec![],
        }
    }

    #[test]
    fn test_new_rect_shape() {
        let rect = new_rect("test", 10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.label, "test");
        assert_eq!(rect.shape_type, "rectangle");
        assert_eq!(rect.points.len(), 4);
        assert_eq!(rect.points[0], (10.0, 20.0));
        assert_eq!(rect.points[1], (110.0, 20.0));
        assert_eq!(rect.points[2], (110.0, 70.0));
        assert_eq!(rect.points[3], (10.0, 70.0));
        assert_eq!(rect.direction, None);
    }

    #[test]
    fn test_editor_state_default() {
        let state = EditorState::default();
        assert!(!state.open);
        assert!(state.current.is_none());
        assert!(state.annotation.is_none());
        assert!(state.texture.is_none());
        assert!(state.selected_shape.is_none());
        assert!(state.drag.is_none());
        assert!(state.label_buffer.is_empty());
        assert_eq!(state.shape_type_buffer, "rotation");
        assert!(!state.dirty);
        assert!(state.last_status_msg.is_none());
        assert!(state.undo_snapshot.is_none());
    }

    #[test]
    fn test_editor_state_open_for_loads_annotation() {
        let temp = std::env::temp_dir().join("bap_test_editor_open");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();

        // Write a valid annotation JSON
        let json = r#"{
  "version": "5.0",
  "flags": {},
  "shapes": [
    {
      "label": "text",
      "shape_type": "rectangle",
      "points": [[10.0, 20.0], [100.0, 20.0], [100.0, 60.0], [10.0, 60.0]],
      "direction": 0.0
    }
  ],
  "imagePath": "x.jpg",
  "imageHeight": 200,
  "imageWidth": 200
}"#;
        let json_path = temp.join("test.json");
        std::fs::write(&json_path, json).unwrap();

        let mut state = EditorState::default();
        state.open_for(0, &json_path);

        assert!(state.open);
        assert_eq!(state.current, Some(0));
        assert!(state.annotation.is_some());
        let ann = state.annotation.as_ref().unwrap();
        assert_eq!(ann.shapes.len(), 1);
        assert_eq!(ann.shapes[0].label, "text");
        assert!(state.texture.is_none());
        assert!(state.selected_shape.is_none());
        assert!(!state.dirty);

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_editor_state_open_for_missing_file() {
        let mut state = EditorState::default();
        state.open_for(0, std::path::Path::new("/nonexistent/path.json"));

        assert!(state.open);
        assert!(state.annotation.is_none());
        assert!(state.last_status_msg.is_some());
        assert!(state.last_status_msg.as_ref().unwrap().contains("Failed to load"));
    }

    #[test]
    fn test_snapshot_shapes() {
        let mut state = EditorState::default();
        let ann = make_test_annotation();
        state.annotation = Some(ann);

        // Initially no snapshot
        assert!(state.undo_snapshot.is_none());

        // Take snapshot
        snapshot_shapes(&mut state);
        assert!(state.undo_snapshot.is_some());
        assert_eq!(state.undo_snapshot.as_ref().unwrap().len(), 0);

        // Add a shape and snapshot again
        if let Some(ann) = &mut state.annotation {
            ann.shapes.push(new_rect("a", 0.0, 0.0, 10.0, 10.0));
        }
        snapshot_shapes(&mut state);
        assert_eq!(state.undo_snapshot.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_snapshot_shapes_no_annotation() {
        let mut state = EditorState::default();
        // No annotation — should not panic
        snapshot_shapes(&mut state);
        assert!(state.undo_snapshot.is_none());
    }
}
