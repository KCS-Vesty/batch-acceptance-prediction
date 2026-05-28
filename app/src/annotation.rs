//! LabelMe annotation file loader and saver.
//!
//! A LabelMe JSON looks like:
//! ```json
//! {
//!   "version": "5.0",
//!   "flags": {},
//!   "shapes": [ { ... }, { ... } ],
//!   "imagePath": "...",
//!   "imageHeight": 640,
//!   "imageWidth": 640
//! }
//! ```
//!
//! We round-trip the file faithfully: the top-level fields (everything except
//! `shapes`) are kept in `AnnotationFile::raw`, and each shape's
//! unrecognised fields (e.g. `score`, `kie_linking`, `group_id`, `description`,
//! `difficult`) are captured into `Shape::extra` via `#[serde(flatten)]`.
//! Saving patches the modified shapes back into the raw object.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

/// A single shape within a LabelMe annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    /// User-assigned label string.
    pub label: String,

    /// Shape type, e.g. "rectangle", "polygon", "line", "point", "rotation".
    pub shape_type: String,

    /// Polygon / bounding-box / line points as `(x, y)` pixel coordinates.
    pub points: Vec<(f64, f64)>,

    /// Rotation angle (LabelMe stores `direction`; absent for non-rotated shapes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<f64>,

    /// Any other fields LabelMe writes (`kie_linking`, `group_id`,
    /// `description`, `difficult`, `score`, …) are kept verbatim so save
    /// round-trips losslessly.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Wrapper around a LabelMe JSON file.
///
/// `raw` holds the top-level object **minus** `shapes`; `shapes` is the parsed
/// list. On save we re-emit `raw` with `shapes` patched back in, preserving
/// every other top-level field (`version`, `imageWidth`, `imageHeight`,
/// `imagePath`, `flags`, …).
#[derive(Debug, Clone)]
pub struct AnnotationFile {
    pub raw: serde_json::Map<String, serde_json::Value>,
    pub shapes: Vec<Shape>,
}

/// Load a LabelMe JSON annotation file.
pub fn load(path: &Path) -> Result<AnnotationFile> {
    let txt = fs::read_to_string(path).with_context(|| format!("reading {:?}", path))?;
    let mut top: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&txt).with_context(|| format!("parsing {:?}", path))?;

    let shapes_value = top
        .remove("shapes")
        .unwrap_or(serde_json::Value::Array(Vec::new()));
    let shapes: Vec<Shape> = serde_json::from_value(shapes_value)
        .with_context(|| format!("deserializing shapes[] in {:?}", path))?;

    Ok(AnnotationFile { raw: top, shapes })
}

/// Save an `AnnotationFile` back to disk.
///
/// Patches the `shapes` array into `ann.raw` and writes the result with
/// 2-space indentation (LabelMe's style).
pub fn save(path: &Path, ann: &AnnotationFile) -> Result<()> {
    let mut top = ann.raw.clone();
    top.insert(
        "shapes".to_string(),
        serde_json::to_value(&ann.shapes)
            .with_context(|| format!("serializing shapes for {:?}", path))?,
    );
    let out = serde_json::to_string_pretty(&serde_json::Value::Object(top))?;
    let mut file = fs::File::create(path).with_context(|| format!("creating {:?}", path))?;
    file.write_all(out.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_serde_roundtrip() {
        let json = r#"{
            "label": "Line",
            "shape_type": "rotation",
            "points": [[10.0, 20.0], [100.0, 200.0], [100.0, 250.0], [10.0, 250.0]],
            "direction": 0.0,
            "difficult": false,
            "kie_linking": [],
            "group_id": null,
            "score": null,
            "description": ""
        }"#;
        let s: Shape = serde_json::from_str(json).unwrap();
        assert_eq!(s.label, "Line");
        assert_eq!(s.shape_type, "rotation");
        assert_eq!(s.points.len(), 4);
        assert_eq!(s.direction, Some(0.0));
        // Extra fields preserved.
        assert!(s.extra.contains_key("kie_linking"));
        assert!(s.extra.contains_key("group_id"));
        assert!(s.extra.contains_key("description"));
        assert!(s.extra.contains_key("difficult"));
    }

    #[test]
    fn test_annotation_file_load_save_roundtrip() {
        let json = r#"{
  "version": "5.0",
  "flags": {},
  "shapes": [
    {
      "label": "Line",
      "shape_type": "rotation",
      "points": [[10.0, 20.0], [100.0, 200.0]],
      "direction": 0.0,
      "difficult": false,
      "kie_linking": [],
      "group_id": null
    }
  ],
  "imagePath": "x.jpg",
  "imageHeight": 640,
  "imageWidth": 640
}"#;

        // Parse via the same machinery `load` uses.
        let mut top: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(json).unwrap();
        let shapes_value = top.remove("shapes").unwrap();
        let shapes: Vec<Shape> = serde_json::from_value(shapes_value).unwrap();
        let ann = AnnotationFile { raw: top, shapes };

        assert_eq!(ann.shapes.len(), 1);
        assert_eq!(ann.raw.get("imageWidth").and_then(|v| v.as_i64()), Some(640));

        // Re-serialize and confirm the original top-level fields survive.
        let mut top2 = ann.raw.clone();
        top2.insert(
            "shapes".to_string(),
            serde_json::to_value(&ann.shapes).unwrap(),
        );
        let out = serde_json::to_string_pretty(&serde_json::Value::Object(top2)).unwrap();
        assert!(out.contains("\"imageWidth\""));
        assert!(out.contains("\"version\""));
        assert!(out.contains("\"kie_linking\""));
    }
}
