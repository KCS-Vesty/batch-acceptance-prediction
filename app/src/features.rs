use crate::image_features::{aggregate_arrays, extract_for_file};
use crate::img_feat_cache::ImgFeatCache;
use crate::types::{FileEntry, IMAGE_SIZE};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// JSON-derived geometry features + aggregated image features for a single row.
/// Either may be `None` when the upstream step (annotation parse, JPG decode,
/// cache miss + extraction failure) couldn't produce them.
#[derive(Debug, Clone, Default)]
pub struct RowFeatures {
    pub json7: Option<[f64; 7]>,
    pub img6: Option<[f64; 6]>,
}

/// Compute JSON-derived annotation features from an already-parsed `AnnotationFile`.
/// Returns 7 features: [n_shapes, coverage, avg_area, std_area, avg_width, avg_height, dir_std].
pub fn json_annotation_features_from_ann(
    ann: &crate::annotation::AnnotationFile,
) -> Option<[f64; 7]> {
    let shapes = &ann.shapes;
    let n = shapes.len();
    if n == 0 {
        return Some([0.0; 7]);
    }

    let mut areas: Vec<f64> = Vec::with_capacity(n);
    let mut widths: Vec<f64> = Vec::with_capacity(n);
    let mut heights: Vec<f64> = Vec::with_capacity(n);
    let mut dirs: Vec<f64> = Vec::with_capacity(n);

    for shape in shapes {
        if shape.points.len() < 2 {
            continue;
        }
        let p0 = &shape.points[0];
        let p1 = &shape.points[1];
        let w = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
        let h = if shape.points.len() > 2 {
            let p2 = &shape.points[2];
            ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt()
        } else {
            w
        };
        areas.push(w * h);
        widths.push(w);
        heights.push(h);
        let d = shape.direction.unwrap_or(0.0).abs();
        dirs.push(d);
    }

    if areas.is_empty() {
        return Some([n as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    let img_area = IMAGE_SIZE * IMAGE_SIZE;
    let nf = n as f64;
    let mean_areas = areas.iter().sum::<f64>() / areas.len() as f64;
    let mean_widths = widths.iter().sum::<f64>() / widths.len() as f64;
    let mean_heights = heights.iter().sum::<f64>() / heights.len() as f64;
    let std_areas = if n > 1 {
        let m = mean_areas;
        (areas.iter().map(|x| (x - m).powi(2)).sum::<f64>() / areas.len() as f64).sqrt()
    } else {
        0.0
    };
    let std_dirs = if n > 1 {
        let m = dirs.iter().sum::<f64>() / dirs.len() as f64;
        (dirs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / dirs.len() as f64).sqrt()
    } else {
        0.0
    };

    Some([
        nf,
        areas.iter().sum::<f64>() / img_area,
        mean_areas / img_area,
        std_areas / img_area,
        mean_widths / IMAGE_SIZE,
        mean_heights / IMAGE_SIZE,
        std_dirs,
    ])
}

/// Load annotations (in parallel), then aggregate per-shape image features
/// through the cache. Both labeled and unprocessed pipelines share this so
/// we read each `.json` exactly once and cache+aggregate with the same logic.
///
/// Returns one `RowFeatures` per stem in the same order. `n_load_fail` counts
/// rows that had a `.json` path in the index but failed to parse — surfaced
/// in the log so silent JSON breakage is visible.
pub fn load_row_features(
    stems: &[&Path],
    file_index: &HashMap<std::path::PathBuf, FileEntry>,
    img_cache: &mut ImgFeatCache,
) -> (Vec<RowFeatures>, usize) {
    let anns: Vec<Option<crate::annotation::AnnotationFile>> = stems
        .par_iter()
        .map(|stem| {
            let jp = file_index.get(*stem)?.json.as_ref()?;
            crate::annotation::load(jp).ok()
        })
        .collect();

    let n_load_fail = anns
        .iter()
        .zip(stems.iter())
        .filter(|(opt, stem)| {
            opt.is_none()
                && file_index
                    .get(**stem)
                    .and_then(|e| e.json.as_ref())
                    .is_some()
        })
        .count();

    let rows: Vec<RowFeatures> = stems
        .iter()
        .zip(anns)
        .map(|(stem, ann_opt)| {
            let json7 = ann_opt.as_ref().and_then(json_annotation_features_from_ann);
            let img6 = resolve_img6(stem, &ann_opt, file_index, img_cache);
            RowFeatures { json7, img6 }
        })
        .collect();

    (rows, n_load_fail)
}

/// Resolve the 6-dim image feature vector for one stem, or `None` when any
/// prerequisite is missing (no entry, no JPG, no JSON, no annotation, or cache
/// miss + extraction failure).
fn resolve_img6(
    stem: &Path,
    ann_opt: &Option<crate::annotation::AnnotationFile>,
    file_index: &HashMap<std::path::PathBuf, FileEntry>,
    img_cache: &mut ImgFeatCache,
) -> Option<[f64; 6]> {
    let entry = file_index.get(stem)?;
    let jpg = entry.jpg.as_ref()?;
    let json = entry.json.as_ref()?;
    let ann = ann_opt.as_ref()?;
    let per_shape = img_cache
        .get_or_insert(jpg, json, |path| {
            let feats = extract_for_file(path, &ann.shapes)?;
            Ok(feats.iter().map(|f| f.to_array()).collect())
        })
        .ok()?;
    Some(aggregate_arrays(&per_shape))
}

/// Concatenate the three feature groups into the canonical 14-element feature
/// vector used by logistic regression: indices 0-6 = JSON geometry, 7-12 =
/// image-derived, 13 = conditional prior.
pub fn build_feature_vector(json7: [f64; 7], img6: [f64; 6], p_cat: f64) -> Vec<f64> {
    let mut f = Vec::with_capacity(14);
    f.extend_from_slice(&json7);
    f.extend_from_slice(&img6);
    f.push(p_cat);
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_annotation_features_empty_shapes() {
        let ann = crate::annotation::AnnotationFile {
            raw: serde_json::Map::new(),
            shapes: vec![],
        };
        let f = json_annotation_features_from_ann(&ann).unwrap();
        assert_eq!(f, [0.0; 7]);
    }

    #[test]
    fn test_json_annotation_features_single_shape() {
        let ann = crate::annotation::AnnotationFile {
            raw: serde_json::Map::new(),
            shapes: vec![crate::annotation::Shape {
                label: "text".into(),
                shape_type: "rectangle".into(),
                points: vec![(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
                direction: Some(0.0),
                extra: serde_json::Map::new(),
            }],
        };
        let f = json_annotation_features_from_ann(&ann).unwrap();
        assert_eq!(f[0], 1.0); // n_shapes = 1
        // coverage = area / img_area = 5000 / (640*640)
        let expected_coverage = 5000.0 / (640.0 * 640.0);
        assert!((f[1] - expected_coverage).abs() < 1e-9);
        // avg_area / img_area
        assert!((f[2] - expected_coverage).abs() < 1e-9);
        // std_area = 0 (single shape)
        assert!((f[3]).abs() < 1e-9);
        // mean_width / IMAGE_SIZE = 100/640
        assert!((f[4] - 100.0 / 640.0).abs() < 1e-9);
        // mean_height / IMAGE_SIZE = 50/640
        assert!((f[5] - 50.0 / 640.0).abs() < 1e-9);
        // std_dirs = 0 (single shape, direction=0)
        assert!((f[6]).abs() < 1e-9);
    }

    #[test]
    fn test_json_annotation_features_shape_with_few_points() {
        // Shape with only 1 point (degenerate) → skipped, areas empty
        let ann = crate::annotation::AnnotationFile {
            raw: serde_json::Map::new(),
            shapes: vec![crate::annotation::Shape {
                label: "point".into(),
                shape_type: "point".into(),
                points: vec![(10.0, 20.0)],
                direction: None,
                extra: serde_json::Map::new(),
            }],
        };
        let f = json_annotation_features_from_ann(&ann).unwrap();
        // n=1 but areas empty → returns [1, 0, 0, 0, 0, 0, 0]
        assert_eq!(f[0], 1.0);
        assert_eq!(f[1], 0.0);
    }

    #[test]
    fn test_json_annotation_features_direction_std() {
        use std::f64::consts::PI;
        let ann = crate::annotation::AnnotationFile {
            raw: serde_json::Map::new(),
            shapes: vec![
                crate::annotation::Shape {
                    label: "a".into(),
                    shape_type: "rectangle".into(),
                    points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                    direction: Some(0.0),
                    extra: serde_json::Map::new(),
                },
                crate::annotation::Shape {
                    label: "b".into(),
                    shape_type: "rectangle".into(),
                    points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                    direction: Some(PI / 2.0),
                    extra: serde_json::Map::new(),
                },
            ],
        };
        let f = json_annotation_features_from_ann(&ann).unwrap();
        assert_eq!(f[0], 2.0); // n_shapes = 2
        // std_dirs should be non-zero since directions differ
        assert!(f[6] > 0.0);
    }
}
