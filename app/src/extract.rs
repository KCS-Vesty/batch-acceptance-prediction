use crate::tier::{classify, Cutoffs};
use crate::types::{Prediction, RiskTier};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Extract options & result (pure API — stays in extract.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub tiers: Vec<RiskTier>,
    pub include_jpg: bool,
    pub include_txt: bool,
    pub move_files: bool,
    pub cutoffs: Cutoffs,
}

#[derive(Debug, Default)]
pub struct ExtractResult {
    pub transferred: Vec<PathBuf>,
    pub errors: Vec<(PathBuf, String)>,
    pub skipped: usize,
}

// ---------------------------------------------------------------------------
// Core extract logic
// ---------------------------------------------------------------------------

pub fn extract(
    predictions: &[Prediction],
    dest: &Path,
    opts: &ExtractOptions,
) -> std::io::Result<ExtractResult> {
    std::fs::create_dir_all(dest)?;
    let mut result = ExtractResult::default();

    for p in predictions {
        let tier = classify(p.p_combined, opts.cutoffs);
        if !opts.tiers.contains(&tier) {
            continue;
        }
        if p.json_path.as_os_str().is_empty() {
            result.skipped += 1;
            continue;
        }

        // The .json is the anchor — if its filename can't be derived we
        // can't transfer the row at all, so skip the whole triplet.
        if p.json_path.file_name().is_none() {
            result.skipped += 1;
            continue;
        }
        transfer_one(&p.json_path, dest, opts.move_files, &mut result);
        if opts.include_jpg {
            if let Some(jpg) = &p.jpg_path {
                transfer_one(jpg, dest, opts.move_files, &mut result);
            }
        }
        if opts.include_txt {
            if let Some(txt) = &p.txt_path {
                transfer_one(txt, dest, opts.move_files, &mut result);
            }
        }
    }

    Ok(result)
}

/// Resolve a unique destination path under `dest`, then move/copy `src` and
/// fold the outcome into `result`. Sources without a filename are silently
/// dropped — callers gate the .json on `file_name()` to surface that as a
/// skip instead.
fn transfer_one(src: &Path, dest: &Path, move_files: bool, result: &mut ExtractResult) {
    let Some(name) = src.file_name() else { return };
    let dst = unique_destination(dest, name);
    match transfer(src, &dst, move_files) {
        Ok(()) => result.transferred.push(dst),
        Err(e) => result.errors.push((src.to_path_buf(), e.to_string())),
    }
}

fn unique_destination(dest_dir: &Path, name: &std::ffi::OsStr) -> PathBuf {
    let primary = dest_dir.join(name);
    if !primary.exists() {
        return primary;
    }
    let stem = Path::new(name).file_stem().unwrap_or(name).to_os_string();
    let ext = Path::new(name)
        .extension()
        .map(|e| e.to_os_string())
        .unwrap_or_default();
    for i in 1..10_000 {
        let mut candidate = stem.clone();
        candidate.push(format!("_{}", i));
        if !ext.is_empty() {
            candidate.push(".");
            candidate.push(&ext);
        }
        let p = dest_dir.join(candidate);
        if !p.exists() {
            return p;
        }
    }
    primary
}

fn transfer(src: &Path, dst: &Path, move_files: bool) -> std::io::Result<()> {
    if move_files {
        match std::fs::rename(src, dst) {
            Ok(()) => Ok(()),
            Err(_) => {
                std::fs::copy(src, dst)?;
                std::fs::remove_file(src)?;
                Ok(())
            }
        }
    } else {
        std::fs::copy(src, dst).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Prediction;
    use std::path::PathBuf;

    fn dummy_prediction(json_path: &str, p_combined: f64) -> Prediction {
        Prediction {
            filename: "test.json".into(),
            batch: 1,
            grade: "grade-1".into(),
            subject: "math".into(),
            doc_type: "homework".into(),
            p_categorical: 0.5,
            p_logreg: Some(0.5),
            p_combined,
            json_path: PathBuf::from(json_path),
            jpg_path: None,
            txt_path: None,
            manually_overridden: false,
        }
    }

    fn write_temp_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn test_extract_copies_json_only() {
        let temp = std::env::temp_dir().join("bap_test_extract_copy");
        let _ = std::fs::remove_dir_all(&temp);
        let src_dir = temp.join("src");
        let dst_dir = temp.join("dst");
        std::fs::create_dir_all(&src_dir).unwrap();

        let json_path = write_temp_file(&src_dir, "a.json", "{}");
        let pred = dummy_prediction(json_path.to_str().unwrap(), 0.1);

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: false,
            include_txt: false,
            move_files: false,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 1);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.skipped, 0);
        assert!(dst_dir.join("a.json").exists());
        // Source still present (copy, not move)
        assert!(src_dir.join("a.json").exists());

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_extract_moves_json() {
        let temp = std::env::temp_dir().join("bap_test_extract_move");
        let _ = std::fs::remove_dir_all(&temp);
        let src_dir = temp.join("src");
        let dst_dir = temp.join("dst");
        std::fs::create_dir_all(&src_dir).unwrap();

        let json_path = write_temp_file(&src_dir, "b.json", "{}");
        let pred = dummy_prediction(json_path.to_str().unwrap(), 0.1);

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: false,
            include_txt: false,
            move_files: true,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 1);
        assert!(dst_dir.join("b.json").exists());
        // Source removed (move)
        assert!(!src_dir.join("b.json").exists());

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_extract_respects_tier_filter() {
        let temp = std::env::temp_dir().join("bap_test_extract_tier");
        let _ = std::fs::remove_dir_all(&temp);
        let src_dir = temp.join("src");
        let dst_dir = temp.join("dst");
        std::fs::create_dir_all(&src_dir).unwrap();

        // p_combined=0.8 → Low tier (≥ 0.7)
        let json_path = write_temp_file(&src_dir, "high_quality.json", "{}");
        let pred = dummy_prediction(json_path.to_str().unwrap(), 0.8);

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: false,
            include_txt: false,
            move_files: false,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 0);
        assert!(!dst_dir.join("high_quality.json").exists());

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_extract_includes_jpg_and_txt_sidecars() {
        let temp = std::env::temp_dir().join("bap_test_extract_sidecars");
        let _ = std::fs::remove_dir_all(&temp);
        let src_dir = temp.join("src");
        let dst_dir = temp.join("dst");
        std::fs::create_dir_all(&src_dir).unwrap();

        let json_path = write_temp_file(&src_dir, "c.json", "{}");
        let jpg_path = write_temp_file(&src_dir, "c.jpg", "fake-jpg");
        let txt_path = write_temp_file(&src_dir, "c.txt", "log");

        let mut pred = dummy_prediction(json_path.to_str().unwrap(), 0.1);
        pred.jpg_path = Some(jpg_path);
        pred.txt_path = Some(txt_path);

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: true,
            include_txt: true,
            move_files: false,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 3);
        assert!(dst_dir.join("c.json").exists());
        assert!(dst_dir.join("c.jpg").exists());
        assert!(dst_dir.join("c.txt").exists());

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_extract_collision_renames() {
        let temp = std::env::temp_dir().join("bap_test_extract_collision");
        let _ = std::fs::remove_dir_all(&temp);
        let src_dir = temp.join("src");
        let dst_dir = temp.join("dst");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::create_dir_all(&dst_dir).unwrap();

        // Pre-create destination file to force collision
        write_temp_file(&dst_dir, "d.json", "existing");

        let json_path = write_temp_file(&src_dir, "d.json", "new");
        let pred = dummy_prediction(json_path.to_str().unwrap(), 0.1);

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: false,
            include_txt: false,
            move_files: false,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 1);
        assert!(dst_dir.join("d.json").exists());
        assert!(dst_dir.join("d_1.json").exists());

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_extract_skips_empty_json_path() {
        let temp = std::env::temp_dir().join("bap_test_extract_skip");
        let _ = std::fs::remove_dir_all(&temp);
        let dst_dir = temp.join("dst");

        let pred = dummy_prediction("", 0.1); // empty path → skipped

        let opts = ExtractOptions {
            tiers: vec![RiskTier::High],
            include_jpg: false,
            include_txt: false,
            move_files: false,
            cutoffs: crate::tier::Cutoffs {
                high_cutoff: 0.3,
                low_cutoff: 0.7,
            },
        };

        let result = extract(&[pred], &dst_dir, &opts).unwrap();
        assert_eq!(result.transferred.len(), 0);
        assert_eq!(result.skipped, 1);

        std::fs::remove_dir_all(&temp).ok();
    }
}
