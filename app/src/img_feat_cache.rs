//! Binary (bincode) cache for per-shape image-feature vectors.
//!
//! Keyed by the JPEG path. Each value is `(jpg_mtime, json_mtime, feats)` so
//! the cache invalidates whenever either the image or its annotation changes
//! on disk — the annotation editor's `save()` bumps the JSON's mtime, which
//! correctly forces a recompute on the next pipeline run.
//!
//! The file is prefixed by a `u32` version number. If `image_features.rs`
//! changes its per-shape feature layout, bump `CACHE_VERSION` — older
//! cache files are then treated as missing rather than returning stale data.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Bump whenever the per-shape feature layout in `image_features.rs` changes.
const CACHE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheVal {
    pub jpg_mtime: u64,
    pub json_mtime: u64,
    /// Per-shape packed features, layout defined by
    /// `image_features::ShapeImageFeats::to_array`.
    pub feats: Vec<[f64; 4]>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ImgFeatCache {
    pub entries: HashMap<PathBuf, CacheVal>,
}

/// Load a cache from `<dataset_root>/.qa_img_cache.bin`. Returns an empty
/// cache on missing file, version mismatch, or any deserialization error.
pub fn load(dataset_root: &Path) -> Result<ImgFeatCache> {
    let path = dataset_root.join(".qa_img_cache.bin");
    if !path.exists() {
        return Ok(ImgFeatCache::default());
    }
    let file = File::open(&path).with_context(|| format!("opening {:?}", path))?;
    let mut reader = BufReader::new(file);
    let version: u32 = match bincode::deserialize_from(&mut reader) {
        Ok(v) => v,
        Err(_) => return Ok(ImgFeatCache::default()),
    };
    if version != CACHE_VERSION {
        return Ok(ImgFeatCache::default());
    }
    let entries: HashMap<PathBuf, CacheVal> = match bincode::deserialize_from(&mut reader) {
        Ok(e) => e,
        Err(_) => return Ok(ImgFeatCache::default()),
    };
    Ok(ImgFeatCache { entries })
}

impl ImgFeatCache {
    pub fn save(&self, dataset_root: &Path) -> Result<()> {
        let path = dataset_root.join(".qa_img_cache.bin");
        let file = File::create(&path).with_context(|| format!("creating {:?}", path))?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &CACHE_VERSION)
            .with_context(|| format!("writing cache version to {:?}", path))?;
        bincode::serialize_into(&mut writer, &self.entries)
            .with_context(|| format!("serializing cache entries to {:?}", path))?;
        writer.flush()?;
        Ok(())
    }

    /// Return cached per-shape features for `jpg`, or compute and store them
    /// via `extract_fn`. A hit requires both `jpg` and `json` mtimes to match.
    pub fn get_or_insert<F>(
        &mut self,
        jpg: &Path,
        json: &Path,
        extract_fn: F,
    ) -> Result<Vec<[f64; 4]>>
    where
        F: FnOnce(&Path) -> Result<Vec<[f64; 4]>>,
    {
        let key = jpg.canonicalize().unwrap_or_else(|_| jpg.to_path_buf());
        let jpg_mtime = mtime_secs(jpg);
        let json_mtime = mtime_secs(json);

        if let Some(v) = self.entries.get(&key) {
            if v.jpg_mtime == jpg_mtime && v.json_mtime == json_mtime {
                return Ok(v.feats.clone());
            }
        }

        let feats = extract_fn(jpg)?;
        self.entries.insert(
            key,
            CacheVal {
                jpg_mtime,
                json_mtime,
                feats: feats.clone(),
            },
        );
        Ok(feats)
    }
}

fn mtime_secs(p: &Path) -> u64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_empty() {
        let cache = ImgFeatCache::default();
        assert!(cache.entries.is_empty());
    }
}
