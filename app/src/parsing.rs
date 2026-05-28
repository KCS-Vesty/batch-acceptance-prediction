use crate::types::{Label, Meta, RejectReason};
use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

/// Regex to parse filename stems of the format:
/// `padded_china_(junior|senior)-high-school_(grade-N)_SUBJECT_DOCTYPE_TIMESTAMP_train_paraN`
/// or `..._test_paraN`.  The TIMESTAMP segment is variable-length digits.
pub fn stem_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"^padded_china_(?:junior|senior)-high-school_(grade-\d)_([^_]+)_([^_]+)_(\d+)_([^_]+)_para\d+\.json$",
        )
        .expect("stem regex")
    })
}

/// Regex to extract batch number from folder path:
/// `POC_P2_20000_XX` where XX is the batch number.
pub fn batch_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"POC_P2_20000_(\d+)").expect("batch regex"))
}

/// Parse a filename stem and return metadata including the batch number
/// extracted from the ancestor folder path (not the filename itself).
pub fn parse_stem(stem: &Path) -> Option<Meta> {
    let fname = stem.file_name()?.to_str()?;
    // build_file_index strips the extension from the stem, but our regex
    // expects `.json` at the end. Normalise so we match in both contexts.
    let fname_for_regex = if fname.ends_with(".json") {
        fname.to_string()
    } else {
        format!("{}.json", fname)
    };
    let caps = stem_re().captures(&fname_for_regex)?;

    // The batch number lives in the *ancestor* folder name
    // (e.g. `POC_P2_20000_16`), not the filename. Search the full path.
    let path_str = stem.to_str().unwrap_or("");
    let batch = batch_re()
        .captures(path_str)
        .and_then(|c| c.get(1).and_then(|m| m.as_str().parse::<i32>().ok()))
        .unwrap_or(-1);

    Some(Meta {
        stem: stem.to_path_buf(),
        grade: caps.get(1)?.as_str().to_string(),
        subject: caps.get(2)?.as_str().to_string(),
        doc_type: caps.get(3)?.as_str().to_string(),
        batch,
    })
}

/// Normalize a raw status string from a review log into a Label.
/// Handles both legacy short forms and explicit reject reasons.
pub fn normalize_status(raw: &str) -> Option<Label> {
    let s = raw.trim().to_lowercase();
    if s.starts_with("region-accept") {
        return Some(Label::Accept);
    }
    if s.starts_with("region-reject") {
        return Some(Label::Reject(RejectReason::Generic));
    }
    match s.as_str() {
        "accept" | "accept-cleared" => Some(Label::Accept),
        "reject" | "reject-cleared" => Some(Label::Reject(RejectReason::Generic)),
        "reject-whitespace" => Some(Label::Reject(RejectReason::Whitespace)),
        "reject-rotation" => Some(Label::Reject(RejectReason::Rotation)),
        "reject-structure" => Some(Label::Reject(RejectReason::Structure)),
        "cut off" => Some(Label::Reject(RejectReason::CutOff)),
        "manual-accept" => Some(Label::Accept),
        s if s.starts_with("manual-reject-") => {
            let reason = match s.strip_prefix("manual-reject-").unwrap_or("") {
                "whitespace" => RejectReason::Whitespace,
                "rotation" => RejectReason::Rotation,
                "structure" => RejectReason::Structure,
                "cutoff" | "cut-off" => RejectReason::CutOff,
                "generic" => RejectReason::Generic,
                _ => RejectReason::Manual,
            };
            Some(Label::Reject(reason))
        }
        _ => None,
    }
}

/// Parse a review-log `.txt` file and return the most recent valid label.
/// Returns `Some((label, is_manual))` where `is_manual` is true when the
/// most recent valid status line is a `manual-*` override (from the editor).
pub fn parse_txt(path: &Path) -> Option<(Label, bool)> {
    let content = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    for line in lines.iter().rev() {
        // Find the timestamp→status separator (the '-' after "hh:mm:ss").
        // Skip lines without a '-' entirely.
        if let Some(idx) = line.find('-') {
            let before = &line[..idx];
            // Guard: if the part before the first '-' has ≥ 2 colons, it looks
            // like "dd/mm/yyyy hh:mm:ss" (real timestamp) → use first hyphen.
            // If it has 0 colons (e.g. "2024-01-01"), it's not a timestamp line
            // → skip it. This prevents "2024-01-01 comment" lines from being
            // mis-parsed as status lines.
            if before.matches(':').count() < 2 {
                continue;
            }
            let raw = line[idx + 1..].trim();
            if raw.is_empty() {
                continue;
            }
            let is_manual = raw.to_lowercase().starts_with("manual-");
            if let Some(s) = normalize_status(raw) {
                return Some((s, is_manual));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stem() {
        // The path must have the batch folder (POC_P2_20000_XX) as ancestor
        // so batch_re can extract the batch number.
        let stem = Path::new(
            "C:/POC_P2_20000_16/padded_china_junior-high-school_grade-1_economics_history_202508041807235332_train_para1.json",
        );
        let meta = parse_stem(stem);
        assert!(meta.is_some(), "parse_stem should succeed on a valid path");
        let m = meta.unwrap();
        assert_eq!(m.batch, 16, "batch extracted from folder name");
        assert_eq!(m.grade, "grade-1");
        assert_eq!(m.subject, "economics");
        assert_eq!(m.doc_type, "history");
    }

    #[test]
    fn test_parse_stem_unknown_batch() {
        // No batch folder → batch defaults to -1
        let stem = Path::new(
            "somewhere/padded_china_junior-high-school_grade-1_economics_history_202508041807235332_train_para1.json",
        );
        let meta = parse_stem(stem);
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().batch, -1);
    }

    #[test]
    fn test_normalize_status() {
        assert_eq!(normalize_status("accept"), Some(Label::Accept));
        assert_eq!(
            normalize_status("reject-whitespace"),
            Some(Label::Reject(RejectReason::Whitespace))
        );
        assert_eq!(
            normalize_status("manual-reject-rotation"),
            Some(Label::Reject(RejectReason::Rotation))
        );
        assert_eq!(
            normalize_status("manual-reject-cutoff"),
            Some(Label::Reject(RejectReason::CutOff))
        );
        assert_eq!(
            normalize_status("manual-reject-unknown"),
            Some(Label::Reject(RejectReason::Manual))
        );
        assert_eq!(normalize_status("invalid"), None);
    }

    #[test]
    fn test_parse_txt() {
        use std::fs;
        

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_review_log.txt");
        let content = "01/01/2024 10:00:00-accept\n02/01/2024 11:00:00-reject-whitespace";
        fs::write(&test_file, content).unwrap();

        let result = parse_txt(&test_file);
        assert!(result.is_some());
        let (label, is_manual) = result.unwrap();
        assert_eq!(label, Label::Reject(RejectReason::Whitespace));
        assert!(!is_manual);

        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_parse_txt_manual_override() {
        use std::fs;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_manual_override.txt");
        let content = "01/01/2024 10:00:00-reject\n02/01/2024 11:00:00-manual-accept";
        fs::write(&test_file, content).unwrap();

        let result = parse_txt(&test_file);
        assert!(result.is_some());
        let (label, is_manual) = result.unwrap();
        assert_eq!(label, Label::Accept);
        assert!(is_manual);

        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_parse_txt_date_prefix_guard() {
        use std::fs;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_date_prefix.txt");
        // "2024-01-01 some comment" has only 2 hyphens — should be skipped.
        // The valid line "01/01/2024 10:00:00-accept" has 6 hyphens and should win.
        let content = "2024-01-01 some comment\n01/01/2024 10:00:00-accept\n";
        fs::write(&test_file, content).unwrap();

        let result = parse_txt(&test_file);
        assert!(result.is_some());
        let (label, _) = result.unwrap();
        assert_eq!(label, Label::Accept);

        fs::remove_file(test_file).ok();
    }
}