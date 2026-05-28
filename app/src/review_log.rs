//! Append-only writer for review-log `.txt` files.
//!
//! Format matches the existing logs: `DD/MM/YYYY HH:MM:SS-<status>` per line.
//! Status tokens used by the editor: `manual-accept`,
//! `manual-reject-{whitespace,rotation,cutoff,structure,generic}`.

use std::io::Write;
use std::path::Path;

/// Append a single status line to `txt_path`, creating the file if missing.
pub fn append_status(txt_path: &Path, status: &str) -> std::io::Result<()> {
    let now = chrono::Local::now().format("%d/%m/%Y %H:%M:%S");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(txt_path)?;
    writeln!(file, "{}-{}", now, status)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::parse_txt;
    use std::fs;

    #[test]
    fn test_append_status_creates_file() {
        let temp = std::env::temp_dir().join("bap_test_review_log_create");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let txt_path = temp.join("review.txt");

        assert!(!txt_path.exists());
        append_status(&txt_path, "accept").unwrap();
        assert!(txt_path.exists());

        let content = fs::read_to_string(&txt_path).unwrap();
        assert!(content.contains("-accept"));
        // Verify timestamp prefix format: DD/MM/YYYY HH:MM:SS-
        assert!(content.contains("/"));
        assert!(content.contains(":"));

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_append_status_roundtrip_with_parse_txt() {
        let temp = std::env::temp_dir().join("bap_test_review_log_roundtrip");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let txt_path = temp.join("review.txt");

        // Write two status lines
        append_status(&txt_path, "reject-whitespace").unwrap();
        append_status(&txt_path, "manual-accept").unwrap();

        // Parse back — should return the LAST status
        let (label, is_manual) = parse_txt(&txt_path).unwrap();
        assert_eq!(label, crate::types::Label::Accept);
        assert!(is_manual);

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_append_status_manual_reject_roundtrip() {
        let temp = std::env::temp_dir().join("bap_test_review_log_manual_rej");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let txt_path = temp.join("review.txt");

        append_status(&txt_path, "manual-reject-rotation").unwrap();

        let (label, is_manual) = parse_txt(&txt_path).unwrap();
        assert_eq!(label, crate::types::Label::Reject(crate::types::RejectReason::Rotation));
        assert!(is_manual);

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_append_status_multiple_lines_last_wins() {
        let temp = std::env::temp_dir().join("bap_test_review_log_multi");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let txt_path = temp.join("review.txt");

        append_status(&txt_path, "accept").unwrap();
        append_status(&txt_path, "reject").unwrap();
        append_status(&txt_path, "accept").unwrap();

        let (label, is_manual) = parse_txt(&txt_path).unwrap();
        assert_eq!(label, crate::types::Label::Accept);
        assert!(!is_manual);

        std::fs::remove_dir_all(&temp).ok();
    }
}