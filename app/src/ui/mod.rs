use eframe::egui;

pub mod central;
pub mod charts;
pub mod editor;
pub mod editor_window;
pub mod log_view;
pub mod onboarding;
pub mod side_panel;
pub mod table;
pub mod table_panel;
pub mod top_bar;

/// Single source of truth for "is this prediction visible under the current
/// tier + subject filters." Used by both the dashboard cards and the table.
pub fn passes_filter(
    p: &crate::types::Prediction,
    tier: Option<crate::types::RiskTier>,
    subject_lc: &str,
    cutoffs: crate::tier::Cutoffs,
) -> bool {
    if let Some(t) = tier {
        if crate::tier::classify(p.p_combined, cutoffs) != t {
            return false;
        }
    }
    if !subject_lc.is_empty() && !p.subject.to_lowercase().contains(subject_lc) {
        return false;
    }
    true
}

// ── shared palette ─────────────────────────────────────────────────────────
pub const C_GREEN: egui::Color32 = egui::Color32::from_rgb(0x16, 0xa3, 0x4a);
pub const C_AMBER: egui::Color32 = egui::Color32::from_rgb(0xd9, 0x77, 0x06);
pub const C_RED: egui::Color32 = egui::Color32::from_rgb(0xdc, 0x26, 0x26);
pub const C_BLUE: egui::Color32 = egui::Color32::from_rgb(0x25, 0x63, 0xeb);
pub const C_INDIGO: egui::Color32 = egui::Color32::from_rgb(0x4f, 0x46, 0xe5);
pub const C_MUTED: egui::Color32 = egui::Color32::from_rgb(0x6b, 0x72, 0x80);

pub fn tier_color(tier: crate::types::RiskTier) -> egui::Color32 {
    match tier {
        crate::types::RiskTier::Low => C_GREEN,
        crate::types::RiskTier::Medium => C_AMBER,
        crate::types::RiskTier::High => C_RED,
    }
}

pub fn tier_short(tier: crate::types::RiskTier) -> &'static str {
    match tier {
        crate::types::RiskTier::Low => "Low",
        crate::types::RiskTier::Medium => "Med",
        crate::types::RiskTier::High => "High",
    }
}

/// Full-word tier label for places with room (KPI strip, editor risk pill).
/// `tier_short` truncates Medium → "Med" for the table; this keeps it spelled out.
pub fn tier_long(tier: crate::types::RiskTier) -> &'static str {
    match tier {
        crate::types::RiskTier::Low => "Low",
        crate::types::RiskTier::Medium => "Medium",
        crate::types::RiskTier::High => "High",
    }
}

pub fn format_int(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}
