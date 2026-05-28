use super::{C_AMBER, C_GREEN, C_MUTED, C_RED};
use crate::types::{Prediction, RiskTier};
use crate::tier::{classify, Cutoffs};
use eframe::egui;
use std::collections::HashMap;

const TITLE_SIZE: f32 = 13.0;
const ROW_HEIGHT: f32 = 20.0;
const MAX_ROWS_PER_CHART: usize = 12;
/// Minimum interior height for a chart card; keeps neighbouring cards in a
/// 2-column row visually aligned even when one is shorter than the other.
const CARD_MIN_HEIGHT: f32 = 280.0;
/// Vertical space between adjacent chart card rows.
pub const ROW_GAP: f32 = 12.0;

/// Stacked horizontal bar showing Low/Med/High distribution.
pub fn risk_distribution(ui: &mut egui::Ui, preds: &[&Prediction], cutoffs: Cutoffs) {
    chart_card(ui, "Risk tier distribution", |ui| {
        let n_low = preds
            .iter()
            .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::Low)
            .count();
        let n_med = preds
            .iter()
            .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::Medium)
            .count();
        let n_high = preds
            .iter()
            .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::High)
            .count();
        let total = preds.len().max(1) as f32;

        let bar_h = 32.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), bar_h),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        let mut x = rect.min.x;
        let segments: [(f32, egui::Color32, String); 3] = [
            (n_low as f32 / total, C_GREEN, format!("Low {}", n_low)),
            (n_med as f32 / total, C_AMBER, format!("Med {}", n_med)),
            (n_high as f32 / total, C_RED, format!("High {}", n_high)),
        ];
        for (frac, color, label) in &segments {
            let sw = rect.width() * *frac;
            if sw <= 0.5 {
                continue;
            }
            let seg = egui::Rect::from_min_size(
                egui::pos2(x, rect.min.y),
                egui::vec2(sw, bar_h),
            );
            painter.rect_filled(seg, 4.0, *color);
            if sw > 50.0 {
                painter.text(
                    seg.center(),
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
            }
            x += sw;
        }

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            legend(ui, C_GREEN, &format!("Low: {} ({:.1}%)", n_low, frac_pct(n_low, total)));
            ui.add_space(12.0);
            legend(ui, C_AMBER, &format!("Medium: {} ({:.1}%)", n_med, frac_pct(n_med, total)));
            ui.add_space(12.0);
            legend(ui, C_RED, &format!("High: {} ({:.1}%)", n_high, frac_pct(n_high, total)));
        });
    });
}

/// Side-by-side comparison of predicted accept vs predicted reject counts.
pub fn accept_vs_reject(ui: &mut egui::Ui, preds: &[&Prediction]) {
    chart_card(ui, "Predicted accept vs reject", |ui| {
        let total = preds.len();
        let exp_accept: f64 = preds.iter().map(|p| p.p_combined).sum();
        let exp_reject = (total as f64 - exp_accept).max(0.0);
        let max = exp_accept.max(exp_reject).max(1.0) as f32;

        for (label, value, color) in [
            ("Accept", exp_accept, C_GREEN),
            ("Reject", exp_reject, C_RED),
        ] {
            ui.horizontal(|ui| {
                ui.allocate_ui(egui::vec2(70.0, 28.0), |ui| {
                    ui.label(egui::RichText::new(label).strong());
                });
                let avail = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(avail, 28.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                let bar_w = rect.width() * (value as f32 / max).clamp(0.0, 1.0);
                let bar = egui::Rect::from_min_size(rect.min, egui::vec2(bar_w, 22.0));
                painter.rect_filled(bar, 4.0, color);
                let pct = if total > 0 { value / total as f64 * 100.0 } else { 0.0 };
                painter.text(
                    egui::pos2(rect.min.x + bar_w + 6.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    format!("{:.0}  ({:.1}%)", value, pct),
                    egui::FontId::proportional(12.0),
                    ui.style().visuals.text_color(),
                );
            });
        }
    });
}

/// Per-folder predicted acceptance rate, sorted worst-first. Folder = the
/// immediate parent directory of the `.json`, which corresponds to a batch
/// drop (e.g. `POC_P2_20000_16`). This is the right grouping for spotting
/// "this folder has a lot of bad annotations and should be re-shipped."
pub fn folder_rates(ui: &mut egui::Ui, preds: &[&Prediction], cutoffs: Cutoffs) {
    let items = aggregate_by_folder(preds);
    rate_chart(ui, "Predicted acceptance rate by folder", &items, cutoffs);
}

/// Per-batch predicted acceptance rate, sorted worst-first.
pub fn batch_rates(ui: &mut egui::Ui, preds: &[&Prediction], cutoffs: Cutoffs) {
    let items = aggregate_by_batch(preds);
    rate_chart(ui, "Predicted acceptance rate by batch", &items, cutoffs);
}

// ── aggregators ────────────────────────────────────────────────────────────

fn aggregate_by_folder(preds: &[&Prediction]) -> Vec<(String, f64, usize)> {
    let mut groups: HashMap<String, (f64, usize)> = HashMap::new();
    for p in preds {
        let folder = p
            .json_path
            .parent()
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "(unknown)".to_string());
        let e = groups.entry(folder).or_insert((0.0, 0));
        e.0 += p.p_combined;
        e.1 += 1;
    }
    finalize(groups.into_iter())
}

fn aggregate_by_batch(preds: &[&Prediction]) -> Vec<(String, f64, usize)> {
    let mut groups: HashMap<i32, (f64, usize)> = HashMap::new();
    for p in preds {
        let e = groups.entry(p.batch).or_insert((0.0, 0));
        e.0 += p.p_combined;
        e.1 += 1;
    }
    finalize(
        groups
            .into_iter()
            .map(|(k, v)| (format!("Batch {:02}", k), v)),
    )
}

fn finalize<I>(iter: I) -> Vec<(String, f64, usize)>
where
    I: Iterator<Item = (String, (f64, usize))>,
{
    let mut items: Vec<(String, f64, usize)> = iter
        .map(|(k, (sum, n))| (k, sum / n as f64 * 100.0, n))
        .collect();
    items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    items
}

// ── rendering ──────────────────────────────────────────────────────────────

fn rate_chart(
    ui: &mut egui::Ui,
    title: &str,
    items: &[(String, f64, usize)],
    cutoffs: Cutoffs,
) {
    chart_card(ui, title, |ui| {
        if items.is_empty() {
            ui.colored_label(C_MUTED, "(no data)");
            return;
        }
        let label_w = 110.0_f32;
        let value_w = 80.0_f32;

        for (name, rate, n) in items.iter().take(MAX_ROWS_PER_CHART) {
            ui.horizontal(|ui| {
                // `add_sized` strictly clamps the widget's size, so long
                // folder/batch names truncate cleanly instead of bleeding
                // into the bar column.
                ui.add_sized(
                    [label_w, ROW_HEIGHT],
                    egui::Label::new(name.as_str()).truncate(),
                );
                let bar_w_avail = (ui.available_width() - value_w).max(40.0);
                let (rect, _) = ui.allocate_exact_size(
                    egui::vec2(bar_w_avail, ROW_HEIGHT),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                // Centre the bar vertically inside the row so it visually
                // aligns with the labels on either side.
                let bar_h = ROW_HEIGHT - 8.0;
                let bg = egui::Rect::from_min_size(
                    egui::pos2(rect.min.x, rect.min.y + (ROW_HEIGHT - bar_h) * 0.5),
                    egui::vec2(rect.width(), bar_h),
                );
                painter.rect_filled(bg, 3.0, ui.style().visuals.faint_bg_color);
                let pct = (rate.clamp(0.0, 100.0) / 100.0) as f32;
                let fg = egui::Rect::from_min_size(
                    bg.min,
                    egui::vec2(bg.width() * pct, bg.height()),
                );
                painter.rect_filled(fg, 3.0, rate_color(*rate, cutoffs));
                ui.add_sized(
                    [value_w, ROW_HEIGHT],
                    egui::Label::new(
                        egui::RichText::new(format!("{:.1}%  n={}", rate, n))
                            .color(rate_color(*rate, cutoffs)),
                    ),
                );
            });
        }
        if items.len() > MAX_ROWS_PER_CHART {
            ui.add_space(4.0);
            ui.colored_label(
                C_MUTED,
                format!("(+{} more not shown)", items.len() - MAX_ROWS_PER_CHART),
            );
        }
    });
}

fn rate_color(rate_pct: f64, cutoffs: Cutoffs) -> egui::Color32 {
    match classify(rate_pct / 100.0, cutoffs) {
        RiskTier::Low => C_GREEN,
        RiskTier::Medium => C_AMBER,
        RiskTier::High => C_RED,
    }
}

fn legend(ui: &mut egui::Ui, color: egui::Color32, text: &str) {
    let dot = 10.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(dot, dot), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), dot / 2.0, color);
    ui.label(text);
}

fn frac_pct(n: usize, total: f32) -> f64 {
    if total <= 0.0 {
        0.0
    } else {
        n as f64 / total as f64 * 100.0
    }
}

fn chart_card<R>(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            // Reserve a stable minimum height so cards in the same row line
            // up regardless of how much content each one paints.
            ui.set_min_height(CARD_MIN_HEIGHT);
            ui.set_min_width(ui.available_width());
            ui.label(
                egui::RichText::new(title)
                    .size(TITLE_SIZE)
                    .strong()
                    .color(ui.style().visuals.strong_text_color()),
            );
            ui.add_space(6.0);
            body(ui)
        })
        .inner
}
