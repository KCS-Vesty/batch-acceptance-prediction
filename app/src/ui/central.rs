use super::{charts, format_int, log_view, onboarding, C_BLUE, C_GREEN, C_INDIGO, C_MUTED, C_RED};
use crate::app::{App, State};
use crate::types::{PipelineResult, Prediction, RiskTier};
use crate::tier::{classify, Cutoffs};
use eframe::egui;

pub fn show(app: &mut App, ctx: &egui::Context) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let running = app.state == State::Running;
        if app.analysis.result.is_ready() {
            show_analytics(ui, app);
        } else if running || !app.log.is_empty() {
            ui.heading("Analysis log");
            log_view::show(ui, &app.log);
        } else {
            onboarding::show_empty(ui, app);
        }
    });
}

fn show_analytics(ui: &mut egui::Ui, app: &mut App) {
    // Snapshot filter state into local copies so the closure below doesn't need
    // a mutable borrow of `app` (which would conflict with the immutable borrow
    // via `app.analysis.result`). Any user-driven changes are written back at the end.
    let mut local_tier = app.analysis.filter_tier;
    let mut local_subject = app.analysis.filter_subject.clone();
    let cutoffs = app.analysis.cutoffs();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let filter_subj_lc = local_subject.to_lowercase();
            let result_ref = app.analysis.result.as_ref().expect("result checked by is_ready");
            let filtered: Vec<&Prediction> = result_ref
                .predictions
                .iter()
                .filter(|p| super::passes_filter(p, local_tier, &filter_subj_lc, cutoffs))
                .collect();

            kpi_strip(ui, result_ref, &filtered, cutoffs);

            ui.add_space(8.0);
            model_chips(ui, result_ref);

            ui.add_space(8.0);
            filters_row(ui, &mut local_tier, &mut local_subject);
            ui.add_space(charts::ROW_GAP);

            ui.columns(2, |cols| {
                charts::risk_distribution(&mut cols[0], &filtered, cutoffs);
                charts::accept_vs_reject(&mut cols[1], &filtered);
            });
            ui.add_space(charts::ROW_GAP);
            ui.columns(2, |cols| {
                charts::folder_rates(&mut cols[0], &filtered, cutoffs);
                charts::batch_rates(&mut cols[1], &filtered, cutoffs);
            });

            ui.add_space(10.0);
            ui.collapsing("Analysis log", |ui| log_view::show(ui, &app.log));
        });

    if app.analysis.filter_tier != local_tier {
        app.analysis.filter_tier = local_tier;
    }
    if app.analysis.filter_subject != local_subject {
        app.analysis.filter_subject = local_subject;
    }
}

/// Compact, scannable model summary in coloured chips instead of a paragraph.
fn model_chips(ui: &mut egui::Ui, result: &PipelineResult) {
    ui.horizontal_wrapped(|ui| {
        chip(ui, "Model", "Logistic regression · 14 features", C_INDIGO);
        chip(ui, "Accuracy", &format!("{:.3}", result.validation.accuracy), score_color(result.validation.accuracy));
        chip(ui, "AUC", &format!("{:.3}", result.validation.auc), score_color(result.validation.auc));
        chip(
            ui,
            "Labeled",
            &format!("{} accept / {} reject", format_int(result.n_accept), format_int(result.n_reject)),
            C_MUTED,
        );
        if let Some(sug) = result.validation.suggested_high_cutoff {
            chip(
                ui,
                "Suggested High cutoff",
                &format!("{:.3} (90% reject recall)", sug),
                C_BLUE,
            );
        }
    });

    // Per-reason recall — one chip per reason that actually appears.
    if !result.validation.recall_by_reason.is_empty() {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.small(egui::RichText::new("Reject recall by reason:").color(C_MUTED));
            let mut sorted: Vec<_> = result.validation.recall_by_reason.iter().collect();
            sorted.sort_by_key(|(reason, _)| reason.display_order());
            for (reason, recall) in sorted {
                chip(
                    ui,
                    &reason.to_string(),
                    &format!("{:.0}%", recall * 100.0),
                    recall_color(*recall),
                );
            }
        });
    }

    // Top-5 features as compact mono-spaced chips.
    if !result.validation.feature_weights.is_empty() {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.small(egui::RichText::new("Top features:").color(C_MUTED));
            for (name, w) in result.validation.feature_weights.iter().take(5) {
                let color = if *w >= 0.0 { C_GREEN } else { C_RED };
                chip(ui, name, &format!("{:+.3}", w), color);
            }
        });
    }

    // Per-run counts in small grey text — context, not headline.
    ui.add_space(2.0);
    ui.small(
        egui::RichText::new(format!(
            "Scored {} / {} unprocessed · Trained on {} labeled · {} ambiguous skipped",
            format_int(result.lr_scored),
            format_int(result.n_unprocessed),
            format_int(result.lr_trained_on),
            format_int(result.n_ambiguous),
        ))
        .color(C_MUTED),
    );
}

fn score_color(v: f64) -> egui::Color32 {
    if v >= 0.9 {
        C_GREEN
    } else if v >= 0.75 {
        super::C_AMBER
    } else {
        C_RED
    }
}

fn recall_color(v: f64) -> egui::Color32 {
    score_color(v)
}

/// Subtle border-only chip with a coloured value. Keeps the dashboard scannable.
fn chip(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    egui::Frame::none()
        .stroke(egui::Stroke::new(1.0, color.gamma_multiply(0.7)))
        .rounding(egui::Rounding::same(4.0))
        .inner_margin(egui::Margin::symmetric(8.0, 3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(egui::RichText::new(label).color(C_MUTED));
                ui.label(
                    egui::RichText::new(value)
                        .color(color)
                        .monospace()
                        .strong(),
                );
            });
        });
}

fn kpi_strip(
    ui: &mut egui::Ui,
    result: &PipelineResult,
    filtered: &[&Prediction],
    cutoffs: Cutoffs,
) {
    let total = filtered.len();
    let exp_accept: f64 = filtered.iter().map(|p| p.p_combined).sum();
    let accept_pct = if total > 0 {
        exp_accept / total as f64 * 100.0
    } else {
        0.0
    };
    let n_high = filtered
        .iter()
        .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::High)
        .count();
    let n_med = filtered
        .iter()
        .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::Medium)
        .count();
    let n_low = filtered
        .iter()
        .filter(|p| classify(p.p_combined, cutoffs) == RiskTier::Low)
        .count();
    let labeled_rate = if result.n_labeled > 0 {
        result.n_accept as f64 / result.n_labeled as f64 * 100.0
    } else {
        0.0
    };

    ui.columns(6, |cols| {
        kpi_card(&mut cols[0], "Files in view", &format_int(total), C_BLUE, None);
        kpi_card(
            &mut cols[1],
            "Expected accept",
            &format!("{:.0}", exp_accept),
            C_GREEN,
            Some(format!("{:.1}% of view", accept_pct)),
        );
        kpi_card(
            &mut cols[2],
            "High risk",
            &format_int(n_high),
            C_RED,
            Some(format!("p_comb < {:.3}", cutoffs.high_cutoff)),
        );
        kpi_card(
            &mut cols[3],
            "Medium risk",
            &format_int(n_med),
            super::C_AMBER,
            Some(format!("{:.3}–{:.3}", cutoffs.high_cutoff, cutoffs.low_cutoff)),
        );
        kpi_card(
            &mut cols[4],
            "Low risk",
            &format_int(n_low),
            C_GREEN,
            Some(format!("≥ {:.3}", cutoffs.low_cutoff)),
        );
        kpi_card(
            &mut cols[5],
            "Labeled baseline",
            &format!("{:.1}%", labeled_rate),
            C_INDIGO,
            Some(format!("{} labeled files", format_int(result.n_labeled))),
        );
    });
}

fn kpi_card(
    ui: &mut egui::Ui,
    label: &str,
    value: &str,
    color: egui::Color32,
    note: Option<String>,
) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(10.0, 8.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new(label).size(11.0).color(C_MUTED));
                ui.label(
                    egui::RichText::new(value)
                        .size(22.0)
                        .strong()
                        .color(color),
                );
                if let Some(n) = note {
                    ui.label(egui::RichText::new(n).size(11.0).color(C_MUTED));
                }
            });
        });
}

fn filters_row(
    ui: &mut egui::Ui,
    tier: &mut Option<RiskTier>,
    subject: &mut String,
) {
    ui.horizontal(|ui| {
        ui.label("Filter tier:");
        egui::ComboBox::from_id_source("tier-filter")
            .selected_text(match *tier {
                None => "All",
                Some(RiskTier::High) => "High",
                Some(RiskTier::Medium) => "Medium",
                Some(RiskTier::Low) => "Low",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(tier, None, "All");
                ui.selectable_value(tier, Some(RiskTier::High), "High");
                ui.selectable_value(tier, Some(RiskTier::Medium), "Medium");
                ui.selectable_value(tier, Some(RiskTier::Low), "Low");
            });
        ui.separator();
        ui.label("Subject contains:");
        ui.add(egui::TextEdit::singleline(subject).hint_text("e.g. math"));
        if ui.button("Clear filters").clicked() {
            subject.clear();
            *tier = None;
        }
    });
}
