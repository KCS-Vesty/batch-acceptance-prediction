use crate::app::{App, State};
use crate::tier::TierMethod;
use crate::types::RiskTier;
use eframe::egui;

// ---------------------------------------------------------------------------
// Extract panel UI state — checkboxes the user toggles before clicking Extract
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ExtractPanelState {
    pub include_high: bool,
    pub include_med: bool,
    pub include_low: bool,
    pub include_jpg: bool,
    pub include_txt: bool,
    pub move_files: bool,
    pub last_msg: Option<String>,
}

impl Default for ExtractPanelState {
    fn default() -> Self {
        Self {
            include_high: true,
            include_med: false,
            include_low: false,
            include_jpg: true,
            include_txt: true,
            move_files: false,
            last_msg: None,
        }
    }
}

impl ExtractPanelState {
    /// Build the set of tiers the user enabled. Returns `None` when nothing is
    /// selected (caller can show an inline error instead of a dialog).
    pub fn enabled_tiers(&self) -> Option<Vec<RiskTier>> {
        let mut tiers = Vec::new();
        if self.include_high {
            tiers.push(RiskTier::High);
        }
        if self.include_med {
            tiers.push(RiskTier::Medium);
        }
        if self.include_low {
            tiers.push(RiskTier::Low);
        }
        if tiers.is_empty() {
            None
        } else {
            Some(tiers)
        }
    }
}

// ---------------------------------------------------------------------------
// Correction panel UI state — simpler variant always targeting High risk
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CorrectionPanelState {
    pub include_med: bool,
    pub include_jpg: bool,
    pub include_txt: bool,
    pub move_files: bool,
    pub last_msg: Option<String>,
}

impl Default for CorrectionPanelState {
    fn default() -> Self {
        Self {
            include_med: false,
            include_jpg: true,
            include_txt: true,
            move_files: false,
            last_msg: None,
        }
    }
}

impl CorrectionPanelState {
    pub fn enabled_tiers(&self) -> Vec<RiskTier> {
        if self.include_med {
            vec![RiskTier::High, RiskTier::Medium]
        } else {
            vec![RiskTier::High]
        }
    }
}

// ---------------------------------------------------------------------------
// Side panel rendering
// ---------------------------------------------------------------------------

pub fn show(app: &mut App, ctx: &egui::Context) {
    let running = app.state == State::Running;
    let has_result = app.analysis.result.is_ready();

    egui::SidePanel::left("controls")
        .min_width(280.0)
        .default_width(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    analysis_section(ui, app, ctx, running);
                    ui.add_space(14.0);
                    ui.separator();
                    extract_section(ui, app, running, has_result);
                    ui.add_space(14.0);
                    ui.separator();
                    correction_section(ui, app, running, has_result);
                    ui.add_space(14.0);
                    ui.separator();
                    if has_result
                        && ui
                            .add(egui::Button::new("Clear results"))
                            .on_hover_text("Drop the current predictions and start a fresh run")
                            .clicked()
                    {
                        app.analysis.result.clear();
                        app.extract.last_msg = None;
                        app.correction.last_msg = None;
                    }
                });
        });
}

fn analysis_section(ui: &mut egui::Ui, app: &mut App, ctx: &egui::Context, running: bool) {
    ui.add_space(8.0);
    section_heading(ui, "Analysis");

    let analysis = &mut app.analysis;
    ui.add_enabled_ui(!running, |ui| {
        ui.add(egui::Slider::new(&mut analysis.cfg.alpha, 0.5..=10.0).text("Alpha"))
            .on_hover_text(
                "Laplace smoothing strength for the batch x subject prior.\n\
                 Higher = pulls extreme rates toward the global average.\n\
                 Default 3.0. Has no effect on the image features.",
            );
    });

    ui.add_space(8.0);
    sub_heading(ui, "Risk tier method");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut analysis.tier_cfg.method, TierMethod::Percentile, "Percentile")
            .on_hover_text("Bottom X% become High risk, top Y% become Low. Always splits.");
        ui.selectable_value(&mut analysis.tier_cfg.method, TierMethod::Absolute, "Absolute")
            .on_hover_text("Use absolute p_combined cutoffs. May leave a tier empty.");
    });
    match analysis.tier_cfg.method {
        TierMethod::Percentile => {
            ui.add(
                egui::Slider::new(&mut analysis.tier_cfg.pct_high, 0.0..=50.0)
                    .text("Bottom % = High"),
            );
            ui.add(
                egui::Slider::new(&mut analysis.tier_cfg.pct_low, 0.0..=90.0)
                    .text("Top % = Low"),
            );
            if analysis.tier_cfg.pct_high + analysis.tier_cfg.pct_low > 100.0 {
                ui.colored_label(
                    super::C_AMBER,
                    "High + Low exceed 100% — Medium band will be empty.",
                );
            }
        }
        TierMethod::Absolute => {
            ui.add(
                egui::Slider::new(&mut analysis.tier_cfg.abs_low, 0.5..=1.0)
                    .step_by(0.005)
                    .text("Low cutoff (p_comb ≥)"),
            );
            ui.add(
                egui::Slider::new(&mut analysis.tier_cfg.abs_high, 0.5..=1.0)
                    .step_by(0.005)
                    .text("High cutoff (p_comb <)"),
            );
            if analysis.tier_cfg.abs_low < analysis.tier_cfg.abs_high {
                ui.colored_label(
                    super::C_AMBER,
                    "Low cutoff is below High cutoff — Medium band is empty.",
                );
            }
            if let Some(sug) = analysis
                .result
                .as_ref()
                .and_then(|r| r.validation.suggested_high_cutoff)
            {
                ui.horizontal(|ui| {
                    ui.small(format!("Suggested {:.3} for 90% reject recall", sug));
                    if ui
                        .small_button("Apply")
                        .on_hover_text(
                            "Use this cutoff so ~90% of validation rejects land in the High tier",
                        )
                        .clicked()
                    {
                        analysis.tier_cfg.abs_high = sug;
                    }
                });
            }
        }
    }
    if let Some(result) = analysis.result.as_ref() {
        let c = analysis.tier_cfg.resolve(&result.sorted_p_combined);
        ui.small(format!(
            "Active cutoffs: High < {:.3}, Low ≥ {:.3}",
            c.high_cutoff, c.low_cutoff
        ));
    }

    ui.add_space(12.0);
    let can_run = app.folder.is_some() && !running;
    let run_btn = egui::Button::new(egui::RichText::new("▶  Run analysis").strong().size(15.0))
        .min_size(egui::vec2(0.0, 36.0))
        .fill(if can_run {
            super::C_BLUE
        } else {
            ui.style().visuals.widgets.inactive.bg_fill
        });
    if ui
        .add_enabled(can_run, run_btn)
        .on_hover_text(if app.folder.is_none() {
            "Pick a folder first"
        } else {
            "Score every .json in the folder against the model"
        })
        .clicked()
    {
        app.start_analysis(ctx);
    }
}

fn extract_section(ui: &mut egui::Ui, app: &mut App, running: bool, has_result: bool) {
    section_heading(ui, "Extract risky files");
    ui.small("Copy (default) or move flagged triplets to a folder.");
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut app.extract.include_high, tier_label("High"));
        ui.checkbox(&mut app.extract.include_med, tier_label("Medium"));
        ui.checkbox(&mut app.extract.include_low, tier_label("Low"));
    });
    ui.add_space(2.0);
    ui.checkbox(&mut app.extract.include_jpg, "Include .jpg sidecar");
    ui.checkbox(&mut app.extract.include_txt, "Include .txt sidecar");
    ui.checkbox(&mut app.extract.move_files, "Move (uncheck = copy)")
        .on_hover_text("Move deletes the source. Default is non-destructive copy.");
    ui.add_space(6.0);
    let can_extract = has_result && !running;
    if ui
        .add_enabled(
            can_extract,
            egui::Button::new("Pick destination & extract").min_size(egui::vec2(0.0, 30.0)),
        )
        .clicked()
    {
        app.do_extract();
    }
    if let Some(msg) = &app.extract.last_msg {
        ui.add_space(4.0);
        ui.colored_label(super::C_MUTED, msg);
    }
}

fn correction_section(ui: &mut egui::Ui, app: &mut App, running: bool, has_result: bool) {
    section_heading(ui, "Send for correction");
    ui.small("Export predicted rejects (High risk) for manual re-review.");
    ui.add_space(4.0);
    ui.checkbox(&mut app.correction.include_med, "Also include Medium risk");
    ui.checkbox(&mut app.correction.include_jpg, "Include .jpg sidecar");
    ui.checkbox(&mut app.correction.include_txt, "Include .txt sidecar");
    ui.checkbox(&mut app.correction.move_files, "Move (uncheck = copy)");
    ui.add_space(6.0);
    let can_extract = has_result && !running;
    if ui
        .add_enabled(
            can_extract,
            egui::Button::new("Pick folder & export for correction")
                .min_size(egui::vec2(0.0, 30.0)),
        )
        .clicked()
    {
        app.do_export_correction();
    }
    if let Some(msg) = &app.correction.last_msg {
        ui.add_space(4.0);
        ui.colored_label(super::C_MUTED, msg);
    }
}

// ── shared widgets ────────────────────────────────────────────────────────

fn section_heading(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).heading().strong());
}

fn sub_heading(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).strong());
}

fn tier_label(name: &str) -> egui::RichText {
    let color = match name {
        "High" => super::C_RED,
        "Medium" => super::C_AMBER,
        "Low" => super::C_GREEN,
        _ => super::C_MUTED,
    };
    egui::RichText::new(name).color(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enabled_tiers_all() {
        let state = ExtractPanelState::default();
        // default: include_high=true, med=false, low=false
        let tiers = state.enabled_tiers();
        assert!(tiers.is_some());
        assert_eq!(tiers.unwrap(), vec![RiskTier::High]);
    }

    #[test]
    fn test_enabled_tiers_none() {
        let state = ExtractPanelState {
            include_high: false,
            ..Default::default()
        };
        assert!(state.enabled_tiers().is_none());
    }

    #[test]
    fn test_correction_panel_tiers() {
        let state = CorrectionPanelState::default();
        assert_eq!(state.enabled_tiers(), vec![RiskTier::High]);

        let state_med = CorrectionPanelState {
            include_med: true,
            ..Default::default()
        };
        assert_eq!(
            state_med.enabled_tiers(),
            vec![RiskTier::High, RiskTier::Medium]
        );
    }
}
