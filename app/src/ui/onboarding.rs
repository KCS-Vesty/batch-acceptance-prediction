//! Onboarding / empty-state UI shown before any analysis has been run.
//!
//! Extracted from `ui/central.rs` to keep the dashboard module focused on
//! analytics rendering.  The two-step flow is:
//!
//! 1. Pick a parent folder containing `POC_P2_*` batch subfolders.
//! 2. Run the analysis (triggered from the side panel).

use crate::app::App;
use crate::ui::C_GREEN;
use crate::ui::C_MUTED;
use eframe::egui;

/// Full-width empty state: title, subtitle, and the two onboarding steps.
pub fn show_empty(ui: &mut egui::Ui, app: &mut App) {
    ui.vertical_centered(|ui| {
        ui.add_space(60.0);
        ui.label(
            egui::RichText::new("Batch Acceptance Prediction")
                .size(28.0)
                .strong(),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Score annotation quality across POC_P2_* batch folders")
                .size(14.0)
                .color(C_MUTED),
        );
        ui.add_space(28.0);

        // Step 1 — pick folder
        let folder_ok = app.folder.is_some();
        step_card(ui, 1, "Pick a parent folder", folder_ok, |ui| {
            ui.label(
                egui::RichText::new("Must contain POC_P2_* batch subfolders.")
                    .color(C_MUTED),
            );
            ui.add_space(6.0);
            if let Some(folder) = &app.folder {
                ui.horizontal(|ui| {
                    ui.colored_label(C_GREEN, "● Selected:");
                    ui.monospace(folder.display().to_string());
                });
                ui.add_space(4.0);
                if ui.button("Switch folder...").clicked() {
                    app.pick_folder();
                }
            } else if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("Choose folder...").strong().size(14.0),
                    )
                    .min_size(egui::vec2(0.0, 32.0)),
                )
                .clicked()
            {
                app.pick_folder();
            }
        });

        ui.add_space(12.0);

        // Step 2 — run analysis
        step_card(ui, 2, "Run analysis", false, |ui| {
            ui.label(
                egui::RichText::new(if folder_ok {
                    "Use \"▶  Run analysis\" on the left to train the model and score every unprocessed .json."
                } else {
                    "Pick a folder first."
                })
                .color(C_MUTED),
            );
        });
    });
}

/// Renders a numbered step card with a green checkmark when complete.
///
/// The `body` closure paints the card's interior content.
pub fn step_card<R>(
    ui: &mut egui::Ui,
    step_n: u32,
    title: &str,
    complete: bool,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let bullet = if complete {
        egui::RichText::new(format!("✓  Step {}", step_n))
            .color(C_GREEN)
            .strong()
    } else {
        egui::RichText::new(format!("{}.  Step {}", step_n, step_n))
            .color(C_MUTED)
            .strong()
    };
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::same(14.0))
        .show(ui, |ui| {
            ui.set_min_width(420.0);
            ui.set_max_width(560.0);
            ui.horizontal(|ui| {
                ui.label(bullet);
                ui.separator();
                ui.label(egui::RichText::new(title).strong().size(16.0));
            });
            ui.add_space(6.0);
            body(ui)
        })
        .inner
}
