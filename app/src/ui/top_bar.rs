use crate::app::{App, State};
use eframe::egui;

pub fn show(app: &mut App, ctx: &egui::Context) {
    let running = app.state == State::Running;
    egui::TopBottomPanel::top("top-bar").show(ctx, |ui| {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Batch Acceptance Prediction");
            ui.separator();

            // Status pill — green dot when idle with a result, blue spinner
            // while running, grey when nothing has been done yet.
            match (running, app.analysis.result.is_ready()) {
                (true, _) => {
                    ui.spinner();
                    ui.colored_label(super::C_BLUE, "Running...");
                }
                (false, true) => {
                    ui.colored_label(super::C_GREEN, "● Ready");
                }
                (false, false) => {
                    ui.colored_label(super::C_MUTED, "○ Idle");
                }
            }
            ui.separator();

            ui.label("Folder:");
            // Show just the last path component to save horizontal space; the
            // full absolute path is one hover away. Mounted to the right of
            // the title so the file name is always visible.
            let (display, full) = match &app.folder {
                Some(p) => {
                    let full = p.display().to_string();
                    let short = p
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| full.clone());
                    (short, Some(full))
                }
                None => ("(none selected)".to_string(), None),
            };
            let resp = ui.monospace(display);
            if let Some(full) = full {
                resp.on_hover_text(full);
            }

            // Push the switch-folder button to the right edge.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!running, egui::Button::new("Switch folder..."))
                    .on_hover_text("Pick a parent directory that contains POC_P2_* subfolders")
                    .clicked()
                {
                    app.pick_folder();
                }
            });
        });
        ui.add_space(4.0);
    });
}
