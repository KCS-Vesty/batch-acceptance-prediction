#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;

fn main() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Batch Acceptance Prediction"),
        ..Default::default()
    };
    eframe::run_native(
        "Batch Acceptance Prediction",
        opts,
        Box::new(|cc| Ok(Box::new(batch_acceptance_app::app::App::new(cc)))),
    )
}
