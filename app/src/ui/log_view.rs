use eframe::egui;

pub fn show(ui: &mut egui::Ui, log: &[String]) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for line in log {
                ui.monospace(line);
            }
        });
}
