use crate::app::App;
use eframe::egui;

pub fn show(app: &mut App, ctx: &egui::Context) {
    if !app.analysis.result.is_ready() {
        return;
    }

    egui::TopBottomPanel::bottom("predictions-table")
        .resizable(true)
        .default_height(280.0)
        .min_height(140.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("Predictions");
                ui.separator();
                ui.colored_label(
                    super::C_MUTED,
                    "Drag the top edge to resize. Click a column header to sort.",
                );
            });
            ui.add_space(4.0);

            let filter_subj_lc = app.analysis.filter_subject.to_lowercase();
            let tier_filter = app.analysis.filter_tier;
            let cutoffs = app.analysis.cutoffs();
            let s_col = app.analysis.sort_col;
            let s_asc = app.analysis.sort_asc;

            // Borrow app.analysis.result immutably for filter + table render, drop the
            // borrow before applying any sort/double-click side effects.
            let action = {
                let predictions = &app.analysis.result.as_ref().expect("result present").predictions;
                let indices: Vec<usize> = predictions
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        super::passes_filter(p, tier_filter, &filter_subj_lc, cutoffs)
                    })
                    .map(|(i, _)| i)
                    .collect();

                ui.label(format!(
                    "Showing {} of {} files",
                    indices.len(),
                    predictions.len()
                ));
                ui.add_space(2.0);

                super::table::show(ui, predictions, &indices, cutoffs, s_col, s_asc)
            };

            if let Some((c, a)) = action.sort_change {
                app.analysis.sort_col = c;
                app.analysis.sort_asc = a;
            }
            if let Some(idx) = action.open_editor_idx {
                app.open_editor(idx);
            }
        });
}