use crate::extract::{self, ExtractOptions};
use crate::ui::side_panel::{CorrectionPanelState, ExtractPanelState};
use crate::pipeline;
use crate::types::{PipelineConfig, PipelineResult, RiskTier, SortCol};
use crate::tier::{Cutoffs, TierConfig};
use crate::ui;
use crate::ui::editor::EditorState;
use eframe::egui;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;

// ---------------------------------------------------------------------------
// AnalysisState — shared state behind a narrow interface
// ---------------------------------------------------------------------------

/// Shared analysis state read by multiple UI panels (dashboard, table, side panel).
///
/// Bundles the pipeline result, tier configuration, and filter/sort settings
/// behind a single struct so panels can receive only this instead of `&mut App`.
#[derive(Default)]
pub struct AnalysisState {
    pub result: Sentinel<PipelineResult>,
    pub tier_cfg: TierConfig,
    pub cfg: PipelineConfig,
    pub filter_tier: Option<RiskTier>,
    pub filter_subject: String,
    pub sort_col: SortCol,
    pub sort_asc: bool,
}

impl AnalysisState {
    /// Resolves the current tier config against the result's sorted distribution.
    /// Returns absolute defaults if no result is loaded.
    pub fn cutoffs(&self) -> Cutoffs {
        match self.result.as_ref() {
            Some(r) => self.tier_cfg.resolve(&r.sorted_p_combined),
            None => Cutoffs::FALLBACK,
        }
    }
}

// ---------------------------------------------------------------------------
// Sentinel
// ---------------------------------------------------------------------------

/// Sentinel that replaces `Option<PipelineResult>` with a self-documenting type.
/// `as_ref()` returns `Some(&PipelineResult)` when data is available, `None` otherwise.
#[derive(Default)]
pub enum Sentinel<T> {
    #[default]
    Empty,
    Ready(T),
}

impl<T> Sentinel<T> {
    pub fn is_ready(&self) -> bool {
        matches!(self, Sentinel::Ready(_))
    }

    pub fn as_ref(&self) -> Option<&T> {
        match self {
            Sentinel::Ready(v) => Some(v),
            Sentinel::Empty => None,
        }
    }

    pub fn as_mut(&mut self) -> Option<&mut T> {
        match self {
            Sentinel::Ready(v) => Some(v),
            Sentinel::Empty => None,
        }
    }

    pub fn clear(&mut self) {
        *self = Sentinel::Empty;
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub enum Msg {
    Log(String),
    Done(Result<PipelineResult, String>),
}

#[derive(PartialEq, Eq)]
pub enum State {
    Idle,
    Running,
}

pub struct App {
    pub folder: Option<PathBuf>,
    pub state: State,
    pub log: Vec<String>,
    pub rx: Option<Receiver<Msg>>,

    /// Shared analysis state — passed to panels that need result data,
    /// filter/sort settings, or tier configuration.
    pub analysis: AnalysisState,

    /// Extract panel UI state (tier checkboxes, JPG/TXT toggles, move flag).
    pub extract: ExtractPanelState,

    /// Correction panel UI state (subset — always targets High risk).
    pub correction: CorrectionPanelState,

    /// Annotation editor window state.
    pub editor_state: EditorState,
}

impl Default for App {
    fn default() -> Self {
        Self {
            folder: None,
            state: State::Idle,
            log: Vec::new(),
            rx: None,
            analysis: AnalysisState::default(),
            extract: ExtractPanelState::default(),
            correction: CorrectionPanelState::default(),
            editor_state: EditorState::default(),
        }
    }
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    pub fn cutoffs(&self) -> Cutoffs {
        self.analysis.cutoffs()
    }

    pub fn pick_folder(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .set_title("Pick dataset root folder")
            .pick_folder()
        {
            self.folder = Some(p);
            self.analysis.result.clear();
            self.log.clear();
            self.extract.last_msg = None;
            self.correction.last_msg = None;
        }
    }

    pub fn start_analysis(&mut self, ctx: &egui::Context) {
        let folder = match self.folder.clone() {
            Some(f) => f,
            None => return,
        };
        let cfg = self.analysis.cfg;
        let (tx, rx) = channel::<Msg>();
        self.rx = Some(rx);
        self.state = State::Running;
        self.log.clear();
        self.analysis.result.clear();
        self.extract.last_msg = None;
        self.correction.last_msg = None;
        let ctx2 = ctx.clone();

        thread::spawn(move || {
            let tx_log = tx.clone();
            let ctx_log = ctx2.clone();
            let log_fn = move |s: &str| {
                let _ = tx_log.send(Msg::Log(s.to_string()));
                ctx_log.request_repaint();
            };
            let res = pipeline::run(&folder, &cfg, &log_fn).map_err(|e| e.to_string());
            let _ = tx.send(Msg::Done(res));
            ctx2.request_repaint();
        });
    }

    pub fn poll(&mut self) {
        let mut done = false;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Log(s) => self.log.push(s),
                    Msg::Done(Ok(r)) => {
                        self.log
                            .push(format!("Analysis complete: {} predictions", r.predictions.len()));
                        self.analysis.result = Sentinel::Ready(r);
                        self.state = State::Idle;
                        done = true;
                    }
                    Msg::Done(Err(e)) => {
                        self.log.push(format!("ERROR: {}", e));
                        self.state = State::Idle;
                        done = true;
                    }
                }
            }
        }
        if done {
            self.rx = None;
        }
    }

    /// Re-open the annotation editor for prediction row `idx`.
    /// Safe to call even if `result` is None or `idx` is out of range.
    pub fn open_editor(&mut self, idx: usize) {
        let json_path = self
            .analysis
            .result
            .as_ref()
            .and_then(|r| r.predictions.get(idx))
            .map(|p| p.json_path.clone());
        if let Some(jp) = json_path {
            self.editor_state.open_for(idx, &jp);
        }
    }

    pub fn do_export_correction(&mut self) {
        let tiers = self.correction.enabled_tiers();
        self.correction.last_msg = run_extract(
            &self.analysis.result,
            &self.analysis.tier_cfg,
            &RunExtractOpts {
                dialog_title: "Pick correction folder (predicted-reject files will be copied/moved here)",
                action_label: "Export",
                tiers,
                include_jpg: self.correction.include_jpg,
                include_txt: self.correction.include_txt,
                move_files: self.correction.move_files,
            },
        );
    }

    pub fn do_extract(&mut self) {
        let tiers = match self.extract.enabled_tiers() {
            Some(t) => t,
            None => {
                self.extract.last_msg = Some("Select at least one risk tier.".into());
                return;
            }
        };
        self.extract.last_msg = run_extract(
            &self.analysis.result,
            &self.analysis.tier_cfg,
            &RunExtractOpts {
                dialog_title: "Pick destination folder for extracted files",
                action_label: "Extract",
                tiers,
                include_jpg: self.extract.include_jpg,
                include_txt: self.extract.include_txt,
                move_files: self.extract.move_files,
            },
        );
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll();
        ui::top_bar::show(self, ctx);
        ui::table_panel::show(self, ctx);
        ui::side_panel::show(self, ctx);
        ui::central::show(self, ctx);
        ui::editor_window::show(self, ctx);
    }
}

// ---------------------------------------------------------------------------
// Shared extract runner (moved from App::run_extract)
// ---------------------------------------------------------------------------

/// Options for the extract operation passed from UI panels.
pub struct RunExtractOpts<'a> {
    pub dialog_title: &'a str,
    pub action_label: &'a str,
    pub tiers: Vec<RiskTier>,
    pub include_jpg: bool,
    pub include_txt: bool,
    pub move_files: bool,
}

/// Prompt for a destination, run the transfer, and return the user-visible message.
/// Returns `None` if the user dismissed the folder picker or if no result is loaded.
pub fn run_extract(
    result: &Sentinel<PipelineResult>,
    tier_cfg: &TierConfig,
    opts: &RunExtractOpts<'_>,
) -> Option<String> {
    let predictions = &result.as_ref()?.predictions;
    let dest = rfd::FileDialog::new()
        .set_title(opts.dialog_title)
        .pick_folder()?;
    let cutoffs = match result.as_ref() {
        Some(r) => tier_cfg.resolve(&r.sorted_p_combined),
        None => Cutoffs::FALLBACK,
    };
    let extract_opts = ExtractOptions {
        tiers: opts.tiers.clone(),
        include_jpg: opts.include_jpg,
        include_txt: opts.include_txt,
        move_files: opts.move_files,
        cutoffs,
    };
    Some(match extract::extract(predictions, &dest, &extract_opts) {
        Ok(r) => {
            let verb = if opts.move_files { "Moved" } else { "Copied" };
            format!(
                "{} {} file(s) to {} (errors: {}, skipped: {})",
                verb,
                r.transferred.len(),
                dest.display(),
                r.errors.len(),
                r.skipped
            )
        }
        Err(e) => format!("{} failed: {}", opts.action_label, e),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentinel_default_is_empty() {
        let s: Sentinel<i32> = Sentinel::default();
        assert!(!s.is_ready());
        assert!(s.as_ref().is_none());
    }

    #[test]
    fn test_sentinel_ready() {
        let mut s = Sentinel::Ready(42);
        assert!(s.is_ready());
        assert_eq!(s.as_ref(), Some(&42));
        assert_eq!(s.as_mut(), Some(&mut 42));
    }

    #[test]
    fn test_sentinel_clear() {
        let mut s = Sentinel::Ready(vec![1, 2, 3]);
        assert!(s.is_ready());
        s.clear();
        assert!(!s.is_ready());
        assert!(s.as_ref().is_none());
    }

    #[test]
    fn test_sentinel_as_mut_modifies() {
        let mut s = Sentinel::Ready(String::from("hello"));
        if let Some(v) = s.as_mut() {
            v.push_str(" world");
        }
        assert_eq!(s.as_ref().unwrap(), "hello world");
    }
}
