//! Batch Acceptance Prediction — library crate.
//!
//! All domain logic, pipeline orchestration, and UI modules live here.
//! `main.rs` is a thin entry point that just launches the egui app.

pub mod app;
pub mod pipeline;
pub mod extract;
pub mod review_log;
pub mod tier;
pub mod ui;
pub mod annotation;
pub mod image_features;
pub mod img_feat_cache;
pub mod parsing;
pub mod features;
pub mod model;
pub mod types;
pub mod validation;

#[cfg(test)]
mod integration_tests;
