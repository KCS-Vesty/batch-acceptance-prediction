//! Model submodules — logistic regression, k-fold CV, and Bayesian prior.
//!
//! Re-exports the public API so `crate::model::LogisticRegression` and
//! friends continue to resolve from the same path they always did.

pub mod logistic_regression;
pub mod kfold;
pub mod conditional_prob;

pub use logistic_regression::{LogisticRegression, L2_LAMBDA};
pub use kfold::run_lr_kfold_cv;
pub use conditional_prob::ConditionalProbModel;
