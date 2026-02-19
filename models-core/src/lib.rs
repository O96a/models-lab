//! LLM Core - Domain types and abstractions
//!
//! This crate provides the core domain types for Models Lab, including
//! models, experiments, datasets, and inference abstractions.

pub mod config;
pub mod domain;
pub mod error;
pub mod inference;

pub use config::Config;
pub use domain::*;
pub use error::{Error, Result};
pub use inference::{InferenceProvider, InferenceRequest, InferenceResponse};