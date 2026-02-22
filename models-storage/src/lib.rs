//! Models Storage - PostgreSQL implementation for evaluation persistence
//!
//! This crate provides storage backends for persisting evaluation results,
//! benchmark data, and model configurations.

pub mod postgres;

pub use postgres::PostgresStorage;
pub use models_core::storage::{EvaluationStorage, InMemoryStorage, StoredBenchmarkResult, StoredEvaluationReport, QueryParams};