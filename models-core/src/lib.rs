//! LLM Core - Domain types and abstractions
//!
//! This crate provides the core domain types for Models Lab, including
//! models, experiments, datasets, and inference abstractions.

pub mod config;
pub mod domain;
pub mod error;
pub mod evaluation;
pub mod inference;
pub mod providers;
pub mod storage;

pub use config::Config;
pub use domain::*;
pub use error::{Error, Result};
pub use evaluation::{
    Benchmark, BenchmarkCategory, BenchmarkRegistry, BenchmarkResultSummary,
    BenchmarkRunConfig, BenchmarkRunResult, BenchmarkRunStatistics, CategoryScore,
    ComparisonReport, CostAnalysis, DetailedMetrics, EvaluationConfig,
    EvaluationOrchestrator, EvaluationReport, ModelComparison,
};
pub use inference::{InferenceProvider, InferenceRequest, InferenceResponse};
pub use providers::{
    ApiFormat, ChunkStream, EmbedRequest, EmbedResponse, FinishReason, GenerateRequest,
    GenerateResponse, HealthStatus, ModelInfo, ModelProvider, ProviderConfig, ProviderFactory,
    ProviderType, StreamChunk, TokenUsage,
};
pub use storage::{
    EvaluationStorage, InMemoryStorage, QueryParams, StoredBenchmarkResult, StoredEvaluationReport,
};