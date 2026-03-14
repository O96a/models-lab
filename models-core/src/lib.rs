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
    ComparisonReport, ComparisonReport as OrchestratorComparisonReport, CostAnalysis,
    DetailedMetrics, EvaluationConfig, EvaluationOrchestrator, EvaluationReport, ModelComparison,
    // Comparison module exports
    BenchmarkWinner, ComparisonConfig, ComparisonReportBuilder,
    ModelBenchmarkResult, ModelCategoryScore, ModelComparisonResult, ModelConfig,
    StatisticalTest, WinnerInfo,
    // HTML Report generation
    generate_comparison_html_report, generate_html_report, generate_html_report_with_config,
    generate_html_from_stored, HtmlReportConfig, HtmlReportGenerator,
    save_comparison_html_report, save_html_report,
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