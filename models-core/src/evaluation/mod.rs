//! Evaluation module

pub mod orchestrator;

pub use orchestrator::{
    Benchmark, BenchmarkCategory, BenchmarkRegistry, BenchmarkResultSummary,
    BenchmarkRunConfig, BenchmarkRunResult, BenchmarkRunStatistics, CategoryScore,
    ComparisonReport, CostAnalysis, DetailedMetrics, EvaluationConfig,
    EvaluationOrchestrator, EvaluationReport, ModelComparison,
};
