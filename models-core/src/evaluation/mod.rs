//! Evaluation module

pub mod comparison;
pub mod orchestrator;
pub mod reports;

pub use comparison::{
    BenchmarkWinner, ComparisonConfig, ComparisonReport, ComparisonReportBuilder,
    ModelBenchmarkResult, ModelCategoryScore, ModelComparisonResult, ModelConfig,
    StatisticalTest, WinnerInfo,
};
pub use orchestrator::{
    Benchmark, BenchmarkCategory, BenchmarkRegistry, BenchmarkResultSummary,
    BenchmarkRunConfig, BenchmarkRunResult, BenchmarkRunStatistics, CategoryScore,
    ComparisonReport as OrchestratorComparisonReport, CostAnalysis, DetailedMetrics,
    EvaluationConfig, EvaluationOrchestrator, EvaluationReport, ModelComparison,
};
pub use reports::{
    generate_comparison_html_report, generate_html_report, generate_html_report_with_config,
    generate_html_from_stored, HtmlReportConfig, HtmlReportGenerator,
    save_comparison_html_report, save_html_report,
};
