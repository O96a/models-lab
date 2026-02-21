//! Evaluation orchestrator for running benchmarks and generating reports

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::providers::ModelProvider;
use crate::storage::{
    EvaluationStorage, StoredBenchmarkResult, StoredEvaluationReport, QueryParams,
};

/// Configuration for benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunConfig {
    /// Maximum samples to evaluate (None = all)
    pub max_samples: Option<usize>,
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Number of few-shot examples
    pub num_few_shot: usize,
    /// Categories to filter (None = all)
    pub categories: Option<Vec<String>>,
}

impl Default for BenchmarkRunConfig {
    fn default() -> Self {
        Self {
            max_samples: None,
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 5,
            categories: None,
        }
    }
}

/// Benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunStatistics {
    /// Total number of samples
    pub total_samples: usize,
    /// Number of correct answers
    pub correct_count: usize,
    /// Overall accuracy (0.0 to 1.0)
    pub accuracy: f64,
    /// Mean latency in milliseconds
    pub mean_latency_ms: f64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
}

/// Benchmark run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResult {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Model evaluated
    pub model_name: String,
    /// Provider used
    pub provider_name: String,
    /// Overall statistics
    pub statistics: BenchmarkRunStatistics,
    /// Per-category statistics
    pub category_stats: HashMap<String, BenchmarkRunStatistics>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Trait for benchmarks (defined here to avoid circular dependencies)
#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Get the benchmark ID
    fn id(&self) -> &str;

    /// Get the benchmark name
    fn name(&self) -> &str;

    /// Get the benchmark category
    fn category(&self) -> BenchmarkCategory;

    /// Run the benchmark with a provider
    async fn run(
        &self,
        provider: &dyn ModelProvider,
        config: BenchmarkRunConfig,
    ) -> Result<BenchmarkRunResult>;
}

/// Category of benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkCategory {
    Reasoning,
    Math,
    Coding,
    Hallucination,
    Context,
    Instruction,
    Multiturn,
    Safety,
    General,
}

/// Configuration for evaluation runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Number of few-shot examples
    pub num_few_shot: usize,
    /// Maximum samples per benchmark
    pub max_samples_per_benchmark: Option<usize>,
    /// Benchmarks to run
    pub benchmarks: Vec<String>,
    /// Save results to storage
    pub save_results: bool,
    /// Generate detailed report
    pub detailed_report: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 5,
            max_samples_per_benchmark: None,
            benchmarks: vec!["mmlu".to_string()],
            save_results: true,
            detailed_report: true,
        }
    }
}

/// Category score in a report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    /// Category name
    pub category: String,
    /// Score (0.0 to 1.0)
    pub score: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Benchmarks in this category
    pub benchmarks: Vec<String>,
}

/// Detailed metrics in a report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMetrics {
    /// Mean latency in milliseconds
    pub mean_latency_ms: f64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Tokens per second (if available)
    pub tokens_per_second: Option<f64>,
    /// Total errors
    pub error_count: usize,
}

/// Cost analysis in a report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    /// Estimated cost per 1K tokens
    pub cost_per_1k_tokens: Option<f64>,
    /// Total estimated cost
    pub total_cost: Option<f64>,
    /// Total tokens used
    pub total_tokens: u64,
}

/// Evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Report ID
    pub id: uuid::Uuid,
    /// Model evaluated
    pub model_name: String,
    /// Provider used
    pub provider_name: String,
    /// Overall score (weighted average)
    pub overall_score: f64,
    /// Per-category scores
    pub category_scores: Vec<CategoryScore>,
    /// Benchmark results
    pub benchmark_results: Vec<BenchmarkResultSummary>,
    /// Detailed metrics
    pub metrics: DetailedMetrics,
    /// Cost analysis
    pub cost_analysis: Option<CostAnalysis>,
    /// Configuration used
    pub config: EvaluationConfig,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Summary of a benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResultSummary {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Accuracy score
    pub accuracy: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Number correct
    pub correct_count: usize,
    /// Mean latency
    pub mean_latency_ms: f64,
}

/// Comparison between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Models compared
    pub models: Vec<ModelComparison>,
    /// Benchmarks used
    pub benchmarks: Vec<String>,
    /// Winner per benchmark
    pub benchmark_winners: HashMap<String, String>,
    /// Overall winner
    pub overall_winner: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Model comparison data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Model name
    pub model_name: String,
    /// Provider name
    pub provider_name: String,
    /// Overall score
    pub overall_score: f64,
    /// Per-benchmark scores
    pub benchmark_scores: HashMap<String, f64>,
}

/// Trait for benchmark registry
pub trait BenchmarkRegistry: Send + Sync {
    /// Get a benchmark by ID
    fn get(&self, id: &str) -> Option<&dyn Benchmark>;
}

/// Evaluation orchestrator
pub struct EvaluationOrchestrator {
    /// Storage backend
    storage: Arc<dyn EvaluationStorage>,
}

impl EvaluationOrchestrator {
    /// Create a new orchestrator with storage
    pub fn new(storage: Arc<dyn EvaluationStorage>) -> Self {
        Self { storage }
    }

    /// Create with in-memory storage (for testing)
    pub fn in_memory() -> Self {
        Self {
            storage: Arc::new(crate::storage::InMemoryStorage::new()),
        }
    }

    /// Run a full evaluation suite
    pub async fn run_full_evaluation(
        &self,
        provider: &dyn ModelProvider,
        benchmark_registry: &dyn BenchmarkRegistry,
        config: EvaluationConfig,
    ) -> Result<EvaluationReport> {
        info!(
            "Starting full evaluation for model {} with {} benchmarks",
            provider.default_model().unwrap_or("unknown"),
            config.benchmarks.len()
        );

        let mut benchmark_results = Vec::new();
        let mut category_scores_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut total_samples = 0usize;
        let mut total_correct = 0usize;
        let mut latencies = Vec::new();
        let mut error_count = 0usize;

        for benchmark_id in &config.benchmarks {
            let benchmark = benchmark_registry
                .get(benchmark_id)
                .ok_or_else(|| Error::Config(format!("Benchmark '{}' not found", benchmark_id)))?;

            debug!("Running benchmark: {}", benchmark_id);

            let bench_config = BenchmarkRunConfig {
                max_samples: config.max_samples_per_benchmark,
                temperature: config.temperature,
                max_tokens: config.max_tokens,
                num_few_shot: config.num_few_shot,
                categories: None,
            };

            match benchmark.run(provider, bench_config).await {
                Ok(result) => {
                    let summary = BenchmarkResultSummary {
                        benchmark_id: result.benchmark_id.clone(),
                        benchmark_name: result.benchmark_name.clone(),
                        accuracy: result.statistics.accuracy,
                        sample_count: result.statistics.total_samples,
                        correct_count: result.statistics.correct_count,
                        mean_latency_ms: result.statistics.mean_latency_ms,
                    };

                    total_samples += result.statistics.total_samples;
                    total_correct += result.statistics.correct_count;
                    latencies.push(result.statistics.mean_latency_ms);

                    // Track category scores
                    for (category, stats) in &result.category_stats {
                        category_scores_map
                            .entry(category.clone())
                            .or_default()
                            .push(stats.accuracy);
                    }

                    // Save to storage if enabled
                    if config.save_results {
                        let stored = StoredBenchmarkResult {
                            id: uuid::Uuid::new_v4(),
                            benchmark_id: result.benchmark_id.clone(),
                            model_name: result.model_name.clone(),
                            provider_name: result.provider_name.clone(),
                            accuracy: result.statistics.accuracy,
                            total_samples: result.statistics.total_samples,
                            correct_count: result.statistics.correct_count,
                            mean_latency_ms: result.statistics.mean_latency_ms,
                            raw_result: serde_json::to_value(&result).unwrap_or(serde_json::json!({})),
                            created_at: Utc::now(),
                        };
                        self.storage.save_benchmark_result(&stored).await?;
                    }

                    benchmark_results.push(summary);
                }
                Err(e) => {
                    error_count += 1;
                    tracing::error!("Benchmark {} failed: {}", benchmark_id, e);
                }
            }
        }

        // Calculate overall score
        let overall_score = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };

        // Calculate category scores
        let category_scores: Vec<CategoryScore> = category_scores_map
            .iter()
            .map(|(category, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                CategoryScore {
                    category: category.clone(),
                    score: avg,
                    sample_count: scores.len(),
                    benchmarks: config.benchmarks.clone(),
                }
            })
            .collect();

        // Calculate metrics
        let mean_latency_ms = if !latencies.is_empty() {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        } else {
            0.0
        };

        let p95_latency_ms = {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if !sorted.is_empty() {
                sorted[(sorted.len() as f64 * 0.95) as usize % sorted.len()]
            } else {
                0.0
            }
        };

        let report = EvaluationReport {
            id: uuid::Uuid::new_v4(),
            model_name: provider.default_model().unwrap_or("unknown").to_string(),
            provider_name: provider.provider_name().to_string(),
            overall_score,
            category_scores,
            benchmark_results,
            metrics: DetailedMetrics {
                mean_latency_ms,
                p95_latency_ms,
                total_tokens: 0,
                tokens_per_second: None,
                error_count,
            },
            cost_analysis: None,
            config,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        // Save report to storage if enabled
        if report.config.save_results {
            let stored = StoredEvaluationReport {
                id: report.id,
                model_name: report.model_name.clone(),
                provider_name: report.provider_name.clone(),
                overall_score: report.overall_score,
                benchmark_count: report.benchmark_results.len(),
                category_scores: serde_json::to_value(&report.category_scores).unwrap_or(serde_json::json!({})),
                raw_report: serde_json::to_value(&report).unwrap_or(serde_json::json!({})),
                created_at: report.timestamp,
            };
            self.storage.save_evaluation_report(&stored).await?;
        }

        info!("Evaluation complete. Overall score: {:.2}%", overall_score * 100.0);
        Ok(report)
    }

    /// Get evaluation history for a model
    pub async fn get_history(
        &self,
        model_name: &str,
        limit: usize,
    ) -> Result<Vec<StoredEvaluationReport>> {
        self.storage
            .list_evaluation_reports(
                QueryParams::new()
                    .with_model(model_name)
                    .with_limit(limit),
            )
            .await
    }

    /// Get benchmark results for a model
    pub async fn get_benchmark_results(
        &self,
        model_name: &str,
        benchmark_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StoredBenchmarkResult>> {
        self.storage
            .get_benchmark_history(model_name, benchmark_id, limit)
            .await
    }
}
