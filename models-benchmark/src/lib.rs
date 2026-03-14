//! Benchmark framework for Models Lab
//!
//! This crate provides a framework for running LLM benchmarks,
//! including built-in benchmarks like MMLU, GSM8K, and more.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod coding;
pub mod context;
pub mod hallucination;
pub mod instruction;
pub mod math;
pub mod multiturn;
pub mod registry;
pub mod reasoning;
pub mod safety;

pub use coding::humaneval::HumanEvalBenchmark;
pub use coding::mbpp::MbppBenchmark;
pub use context::long_context::LongContextBenchmark;
pub use context::needle::NeedleInHaystackBenchmark;
pub use hallucination::truthfulqa::TruthfulQABenchmark;
pub use instruction::alpaca_eval::AlpacaEvalBenchmark;
pub use math::gsm8k::GSM8KBenchmark;
pub use math::math::MATHBenchmark;
pub use multiturn::mt_bench::MTBenchBenchmark;
pub use registry::BenchmarkRegistry;
pub use reasoning::arc::{ARCBenchmark, ARCVariant};
pub use reasoning::bbh::BbhBenchmark;
pub use reasoning::mmlu::MMLUBenchmark;
pub use safety::safety::SafetyBenchmark;

// ============================================================================
// Core Types
// ============================================================================

/// Category of benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkCategory {
    /// Reasoning benchmarks (MMLU, BBH, ARC)
    Reasoning,
    /// Math benchmarks (GSM8K, MATH)
    Math,
    /// Coding benchmarks (HumanEval, MBPP)
    Coding,
    /// Hallucination benchmarks (TruthfulQA)
    Hallucination,
    /// Long context benchmarks
    Context,
    /// Instruction following benchmarks
    Instruction,
    /// Multi-turn conversation benchmarks
    Multiturn,
    /// Safety benchmarks
    Safety,
    /// General purpose
    General,
}

impl std::fmt::Display for BenchmarkCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchmarkCategory::Reasoning => write!(f, "Reasoning"),
            BenchmarkCategory::Math => write!(f, "Math"),
            BenchmarkCategory::Coding => write!(f, "Coding"),
            BenchmarkCategory::Hallucination => write!(f, "Hallucination"),
            BenchmarkCategory::Context => write!(f, "Context"),
            BenchmarkCategory::Instruction => write!(f, "Instruction Following"),
            BenchmarkCategory::Multiturn => write!(f, "Multi-turn"),
            BenchmarkCategory::Safety => write!(f, "Safety"),
            BenchmarkCategory::General => write!(f, "General"),
        }
    }
}

/// Difficulty level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

impl Default for Difficulty {
    fn default() -> Self {
        Self::Medium
    }
}

// ============================================================================
// Dataset Types
// ============================================================================

/// Metadata about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Name of the dataset
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Source URL
    pub source_url: Option<String>,
    /// Number of samples
    pub num_samples: u64,
    /// Categories covered
    pub categories: Vec<String>,
    /// License
    pub license: Option<String>,
}

/// A single data sample for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    /// Unique identifier
    pub id: String,
    /// The input prompt/question
    pub input: String,
    /// Expected output (for evaluation)
    pub expected_output: Option<String>,
    /// Difficulty level
    pub difficulty: Option<Difficulty>,
    /// Category/subject
    pub category: Option<String>,
    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DataSample {
    pub fn new(id: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected_output: None,
            difficulty: None,
            category: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected_output = Some(expected.into());
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = Some(difficulty);
        self
    }
}

/// A dataset for benchmarking
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Dataset name
    pub name: String,
    /// Dataset samples
    pub samples: Vec<DataSample>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

impl Dataset {
    pub fn new(name: impl Into<String>, samples: Vec<DataSample>) -> Self {
        let name = name.into();
        let num_samples = samples.len() as u64;
        Self {
            name: name.clone(),
            samples,
            metadata: DatasetMetadata {
                name,
                description: None,
                source_url: None,
                num_samples,
                categories: Vec::new(),
                license: None,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result for a single sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResult {
    /// Sample ID
    pub sample_id: String,
    /// Generated output
    pub generated_output: String,
    /// Expected output
    pub expected_output: Option<String>,
    /// Whether the answer is correct
    pub correct: bool,
    /// Score (0.0 to 1.0)
    pub score: f64,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Error message if any
    pub error: Option<String>,
}

impl SampleResult {
    pub fn correct(sample_id: impl Into<String>, generated: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            sample_id: sample_id.into(),
            generated_output: generated.into(),
            expected_output: None,
            correct: true,
            score: 1.0,
            latency_ms,
            error: None,
        }
    }

    pub fn incorrect(sample_id: impl Into<String>, generated: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            sample_id: sample_id.into(),
            generated_output: generated.into(),
            expected_output: None,
            correct: false,
            score: 0.0,
            latency_ms,
            error: None,
        }
    }

    pub fn error(sample_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            sample_id: sample_id.into(),
            generated_output: String::new(),
            expected_output: None,
            correct: false,
            score: 0.0,
            latency_ms: 0,
            error: Some(error.into()),
        }
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected_output = Some(expected.into());
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self.correct = score >= 0.5;
        self
    }
}

/// Statistics for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
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
    /// Mean score
    pub mean_score: f64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Tokens per second (if available)
    pub tokens_per_second: Option<f64>,
}

impl BenchmarkStatistics {
    pub fn from_results(results: &[SampleResult]) -> Self {
        if results.is_empty() {
            return Self {
                total_samples: 0,
                correct_count: 0,
                accuracy: 0.0,
                mean_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                mean_score: 0.0,
                total_tokens: 0,
                tokens_per_second: None,
            };
        }

        let total_samples = results.len();
        let correct_count = results.iter().filter(|r| r.correct).count();
        let accuracy = correct_count as f64 / total_samples as f64;

        let latencies: Vec<f64> = results.iter().map(|r| r.latency_ms as f64).collect();
        let mean_latency_ms = latencies.iter().sum::<f64>() / total_samples as f64;

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_latency_ms = sorted_latencies[(total_samples as f64 * 0.95) as usize % total_samples];

        let mean_score = results.iter().map(|r| r.score).sum::<f64>() / total_samples as f64;

        Self {
            total_samples,
            correct_count,
            accuracy,
            mean_latency_ms,
            p95_latency_ms,
            mean_score,
            total_tokens: 0, // Would need token counts from results
            tokens_per_second: None,
        }
    }
}

/// Configuration for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Maximum samples to evaluate (None = all)
    pub max_samples: Option<usize>,
    /// Random seed for sampling
    pub seed: Option<u64>,
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Number of few-shot examples
    pub num_few_shot: usize,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Categories to filter (None = all)
    pub categories: Option<Vec<String>>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            max_samples: None,
            seed: None,
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 0,
            stop_sequences: Vec::new(),
            categories: None,
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Model evaluated
    pub model_name: String,
    /// Provider used
    pub provider_name: String,
    /// Overall statistics
    pub statistics: BenchmarkStatistics,
    /// Individual sample results
    pub sample_results: Vec<SampleResult>,
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Per-category statistics
    pub category_stats: HashMap<String, BenchmarkStatistics>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ============================================================================
// Benchmark Trait
// ============================================================================

/// Trait that all benchmarks must implement
#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Get the benchmark ID
    fn id(&self) -> &str;

    /// Get the benchmark name
    fn name(&self) -> &str;

    /// Get the benchmark category
    fn category(&self) -> BenchmarkCategory;

    /// Get the benchmark description
    fn description(&self) -> &str;

    /// Load the dataset for this benchmark
    async fn load_dataset(&self) -> models_core::error::Result<Dataset>;

    /// Run the benchmark with a provider
    async fn run(
        &self,
        provider: &dyn models_core::providers::ModelProvider,
        config: BenchmarkConfig,
    ) -> models_core::error::Result<BenchmarkResult>;

    /// Evaluate a single response
    fn evaluate_response(&self, sample: &DataSample, response: &str) -> f64;

    /// Format a prompt for the model
    fn format_prompt(&self, sample: &DataSample, few_shot_examples: &[DataSample]) -> String;
}
