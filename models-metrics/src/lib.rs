//! Metrics engine for Models Lab
//!
//! This crate provides a metrics framework for evaluating LLM outputs,
//! including speed metrics, accuracy metrics, and more.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod accuracy;
pub mod drift;
pub mod efficiency;
pub mod hallucination;
pub mod speed;

pub use accuracy::{BleuMetric, ExactMatchMetric, F1ScoreMetric, RougeMetric, RougeType};
pub use drift::{OutputDriftMetric, TemperatureVarianceMetric};
pub use efficiency::{
    CostEfficiencyMetric, CostPerTokenMetric, LatencyEfficiencyMetric, QualityPerDollarMetric,
    TotalCostMetric,
};
pub use hallucination::{FactualConsistencyMetric, HallucinationRateMetric};
pub use speed::{FirstTokenLatencyMetric, P95LatencyMetric, TokensPerSecondMetric};

// ============================================================================
// Core Types
// ============================================================================

/// Unit of measurement for a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricUnit {
    /// Tokens per second
    TokensPerSecond,
    /// Milliseconds
    Milliseconds,
    /// Percentage (0-100)
    Percentage,
    /// Score (0.0-1.0)
    Score,
    /// Count
    Count,
    /// Custom unit
    Custom,
}

impl std::fmt::Display for MetricUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricUnit::TokensPerSecond => write!(f, "tokens/s"),
            MetricUnit::Milliseconds => write!(f, "ms"),
            MetricUnit::Percentage => write!(f, "%"),
            MetricUnit::Score => write!(f, "score"),
            MetricUnit::Count => write!(f, "count"),
            MetricUnit::Custom => write!(f, "custom"),
        }
    }
}

/// Direction for metric comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Higher is better
    HigherIsBetter,
    /// Lower is better
    LowerIsBetter,
}

/// Data required for metric calculation
#[derive(Debug, Clone)]
pub struct MetricData {
    /// Generated text from the model
    pub generated_text: String,
    /// Expected/reference text (if applicable)
    pub expected_text: Option<String>,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: Option<u64>,
    /// Number of tokens generated
    pub tokens_generated: Option<u64>,
    /// Number of prompt tokens
    pub prompt_tokens: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MetricData {
    /// Create new metric data
    pub fn new(generated_text: impl Into<String>) -> Self {
        Self {
            generated_text: generated_text.into(),
            expected_text: None,
            latency_ms: 0,
            time_to_first_token_ms: None,
            tokens_generated: None,
            prompt_tokens: None,
            metadata: HashMap::new(),
        }
    }

    /// Set expected text
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected_text = Some(expected.into());
        self
    }

    /// Set latency
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Set time to first token
    pub fn with_ttft(mut self, ms: u64) -> Self {
        self.time_to_first_token_ms = Some(ms);
        self
    }

    /// Set token counts
    pub fn with_tokens(mut self, generated: u64, prompt: u64) -> Self {
        self.tokens_generated = Some(generated);
        self.prompt_tokens = Some(prompt);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Result of a metric calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Name of the metric
    pub name: String,
    /// The calculated value
    pub value: f64,
    /// Unit of measurement
    pub unit: MetricUnit,
    /// Whether higher is better
    pub direction: MetricDirection,
    /// Optional details about the calculation
    pub details: Option<String>,
}

impl MetricResult {
    pub fn new(name: impl Into<String>, value: f64, unit: MetricUnit, direction: MetricDirection) -> Self {
        Self {
            name: name.into(),
            value,
            unit,
            direction,
            details: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

// ============================================================================
// Metric Trait
// ============================================================================

/// Trait for all metrics
pub trait Metric: Send + Sync {
    /// Get the metric name
    fn name(&self) -> &str;

    /// Get the unit of measurement
    fn unit(&self) -> MetricUnit;

    /// Get the comparison direction
    fn direction(&self) -> MetricDirection;

    /// Calculate the metric from the given data
    fn calculate(&self, data: &MetricData) -> f64;

    /// Calculate and return a full result
    fn calculate_result(&self, data: &MetricData) -> MetricResult {
        MetricResult::new(
            self.name(),
            self.calculate(data),
            self.unit(),
            self.direction(),
        )
    }

    /// Get a description of the metric
    fn description(&self) -> &str;
}

// ============================================================================
// Metrics Engine
// ============================================================================

/// Engine for running multiple metrics
pub struct MetricsEngine {
    /// Registered metrics
    metrics: HashMap<String, Box<dyn Metric>>,
}

impl MetricsEngine {
    /// Create a new empty engine
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Create an engine with all built-in metrics registered
    pub fn with_builtin_metrics() -> Self {
        let mut engine = Self::new();
        engine.register_all_builtin();
        engine
    }

    /// Register a metric
    pub fn register<M: Metric + 'static>(&mut self, metric: M) {
        self.metrics.insert(metric.name().to_string(), Box::new(metric));
    }

    /// Register all built-in metrics
    pub fn register_all_builtin(&mut self) {
        // Speed metrics
        self.register(TokensPerSecondMetric::new());
        self.register(FirstTokenLatencyMetric::new());
        self.register(P95LatencyMetric::new());

        // Accuracy metrics
        self.register(ExactMatchMetric::new());
        self.register(F1ScoreMetric::new());
        self.register(BleuMetric::new());
        self.register(RougeMetric::rouge1());
        self.register(RougeMetric::rouge2());
        self.register(RougeMetric::rouge_l());

        // Efficiency metrics
        self.register(CostPerTokenMetric::new());
        self.register(TotalCostMetric::new());
        self.register(QualityPerDollarMetric::new());
        self.register(LatencyEfficiencyMetric::new());
        self.register(CostEfficiencyMetric::new());

        // Hallucination metrics
        self.register(HallucinationRateMetric::new());
        self.register(FactualConsistencyMetric::new());

        // Drift metrics
        self.register(OutputDriftMetric::new());
        self.register(TemperatureVarianceMetric::new());
    }

    /// Get a metric by name
    pub fn get(&self, name: &str) -> Option<&dyn Metric> {
        self.metrics.get(name).map(|b| b.as_ref())
    }

    /// List all registered metrics
    pub fn list_metrics(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }

    /// Calculate a specific metric
    pub fn calculate(&self, name: &str, data: &MetricData) -> Option<MetricResult> {
        self.metrics.get(name).map(|m| m.calculate_result(data))
    }

    /// Calculate all registered metrics
    pub fn calculate_all(&self, data: &MetricData) -> HashMap<String, MetricResult> {
        self.metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.calculate_result(data)))
            .collect()
    }

    /// Calculate metrics for multiple samples
    pub fn calculate_aggregate(
        &self,
        metric_names: &[&str],
        samples: &[MetricData],
    ) -> HashMap<String, AggregateStats> {
        let mut results: HashMap<String, AggregateStats> = HashMap::new();

        for name in metric_names {
            if let Some(metric) = self.metrics.get(*name) {
                let values: Vec<f64> = samples
                    .iter()
                    .map(|data| metric.calculate(data))
                    .collect();
                results.insert(name.to_string(), AggregateStats::from_values(&values));
            }
        }

        results
    }
}

impl Default for MetricsEngine {
    fn default() -> Self {
        Self::with_builtin_metrics()
    }
}

// ============================================================================
// Aggregate Statistics
// ============================================================================

/// Aggregate statistics for a metric across multiple samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of samples
    pub count: usize,
    /// P5 percentile
    pub p5: f64,
    /// P25 percentile
    pub p25: f64,
    /// P75 percentile
    pub p75: f64,
    /// P95 percentile
    pub p95: f64,
}

impl AggregateStats {
    /// Calculate aggregate statistics from values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                count: 0,
                p5: 0.0,
                p25: 0.0,
                p75: 0.0,
                p95: 0.0,
            };
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        // Sort for percentiles
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile(&sorted, 50.0);
        let p5 = percentile(&sorted, 5.0);
        let p25 = percentile(&sorted, 25.0);
        let p75 = percentile(&sorted, 75.0);
        let p95 = percentile(&sorted, 95.0);

        // Standard deviation
        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        Self {
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[count - 1],
            count,
            p5,
            p25,
            p75,
            p95,
        }
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_engine_creation() {
        let engine = MetricsEngine::with_builtin_metrics();
        let metrics = engine.list_metrics();
        assert!(metrics.contains(&"tokens_per_second"));
        assert!(metrics.contains(&"exact_match"));
        assert!(metrics.contains(&"f1_score"));
    }

    #[test]
    fn test_aggregate_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = AggregateStats::from_values(&values);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_metric_data() {
        let data = MetricData::new("test output")
            .with_expected("expected output")
            .with_latency(100)
            .with_tokens(50, 10);

        assert_eq!(data.generated_text, "test output");
        assert_eq!(data.expected_text, Some("expected output".to_string()));
        assert_eq!(data.latency_ms, 100);
        assert_eq!(data.tokens_generated, Some(50));
    }
}
