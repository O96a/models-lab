//! Model comparison functionality
//!
//! This module provides comprehensive model comparison capabilities including
//! statistical significance testing, per-category scoring, and winner determination.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Models to compare with their configurations
    pub models: Vec<ModelConfig>,
    /// Benchmarks to run
    pub benchmarks: Vec<String>,
    /// Maximum samples per benchmark
    pub max_samples: Option<usize>,
    /// Temperature for generation
    pub temperature: f64,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Number of few-shot examples
    pub num_few_shot: usize,
    /// Confidence level for statistical tests (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            benchmarks: vec!["mmlu".to_string()],
            max_samples: None,
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 5,
            confidence_level: 0.95,
        }
    }
}

/// Configuration for a single model in comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Provider name (ollama, openai, etc.)
    pub provider: String,
    /// Model name
    pub model: String,
    /// Base URL for the provider (if applicable)
    pub base_url: Option<String>,
    /// API key (if applicable)
    pub api_key: Option<String>,
}

/// Per-category score for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCategoryScore {
    /// Category name
    pub category: String,
    /// Average score (0.0 to 1.0)
    pub score: f64,
    /// Number of benchmarks in this category
    pub benchmark_count: usize,
    /// Individual benchmark scores in this category
    pub benchmark_scores: HashMap<String, f64>,
}

/// Per-benchmark result for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmarkResult {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Benchmark name
    pub benchmark_name: String,
    /// Category
    pub category: String,
    /// Accuracy score
    pub accuracy: f64,
    /// Number of samples evaluated
    pub sample_count: usize,
    /// Mean latency in milliseconds
    pub mean_latency_ms: f64,
    /// Standard deviation of scores (if available)
    pub std_dev: Option<f64>,
}

/// Statistical significance test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Benchmark being tested
    pub benchmark_id: String,
    /// Model A name
    pub model_a: String,
    /// Model B name
    pub model_b: String,
    /// Difference in scores (A - B)
    pub score_difference: f64,
    /// P-value from statistical test
    pub p_value: f64,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Confidence level used
    pub confidence_level: f64,
    /// Winner (None if not significant)
    pub winner: Option<String>,
}

/// Overall winner determination with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinnerInfo {
    /// Winning model name
    pub model: String,
    /// Winning model provider
    pub provider: String,
    /// Overall score
    pub overall_score: f64,
    /// Number of benchmarks won
    pub benchmarks_won: usize,
    /// Number of categories won
    pub categories_won: usize,
    /// Winning margin (percentage points)
    pub margin: f64,
    /// Reasoning for the win
    pub reasoning: String,
}

/// Per-benchmark winner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkWinner {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Winning model name
    pub model: String,
    /// Winning score
    pub score: f64,
    /// Score difference from second place
    pub margin: f64,
    /// Whether win is statistically significant
    pub is_significant: bool,
}

/// Complete comparison report for multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Unique report ID
    pub id: uuid::Uuid,
    /// Models compared with their detailed results
    pub model_results: Vec<ModelComparisonResult>,
    /// Benchmarks used in comparison
    pub benchmarks: Vec<String>,
    /// Per-benchmark winners
    pub benchmark_winners: Vec<BenchmarkWinner>,
    /// Per-category winners
    pub category_winners: HashMap<String, String>,
    /// Statistical significance tests
    pub statistical_tests: Vec<StatisticalTest>,
    /// Overall winner
    pub overall_winner: WinnerInfo,
    /// Comparison timestamp
    pub timestamp: DateTime<Utc>,
    /// Comparison configuration
    pub config: ComparisonConfig,
}

/// Comparison result for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    /// Model name
    pub model_name: String,
    /// Provider name
    pub provider_name: String,
    /// Overall score (weighted average across all benchmarks)
    pub overall_score: f64,
    /// Per-category scores
    pub category_scores: Vec<ModelCategoryScore>,
    /// Per-benchmark results
    pub benchmark_results: Vec<ModelBenchmarkResult>,
    /// Rank (1 = best)
    pub rank: usize,
    /// Total samples evaluated
    pub total_samples: usize,
    /// Average latency across all benchmarks
    pub avg_latency_ms: f64,
}

impl ComparisonReport {
    /// Create a new empty comparison report
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            model_results: Vec::new(),
            benchmarks: config.benchmarks.clone(),
            benchmark_winners: Vec::new(),
            category_winners: HashMap::new(),
            statistical_tests: Vec::new(),
            overall_winner: WinnerInfo {
                model: String::new(),
                provider: String::new(),
                overall_score: 0.0,
                benchmarks_won: 0,
                categories_won: 0,
                margin: 0.0,
                reasoning: String::new(),
            },
            timestamp: Utc::now(),
            config,
        }
    }

    /// Add a model's comparison result
    pub fn add_model_result(&mut self, result: ModelComparisonResult) {
        self.model_results.push(result);
    }

    /// Calculate per-benchmark winners
    pub fn calculate_benchmark_winners(&mut self) {
        let mut benchmark_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        // Collect scores for each benchmark
        for model_result in &self.model_results {
            for bench_result in &model_result.benchmark_results {
                benchmark_scores
                    .entry(bench_result.benchmark_id.clone())
                    .or_default()
                    .push((model_result.model_name.clone(), bench_result.accuracy));
            }
        }

        // Determine winner for each benchmark
        for (benchmark_id, mut scores) in benchmark_scores {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((winner, winner_score)) = scores.first() {
                let margin = if scores.len() > 1 {
                    winner_score - scores[1].1
                } else {
                    0.0
                };

                self.benchmark_winners.push(BenchmarkWinner {
                    benchmark_id,
                    model: winner.clone(),
                    score: *winner_score,
                    margin,
                    is_significant: margin > 0.05, // 5% margin threshold
                });
            }
        }
    }

    /// Calculate per-category winners
    pub fn calculate_category_winners(&mut self) {
        let mut category_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        // Collect scores for each category
        for model_result in &self.model_results {
            for cat_score in &model_result.category_scores {
                category_scores
                    .entry(cat_score.category.clone())
                    .or_default()
                    .push((model_result.model_name.clone(), cat_score.score));
            }
        }

        // Determine winner for each category
        for (category, mut scores) in category_scores {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((winner, _)) = scores.first() {
                self.category_winners.insert(category, winner.clone());
            }
        }
    }

    /// Perform statistical significance tests between model pairs
    pub fn calculate_statistical_significance(&mut self) {
        let confidence_level = self.config.confidence_level;
        let alpha = 1.0 - confidence_level;

        // For each benchmark, compare each pair of models
        for benchmark_id in &self.benchmarks {
            let mut model_scores: Vec<(String, f64)> = Vec::new();

            for model_result in &self.model_results {
                if let Some(bench_result) = model_result
                    .benchmark_results
                    .iter()
                    .find(|r| &r.benchmark_id == benchmark_id)
                {
                    model_scores.push((model_result.model_name.clone(), bench_result.accuracy));
                }
            }

            // Compare each pair
            for i in 0..model_scores.len() {
                for j in (i + 1)..model_scores.len() {
                    let (model_a, score_a) = &model_scores[i];
                    let (model_b, score_b) = &model_scores[j];

                    let difference = score_a - score_b;

                    // Simplified significance test based on sample size and score difference
                    // In a real implementation, this would use proper statistical tests
                    // like Welch's t-test with actual variance data
                    let p_value = estimate_p_value(*score_a, *score_b);
                    let is_significant = p_value < alpha;

                    let winner = if is_significant {
                        if difference > 0.0 {
                            Some(model_a.clone())
                        } else {
                            Some(model_b.clone())
                        }
                    } else {
                        None
                    };

                    self.statistical_tests.push(StatisticalTest {
                        benchmark_id: benchmark_id.clone(),
                        model_a: model_a.clone(),
                        model_b: model_b.clone(),
                        score_difference: difference.abs(),
                        p_value,
                        is_significant,
                        confidence_level,
                        winner,
                    });
                }
            }
        }
    }

    /// Calculate overall winner based on all metrics
    pub fn calculate_overall_winner(&mut self) {
        if self.model_results.is_empty() {
            return;
        }

        // Sort by overall score (descending)
        self.model_results
            .sort_by(|a, b| b.overall_score.partial_cmp(&a.overall_score).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        for (i, result) in self.model_results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        let winner = &self.model_results[0];
        let runner_up = self.model_results.get(1);

        // Count benchmarks won
        let benchmarks_won = self
            .benchmark_winners
            .iter()
            .filter(|w| w.model == winner.model_name)
            .count();

        // Count categories won
        let categories_won = self
            .category_winners
            .values()
            .filter(|&v| v == &winner.model_name)
            .count();

        // Calculate margin
        let margin = if let Some(runner) = runner_up {
            (winner.overall_score - runner.overall_score) * 100.0
        } else {
            0.0
        };

        // Generate reasoning
        let reasoning = generate_winner_reasoning(
            &winner.model_name,
            winner.overall_score,
            benchmarks_won,
            categories_won,
            margin,
        );

        self.overall_winner = WinnerInfo {
            model: winner.model_name.clone(),
            provider: winner.provider_name.clone(),
            overall_score: winner.overall_score,
            benchmarks_won,
            categories_won,
            margin,
            reasoning,
        };
    }

    /// Generate a summary table for terminal display
    pub fn generate_summary_table(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&"=".repeat(80));
        output.push('\n');
        output.push_str("MODEL COMPARISON REPORT\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');
        output.push_str(&format!("Report ID: {}\n", self.id));
        output.push_str(&format!("Generated: {}\n", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str(&format!("Benchmarks: {}\n", self.benchmarks.join(", ")));
        output.push('\n');

        // Overall ranking table
        output.push_str("OVERALL RANKINGS\n");
        output.push_str(&"-".repeat(80));
        output.push('\n');
        output.push_str(&format!("{:<5} {:<25} {:<15} {:<12} {:<12}\n", "Rank", "Model", "Provider", "Score", "Latency"));
        output.push_str(&"-".repeat(80));
        output.push('\n');

        for result in &self.model_results {
            output.push_str(&format!(
                "{:<5} {:<25} {:<15} {:>10.2}% {:>10.0}ms\n",
                result.rank,
                truncate(&result.model_name, 25),
                truncate(&result.provider_name, 15),
                result.overall_score * 100.0,
                result.avg_latency_ms
            ));
        }

        output.push('\n');

        // Overall winner
        output.push_str("OVERALL WINNER\n");
        output.push_str(&"-".repeat(80));
        output.push('\n');
        output.push_str(&format!("Model: {}\n", self.overall_winner.model));
        output.push_str(&format!("Provider: {}\n", self.overall_winner.provider));
        output.push_str(&format!("Score: {:.2}%\n", self.overall_winner.overall_score * 100.0));
        output.push_str(&format!("Benchmarks Won: {}\n", self.overall_winner.benchmarks_won));
        output.push_str(&format!("Categories Won: {}\n", self.overall_winner.categories_won));
        output.push_str(&format!("Winning Margin: {:.2}pp\n", self.overall_winner.margin));
        output.push('\n');
        output.push_str(&format!("Reasoning: {}\n", self.overall_winner.reasoning));
        output.push('\n');

        // Per-benchmark results
        output.push_str("PER-BENCHMARK RESULTS\n");
        output.push_str(&"-".repeat(80));
        output.push('\n');

        // Build table header with model names
        let mut header = format!("{:<20}", "Benchmark");
        for result in &self.model_results {
            header.push_str(&format!(" {:>12}", truncate(&result.model_name, 12)));
        }
        header.push_str(&format!(" {:>12}\n", "Winner"));
        output.push_str(&header);
        output.push_str(&"-".repeat(80));
        output.push('\n');

        // Build rows for each benchmark
        for benchmark_id in &self.benchmarks {
            let winner = self
                .benchmark_winners
                .iter()
                .find(|w| &w.benchmark_id == benchmark_id);

            let mut row = format!("{:<20}", truncate(benchmark_id, 20));
            for model_result in &self.model_results {
                if let Some(bench_result) = model_result
                    .benchmark_results
                    .iter()
                    .find(|r| &r.benchmark_id == benchmark_id)
                {
                    row.push_str(&format!(" {:>11.1}%", bench_result.accuracy * 100.0));
                } else {
                    row.push_str(&format!(" {:>12}", "N/A"));
                }
            }

            if let Some(w) = winner {
                row.push_str(&format!(" {:>12}\n", truncate(&w.model, 12)));
            } else {
                row.push_str(&format!(" {:>12}\n", "-"));
            }
            output.push_str(&row);
        }

        output.push('\n');

        // Per-category results
        if !self.category_winners.is_empty() {
            output.push_str("PER-CATEGORY RESULTS\n");
            output.push_str(&"-".repeat(80));
            output.push('\n');

            let mut header = format!("{:<20}", "Category");
            for result in &self.model_results {
                header.push_str(&format!(" {:>12}", truncate(&result.model_name, 12)));
            }
            header.push_str(&format!(" {:>12}\n", "Winner"));
            output.push_str(&header);
            output.push_str(&"-".repeat(80));
            output.push('\n');

            // Get all categories
            let mut categories: Vec<String> = self.category_winners.keys().cloned().collect();
            categories.sort();

            for category in categories {
                let winner = self.category_winners.get(&category);

                let mut row = format!("{:<20}", truncate(&category, 20));
                for model_result in &self.model_results {
                    if let Some(cat_score) = model_result
                        .category_scores
                        .iter()
                        .find(|c| c.category == category)
                    {
                        row.push_str(&format!(" {:>11.1}%", cat_score.score * 100.0));
                    } else {
                        row.push_str(&format!(" {:>12}", "N/A"));
                    }
                }

                if let Some(w) = winner {
                    row.push_str(&format!(" {:>12}\n", truncate(w, 12)));
                } else {
                    row.push_str(&format!(" {:>12}\n", "-"));
                }
                output.push_str(&row);
            }

            output.push('\n');
        }

        // Statistical significance results
        if !self.statistical_tests.is_empty() {
            output.push_str("STATISTICAL SIGNIFICANCE TESTS\n");
            output.push_str(&"-".repeat(80));
            output.push('\n');
            output.push_str(&format!(
                "Confidence Level: {:.0}%\n",
                self.config.confidence_level * 100.0
            ));
            output.push('\n');

            output.push_str(&format!(
                "{:<20} {:<15} {:<15} {:>12} {:>12} {:>12}\n",
                "Benchmark", "Model A", "Model B", "Diff", "P-Value", "Significant"
            ));
            output.push_str(&"-".repeat(80));
            output.push('\n');

            for test in &self.statistical_tests {
                output.push_str(&format!(
                    "{:<20} {:<15} {:<15} {:>11.2}% {:>12.4} {:>12}\n",
                    truncate(&test.benchmark_id, 20),
                    truncate(&test.model_a, 15),
                    truncate(&test.model_b, 15),
                    test.score_difference * 100.0,
                    test.p_value,
                    if test.is_significant { "Yes" } else { "No" }
                ));
            }

            output.push('\n');
        }

        output.push_str(&"=".repeat(80));
        output.push('\n');

        output
    }

    /// Get models sorted by rank
    pub fn get_ranked_models(&self) -> &[ModelComparisonResult] {
        &self.model_results
    }

    /// Get winner for a specific benchmark
    pub fn get_benchmark_winner(&self, benchmark_id: &str) -> Option<&BenchmarkWinner> {
        self.benchmark_winners.iter().find(|w| w.benchmark_id == benchmark_id)
    }

    /// Get winner for a specific category
    pub fn get_category_winner(&self, category: &str) -> Option<&String> {
        self.category_winners.get(category)
    }
}

/// Helper function to truncate strings
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Estimate p-value for score difference (simplified)
/// In a real implementation, this would use proper statistical tests
fn estimate_p_value(score_a: f64, score_b: f64) -> f64 {
    let diff = (score_a - score_b).abs();

    // Simplified estimation based on effect size
    // This is a placeholder for proper statistical testing
    if diff < 0.02 {
        0.5 // Not significant
    } else if diff < 0.05 {
        0.2 // Weak evidence
    } else if diff < 0.10 {
        0.05 // Significant at 95% level
    } else if diff < 0.15 {
        0.01 // Highly significant
    } else {
        0.001 // Very highly significant
    }
}

/// Generate reasoning text for the winner
fn generate_winner_reasoning(
    model_name: &str,
    score: f64,
    benchmarks_won: usize,
    categories_won: usize,
    margin: f64,
) -> String {
    let mut parts = Vec::new();

    parts.push(format!("{} achieved the highest overall score of {:.2}%", model_name, score * 100.0));

    if benchmarks_won > 0 {
        parts.push(format!("won {} benchmark(s)", benchmarks_won));
    }

    if categories_won > 0 {
        parts.push(format!("led in {} category(s)", categories_won));
    }

    if margin > 5.0 {
        parts.push(format!("with a substantial margin of {:.2} percentage points", margin));
    } else if margin > 2.0 {
        parts.push(format!("with a margin of {:.2} percentage points", margin));
    } else {
        parts.push("by a narrow margin".to_string());
    }

    parts.join(", ") + "."
}

/// Builder for constructing comparison reports
pub struct ComparisonReportBuilder {
    config: ComparisonConfig,
    results: Vec<ModelComparisonResult>,
}

impl ComparisonReportBuilder {
    /// Create a new builder with the given configuration
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Add a model result
    pub fn add_result(mut self, result: ModelComparisonResult) -> Self {
        self.results.push(result);
        self
    }

    /// Build the final report with all calculations
    pub fn build(self) -> ComparisonReport {
        let mut report = ComparisonReport::new(self.config);

        for result in self.results {
            report.add_model_result(result);
        }

        report.calculate_benchmark_winners();
        report.calculate_category_winners();
        report.calculate_statistical_significance();
        report.calculate_overall_winner();

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_report_creation() {
        let config = ComparisonConfig::default();
        let report = ComparisonReport::new(config);

        assert!(report.model_results.is_empty());
        assert!(report.benchmark_winners.is_empty());
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_benchmark_winner_calculation() {
        let mut report = ComparisonReport::new(ComparisonConfig::default());

        // Add model results
        report.add_model_result(ModelComparisonResult {
            model_name: "model_a".to_string(),
            provider_name: "ollama".to_string(),
            overall_score: 0.8,
            category_scores: vec![],
            benchmark_results: vec![
                ModelBenchmarkResult {
                    benchmark_id: "mmlu".to_string(),
                    benchmark_name: "MMLU".to_string(),
                    category: "reasoning".to_string(),
                    accuracy: 0.85,
                    sample_count: 100,
                    mean_latency_ms: 100.0,
                    std_dev: None,
                },
            ],
            rank: 0,
            total_samples: 100,
            avg_latency_ms: 100.0,
        });

        report.add_model_result(ModelComparisonResult {
            model_name: "model_b".to_string(),
            provider_name: "ollama".to_string(),
            overall_score: 0.7,
            category_scores: vec![],
            benchmark_results: vec![
                ModelBenchmarkResult {
                    benchmark_id: "mmlu".to_string(),
                    benchmark_name: "MMLU".to_string(),
                    category: "reasoning".to_string(),
                    accuracy: 0.75,
                    sample_count: 100,
                    mean_latency_ms: 120.0,
                    std_dev: None,
                },
            ],
            rank: 0,
            total_samples: 100,
            avg_latency_ms: 120.0,
        });

        report.calculate_benchmark_winners();

        assert_eq!(report.benchmark_winners.len(), 1);
        assert_eq!(report.benchmark_winners[0].model, "model_a");
        assert_eq!(report.benchmark_winners[0].score, 0.85);
    }
}
