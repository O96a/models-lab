//! MATH Benchmark
//!
//! Tests mathematical reasoning with competition mathematics problems.
//! Covers various categories including algebra, counting, geometry, intermediate algebra,
//! number theory, prealgebra, and precalculus.
//!
//! The MATH dataset uses LaTeX-formatted answers, often in \\boxed{answer} format.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, Difficulty, SampleResult,
};

/// MATH Benchmark implementation
pub struct MATHBenchmark {
    /// Few-shot examples for prompting
    few_shot_examples: Vec<DataSample>,
}

impl MATHBenchmark {
    /// Create a new MATH benchmark
    pub fn new() -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
        }
    }

    /// Default few-shot examples for MATH
    fn default_few_shot_examples() -> Vec<DataSample> {
        vec![
            DataSample::new("math_fs1",
                "What is the sum of the positive odd divisors of 60?")
                .with_expected("24")
                .with_category("number_theory")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("math_fs2",
                "If $x = 2$ and $y = 3$, what is the value of $2x + 3y$?")
                .with_expected("13")
                .with_category("algebra")
                .with_difficulty(Difficulty::Easy),
        ]
    }

    /// Find the index of the matching closing brace, accounting for nested braces
    fn find_matching_brace(s: &str) -> Option<usize> {
        let mut depth = 1; // We start after the opening brace
        for (i, c) in s.chars().enumerate() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Load sample MATH data (competition mathematics problems)
    fn load_sample_data() -> Vec<DataSample> {
        vec![
            // Algebra problems
            DataSample::new("math_algebra_1",
                "Solve for $x$: $2x + 5 = 15$")
                .with_expected("5")
                .with_category("algebra")
                .with_difficulty(Difficulty::Easy),

            DataSample::new("math_algebra_2",
                "If $x - y = 4$ and $xy = 21$, what is the value of $x^2 + y^2$?")
                .with_expected("58")
                .with_category("algebra")
                .with_difficulty(Difficulty::Medium),

            // Counting and probability
            DataSample::new("math_counting_1",
                "How many ways can 6 people be arranged in a row?")
                .with_expected("720")
                .with_category("counting")
                .with_difficulty(Difficulty::Medium),

            // Geometry
            DataSample::new("math_geometry_1",
                "What is the area of a circle with radius 3?")
                .with_expected("9\\pi")
                .with_category("geometry")
                .with_difficulty(Difficulty::Easy),

            DataSample::new("math_geometry_2",
                "In a right triangle with legs of length 3 and 4, what is the length of the hypotenuse?")
                .with_expected("5")
                .with_category("geometry")
                .with_difficulty(Difficulty::Easy),

            // Intermediate algebra
            DataSample::new("math_intermediate_1",
                "What is the sum of the roots of $x^2 - 5x + 6 = 0$?")
                .with_expected("5")
                .with_category("intermediate_algebra")
                .with_difficulty(Difficulty::Medium),

            // Number theory
            DataSample::new("math_num_theory_1",
                "What is the remainder when $2^{10}$ is divided by 100?")
                .with_expected("24")
                .with_category("number_theory")
                .with_difficulty(Difficulty::Medium),

            DataSample::new("math_num_theory_2",
                "What is the sum of the prime divisors of 30?")
                .with_expected("10")
                .with_category("number_theory")
                .with_difficulty(Difficulty::Easy),

            // Prealgebra
            DataSample::new("math_prealgebra_1",
                "What is $2^3 + 3^2$?")
                .with_expected("17")
                .with_category("prealgebra")
                .with_difficulty(Difficulty::Easy),

            // Precalculus
            DataSample::new("math_precalc_1",
                r"If $\sin(\theta) = \frac{3}{5}$ and $0 < \theta < \frac{\pi}{2}$, what is $\cos(\theta)$?")
                .with_expected(r"\frac{4}{5}")
                .with_category("precalculus")
                .with_difficulty(Difficulty::Hard),
        ]
    }

    /// Extract the answer from a response
    /// Handles formats: \\boxed{answer}, Answer: X, The answer is X, or just X
    fn extract_answer(response: &str) -> Option<String> {
        let response = response.trim();

        // Try to find \\boxed{...} format (LaTeX boxed answer)
        if let Some(start) = response.find("\\boxed{") {
            let after_boxed = &response[start + 7..]; // Skip "\\boxed{"
            if let Some(end) = Self::find_matching_brace(after_boxed) {
                let answer = &after_boxed[..end];
                return Some(answer.trim().to_string());
            }
        }

        // Try to find $\\boxed{...}$ format (boxed in math mode)
        if let Some(start) = response.find("$\\boxed{") {
            let after_boxed = &response[start + 8..]; // Skip "$\\boxed{"
            if let Some(end) = Self::find_matching_brace(after_boxed) {
                let answer = &after_boxed[..end];
                return Some(answer.trim().to_string());
            }
        }

        // Try to find "answer is" patterns
        for pattern in &["the answer is", "answer is", "answer:", "answer =", "result is", "therefore"] {
            if let Some(pos) = response.to_lowercase().find(pattern) {
                let after = &response[pos + pattern.len()..];
                // Skip any colons or whitespace
                let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '=');
                if let Some(ans) = Self::extract_math_expression(after) {
                    return Some(ans);
                }
            }
        }

        // Try to find "final answer" pattern
        if let Some(pos) = response.to_lowercase().find("final answer") {
            let after = &response[pos + 12..];
            let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '=');
            if let Some(ans) = Self::extract_math_expression(after) {
                return Some(ans);
            }
        }

        // Try to extract from the last line (common for math problems)
        if let Some(last_line) = response.lines().last() {
            if let Some(ans) = Self::extract_math_expression(last_line.trim()) {
                return Some(ans);
            }
        }

        // Last resort: try to extract any math expression
        Self::extract_math_expression(response)
    }

    /// Extract a mathematical expression from a string
    fn extract_math_expression(s: &str) -> Option<String> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        // If it's just a number or simple expression, return it
        if Self::is_simple_math(s) {
            return Some(s.to_string());
        }

        // Look for patterns like "24", "9\\pi", "\\frac{3}{5}", etc.
        // Extract until we hit whitespace or a period at the end
        let mut result = String::new();
        let mut paren_depth = 0;
        let mut brace_depth = 0;
        let mut bracket_depth = 0;

        for c in s.chars() {
            match c {
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                _ => {}
            }

            // Stop at period if we're at top level and have content
            if c == '.' && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 && !result.is_empty() {
                break;
            }

            // Stop at whitespace if we're at top level and have content
            if c.is_whitespace() && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 && !result.is_empty() {
                break;
            }

            result.push(c);
        }

        if result.is_empty() {
            return None;
        }

        Some(result.trim().to_string())
    }

    /// Check if a string is a simple math expression (number, fraction, or simple expression)
    fn is_simple_math(s: &str) -> bool {
        // Remove common math formatting
        let cleaned = s
            .replace('$', "")
            .replace('\\', "");

        // Check if it's mostly digits, operators, and common math symbols
        cleaned.chars().all(|c| {
            c.is_ascii_digit() ||
            c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '=' || c == '^' || c == '(' || c == ')' ||
            c == '{' || c == '}' || c == '[' || c == ']' ||
            c == ' ' || c == '.' || c == 'p' || c == 'i' || // for pi
            c == 'f' || c == 'r' || c == 'a' || c == 'c' // for frac
        })
    }

    /// Check if two mathematical expressions are equivalent
    /// This is a simplified symbolic equivalence checker
    fn expressions_equivalent(expected: &str, actual: &str) -> bool {
        // First normalize both expressions
        let expected_norm = Self::normalize_expression(expected);
        let actual_norm = Self::normalize_expression(actual);

        // Direct string comparison after normalization
        if expected_norm == actual_norm {
            return true;
        }

        // Try numerical comparison if both parse as numbers
        if let (Some(exp_num), Some(act_num)) = (
            Self::parse_number(&expected_norm),
            Self::parse_number(&actual_norm)
        ) {
            return (exp_num - act_num).abs() < 0.01;
        }

        // Check for equivalent fractions
        if Self::fractions_equivalent(&expected_norm, &actual_norm) {
            return true;
        }

        // Check for equivalent pi expressions (e.g., "9pi" and "9\\pi")
        if Self::pi_expressions_equivalent(&expected_norm, &actual_norm) {
            return true;
        }

        false
    }

    /// Normalize a mathematical expression for comparison
    fn normalize_expression(expr: &str) -> String {
        let mut result = expr
            .trim()
            .to_lowercase()
            .replace('$', "")
            .replace(' ', "");

        // Normalize LaTeX fractions
        result = result.replace("\\frac", "frac");

        // Normalize pi representations
        result = result.replace("\\pi", "pi");

        // Remove outer braces if present
        while result.starts_with('{') && result.ends_with('}') {
            result = result[1..result.len()-1].to_string();
        }

        result
    }

    /// Parse a number from a string, handling integers, decimals, and pi expressions
    fn parse_number(s: &str) -> Option<f64> {
        let s = s.trim();

        // Try direct parsing first
        if let Ok(n) = s.parse::<f64>() {
            return Some(n);
        }

        // Handle pi expressions like "9pi" or "3.14pi"
        if s.ends_with("pi") || s.ends_with("\\pi") {
            let coeff_str = s
                .trim_end_matches("\\pi")
                .trim_end_matches("pi");
            let coeff: f64 = if coeff_str.is_empty() {
                1.0
            } else {
                coeff_str.parse().ok()?
            };
            return Some(coeff * std::f64::consts::PI);
        }

        None
    }

    /// Check if two fractions are equivalent
    fn fractions_equivalent(expected: &str, actual: &str) -> bool {
        // Parse fraction format: frac{numerator}{denominator}
        let exp_frac = Self::parse_fraction(expected);
        let act_frac = Self::parse_fraction(actual);

        if let (Some((e_num, e_den)), Some((a_num, a_den))) = (exp_frac, act_frac) {
            // Check if fractions are equivalent
            return e_num * a_den == a_num * e_den;
        }

        false
    }

    /// Parse a fraction in the format frac{numerator}{denominator}
    fn parse_fraction(s: &str) -> Option<(i64, i64)> {
        if !s.starts_with("frac{") {
            return None;
        }

        let inner = &s[5..]; // Skip "frac{"
        let brace_pos = inner.find('}')?;
        let numerator: i64 = inner[..brace_pos].parse().ok()?;

        let after_num = &inner[brace_pos + 1..];
        if !after_num.starts_with('{') {
            return None;
        }

        let inner2 = &after_num[1..];
        let brace_pos2 = inner2.find('}')?;
        let denominator: i64 = inner2[..brace_pos2].parse().ok()?;

        if denominator == 0 {
            return None;
        }

        Some((numerator, denominator))
    }

    /// Check if two pi expressions are equivalent
    fn pi_expressions_equivalent(expected: &str, actual: &str) -> bool {
        // Extract coefficient of pi from both
        let exp_coeff = Self::extract_pi_coefficient(expected);
        let act_coeff = Self::extract_pi_coefficient(actual);

        if let (Some(e), Some(a)) = (exp_coeff, act_coeff) {
            return (e - a).abs() < 0.01;
        }

        false
    }

    /// Extract the coefficient of pi from an expression
    fn extract_pi_coefficient(s: &str) -> Option<f64> {
        let s = s.to_lowercase();

        if s.contains("pi") || s.contains("\\pi") {
            let coeff_part = s
                .replace("\\pi", "")
                .replace("pi", "");

            if coeff_part.is_empty() {
                return Some(1.0);
            }

            return coeff_part.parse().ok();
        }

        None
    }
}

impl Default for MATHBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for MATHBenchmark {
    fn id(&self) -> &str {
        "math"
    }

    fn name(&self) -> &str {
        "MATH (Mathematical Reasoning)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Math
    }

    fn description(&self) -> &str {
        "Tests mathematical reasoning with competition mathematics problems across algebra, geometry, number theory, and more"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples = Self::load_sample_data();
        Ok(Dataset::new("MATH", samples))
    }

    async fn run(
        &self,
        provider: &dyn models_core::providers::ModelProvider,
        config: BenchmarkConfig,
    ) -> models_core::error::Result<BenchmarkResult> {
        let dataset = self.load_dataset().await?;
        let mut samples = dataset.samples;

        // Apply max_samples limit
        if let Some(max) = config.max_samples {
            samples.truncate(max);
        }

        // Apply category filter
        if let Some(ref categories) = config.categories {
            samples.retain(|s| {
                s.category
                    .as_ref()
                    .map(|c| categories.contains(c))
                    .unwrap_or(false)
            });
        }

        debug!("Running MATH benchmark with {} samples", samples.len());

        let mut results = Vec::new();
        let mut category_results: HashMap<String, Vec<SampleResult>> = HashMap::new();

        for sample in &samples {
            let prompt = self.format_prompt(sample, &self.few_shot_examples[..config.num_few_shot.min(2)]);
            let request = models_core::providers::GenerateRequest::new(&prompt)
                .with_temperature(config.temperature)
                .with_max_tokens(config.max_tokens);

            let start = Instant::now();
            let result = provider.generate(request).await;
            let latency_ms = start.elapsed().as_millis() as u64;

            let sample_result = match result {
                Ok(response) => {
                    let score = self.evaluate_response(sample, &response.text);
                    let correct = score >= 0.5;

                    SampleResult {
                        sample_id: sample.id.clone(),
                        generated_output: response.text.clone(),
                        expected_output: sample.expected_output.clone(),
                        correct,
                        score,
                        latency_ms,
                        error: None,
                    }
                }
                Err(e) => SampleResult::error(&sample.id, e.to_string()),
            };

            // Track by category
            if let Some(ref category) = sample.category {
                category_results
                    .entry(category.clone())
                    .or_default()
                    .push(sample_result.clone());
            }

            results.push(sample_result);
        }

        // Calculate statistics
        let statistics = BenchmarkStatistics::from_results(&results);

        // Calculate per-category statistics
        let mut category_stats = HashMap::new();
        for (category, cat_results) in category_results {
            category_stats.insert(category, BenchmarkStatistics::from_results(&cat_results));
        }

        Ok(BenchmarkResult {
            benchmark_id: self.id().to_string(),
            benchmark_name: self.name().to_string(),
            model_name: provider.default_model().unwrap_or("unknown").to_string(),
            provider_name: provider.provider_name().to_string(),
            statistics,
            sample_results: results,
            config,
            category_stats,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
    }

    fn evaluate_response(&self, sample: &DataSample, response: &str) -> f64 {
        let expected = match &sample.expected_output {
            Some(e) => e.trim(),
            None => return 0.0,
        };

        let extracted = match Self::extract_answer(response) {
            Some(ans) => ans,
            None => return 0.0,
        };

        // Use symbolic equivalence checking
        if Self::expressions_equivalent(expected, &extracted) {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, few_shot_examples: &[DataSample]) -> String {
        let mut prompt = String::new();

        prompt.push_str("Solve the following mathematics problem step by step. ");
        prompt.push_str("Show your work and give the final answer in the format: \\boxed{answer}\n\n");

        // Add few-shot examples
        for example in few_shot_examples {
            prompt.push_str(&format!(
                "Problem: {}\nSolution: Let's solve this step by step.\\boxed{{{}}}\n\n",
                example.input,
                example.expected_output.as_deref().unwrap_or("")
            ));
        }

        // Add the actual problem
        prompt.push_str(&format!("Problem: {}\nSolution:", sample.input));

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_benchmark_creation() {
        let benchmark = MATHBenchmark::new();
        assert_eq!(benchmark.id(), "math");
        assert_eq!(benchmark.category(), BenchmarkCategory::Math);
        assert_eq!(benchmark.name(), "MATH (Mathematical Reasoning)");
    }

    #[test]
    fn test_extract_answer_boxed() {
        assert_eq!(
            MATHBenchmark::extract_answer("The answer is \\boxed{42}"),
            Some("42".to_string())
        );
        assert_eq!(
            MATHBenchmark::extract_answer("Therefore, \\boxed{x = 5}"),
            Some("x = 5".to_string())
        );
        assert_eq!(
            MATHBenchmark::extract_answer("$\\boxed{9\\pi}$"),
            Some("9\\pi".to_string())
        );
    }

    #[test]
    fn test_extract_answer_plain() {
        assert_eq!(
            MATHBenchmark::extract_answer("The answer is 24"),
            Some("24".to_string())
        );
        assert_eq!(
            MATHBenchmark::extract_answer("Answer: 42"),
            Some("42".to_string())
        );
        assert_eq!(
            MATHBenchmark::extract_answer("Result is 5"),
            Some("5".to_string())
        );
    }

    #[test]
    fn test_extract_answer_fractions() {
        assert_eq!(
            MATHBenchmark::extract_answer("\\boxed{\\frac{3}{5}}"),
            Some("\\frac{3}{5}".to_string())
        );
    }

    #[test]
    fn test_expressions_equivalent_numbers() {
        assert!(MATHBenchmark::expressions_equivalent("5", "5"));
        assert!(MATHBenchmark::expressions_equivalent("42", "42"));
        assert!(!MATHBenchmark::expressions_equivalent("5", "6"));
    }

    #[test]
    fn test_expressions_equivalent_pi() {
        assert!(MATHBenchmark::expressions_equivalent("9pi", "9\\pi"));
        assert!(MATHBenchmark::expressions_equivalent("9\\pi", "9pi"));
        assert!(MATHBenchmark::expressions_equivalent("$9\\pi$", "9pi"));
    }

    #[test]
    fn test_expressions_equivalent_fractions() {
        assert!(MATHBenchmark::expressions_equivalent("\\frac{3}{5}", "\\frac{3}{5}"));
        assert!(MATHBenchmark::expressions_equivalent("\\frac{2}{4}", "\\frac{1}{2}"));
        assert!(!MATHBenchmark::expressions_equivalent("\\frac{1}{2}", "\\frac{1}{3}"));
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = MATHBenchmark::new();
        let sample = DataSample::new("test", "What is 5 + 5?").with_expected("10");

        assert_eq!(benchmark.evaluate_response(&sample, "\\boxed{10}"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is 10"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "\\boxed{11}"), 0.0);
    }

    #[test]
    fn test_evaluate_response_fractions() {
        let benchmark = MATHBenchmark::new();
        let sample = DataSample::new("test", "What is 1/2 + 1/2?").with_expected("\\frac{2}{4}");

        assert_eq!(benchmark.evaluate_response(&sample, "\\boxed{\\frac{1}{2}}"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "\\boxed{\\frac{2}{4}}"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "\\boxed{1}"), 0.0);
    }

    #[test]
    fn test_load_sample_data() {
        let samples = MATHBenchmark::load_sample_data();
        assert_eq!(samples.len(), 10);

        // Check categories
        let categories: std::collections::HashSet<_> = samples
            .iter()
            .filter_map(|s| s.category.clone())
            .collect();

        assert!(categories.contains("algebra"));
        assert!(categories.contains("geometry"));
        assert!(categories.contains("number_theory"));
        assert!(categories.contains("counting"));
        assert!(categories.contains("intermediate_algebra"));
        assert!(categories.contains("prealgebra"));
        assert!(categories.contains("precalculus"));
    }
}
