//! HumanEval Benchmark
//!
//! Tests code generation capabilities with function completion tasks.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// HumanEval Benchmark implementation
pub struct HumanEvalBenchmark {
    /// Sample problems
    problems: Vec<HumanEvalProblem>,
}

/// A HumanEval problem
#[derive(Debug, Clone)]
struct HumanEvalProblem {
    id: String,
    description: String,
    function_signature: String,
    test_cases: Vec<(Vec<String>, String)>,
    category: String,
}

impl HumanEvalBenchmark {
    /// Create a new HumanEval benchmark
    pub fn new() -> Self {
        Self {
            problems: Self::load_sample_problems(),
        }
    }

    /// Load sample HumanEval problems
    fn load_sample_problems() -> Vec<HumanEvalProblem> {
        vec![
            HumanEvalProblem {
                id: "humaneval_0".to_string(),
                description: "Check if a number is even".to_string(),
                function_signature: "def is_even(n: int) -> bool:".to_string(),
                test_cases: vec![
                    (vec!["2".to_string()], "True".to_string()),
                    (vec!["3".to_string()], "False".to_string()),
                    (vec!["0".to_string()], "True".to_string()),
                ],
                category: "basic".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_1".to_string(),
                description: "Return the sum of two numbers".to_string(),
                function_signature: "def add(a: int, b: int) -> int:".to_string(),
                test_cases: vec![
                    (vec!["1".to_string(), "2".to_string()], "3".to_string()),
                    (vec!["5".to_string(), "7".to_string()], "12".to_string()),
                    (vec!["-1".to_string(), "1".to_string()], "0".to_string()),
                ],
                category: "basic".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_2".to_string(),
                description: "Find the maximum of a list of numbers".to_string(),
                function_signature: "def find_max(numbers: list) -> int:".to_string(),
                test_cases: vec![
                    (vec!["[1, 2, 3]".to_string()], "3".to_string()),
                    (vec!["[5, 2, 8, 1]".to_string()], "8".to_string()),
                    (vec!["[-1, -5, -2]".to_string()], "-1".to_string()),
                ],
                category: "basic".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_3".to_string(),
                description: "Check if a string is a palindrome".to_string(),
                function_signature: "def is_palindrome(s: str) -> bool:".to_string(),
                test_cases: vec![
                    (vec!["\"racecar\"".to_string()], "True".to_string()),
                    (vec!["\"hello\"".to_string()], "False".to_string()),
                    (vec!["\"a\"".to_string()], "True".to_string()),
                ],
                category: "string".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_4".to_string(),
                description: "Reverse a string".to_string(),
                function_signature: "def reverse_string(s: str) -> str:".to_string(),
                test_cases: vec![
                    (vec!["\"hello\"".to_string()], "\"olleh\"".to_string()),
                    (vec!["\"a\"".to_string()], "\"a\"".to_string()),
                    (vec!["\"\"".to_string()], "\"\"".to_string()),
                ],
                category: "string".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_5".to_string(),
                description: "Count the number of vowels in a string".to_string(),
                function_signature: "def count_vowels(s: str) -> int:".to_string(),
                test_cases: vec![
                    (vec!["\"hello\"".to_string()], "2".to_string()),
                    (vec!["\"aeiou\"".to_string()], "5".to_string()),
                    (vec!["\"xyz\"".to_string()], "0".to_string()),
                ],
                category: "string".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_6".to_string(),
                description: "Calculate factorial of a number".to_string(),
                function_signature: "def factorial(n: int) -> int:".to_string(),
                test_cases: vec![
                    (vec!["0".to_string()], "1".to_string()),
                    (vec!["1".to_string()], "1".to_string()),
                    (vec!["5".to_string()], "120".to_string()),
                ],
                category: "math".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_7".to_string(),
                description: "Check if a number is prime".to_string(),
                function_signature: "def is_prime(n: int) -> bool:".to_string(),
                test_cases: vec![
                    (vec!["2".to_string()], "True".to_string()),
                    (vec!["4".to_string()], "False".to_string()),
                    (vec!["17".to_string()], "True".to_string()),
                ],
                category: "math".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_8".to_string(),
                description: "Sort a list in ascending order".to_string(),
                function_signature: "def sort_list(lst: list) -> list:".to_string(),
                test_cases: vec![
                    (vec!["[3, 1, 2]".to_string()], "[1, 2, 3]".to_string()),
                    (vec!["[5, 5, 5]".to_string()], "[5, 5, 5]".to_string()),
                    (vec!["[]".to_string()], "[]".to_string()),
                ],
                category: "list".to_string(),
            },
            HumanEvalProblem {
                id: "humaneval_9".to_string(),
                description: "Return the nth Fibonacci number".to_string(),
                function_signature: "def fibonacci(n: int) -> int:".to_string(),
                test_cases: vec![
                    (vec!["0".to_string()], "0".to_string()),
                    (vec!["1".to_string()], "1".to_string()),
                    (vec!["10".to_string()], "55".to_string()),
                ],
                category: "math".to_string(),
            },
        ]
    }

    /// Extract code from response
    fn extract_code(response: &str) -> String {
        // Try to extract code from markdown blocks with "python" language tag
        if let Some(start) = response.find("```python") {
            let after_start = start + "```python".len();
            if let Some(end) = response[after_start..].find("```") {
                return response[after_start..after_start + end].trim().to_string();
            }
        }
        // Try generic code block
        if let Some(start) = response.find("```") {
            let after_start = start + 3;
            // Skip language identifier if present
            let code_start = response[after_start..].find(|c: char| c == '\n')
                .map(|nl| after_start + nl + 1)
                .unwrap_or(after_start);
            if let Some(end) = response[code_start..].find("```") {
                return response[code_start..code_start + end].trim().to_string();
            }
        }

        // Fall back to the entire response
        response.trim().to_string()
    }

    /// Simple syntax check for generated code
    fn check_code_validity(code: &str, signature: &str) -> bool {
        // Check that the function signature is present or similar
        code.contains("def ") ||
        code.contains("return") ||
        code.contains(signature.split(':').next().unwrap_or(""))
    }
}

impl Default for HumanEvalBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for HumanEvalBenchmark {
    fn id(&self) -> &str {
        "humaneval"
    }

    fn name(&self) -> &str {
        "HumanEval"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Coding
    }

    fn description(&self) -> &str {
        "Code generation benchmark with function completion tasks"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples: Vec<DataSample> = self.problems.iter().map(|p| {
            let input = format!(
                "# {}\n{}\n# Test cases: {:?}\n# Write the function body",
                p.description, p.function_signature, p.test_cases
            );
            DataSample::new(&p.id, input)
                .with_expected(&p.function_signature)
                .with_category(&p.category)
        }).collect();

        Ok(Dataset::new("HumanEval", samples))
    }

    async fn run(
        &self,
        provider: &dyn models_core::providers::ModelProvider,
        config: BenchmarkConfig,
    ) -> models_core::error::Result<BenchmarkResult> {
        let dataset = self.load_dataset().await?;
        let mut samples = dataset.samples;

        if let Some(max) = config.max_samples {
            samples.truncate(max);
        }

        debug!("Running HumanEval benchmark with {} samples", samples.len());

        let mut results = Vec::new();
        let mut category_results: HashMap<String, Vec<SampleResult>> = HashMap::new();

        for (idx, sample) in samples.iter().enumerate() {
            let problem = &self.problems[idx];
            let prompt = self.format_prompt(sample, &[]);

            let request = models_core::providers::GenerateRequest::new(&prompt)
                .with_temperature(config.temperature)
                .with_max_tokens(config.max_tokens);

            let start = Instant::now();
            let result = provider.generate(request).await;
            let latency_ms = start.elapsed().as_millis() as u64;

            let sample_result = match result {
                Ok(response) => {
                    let code = Self::extract_code(&response.text);
                    let valid = Self::check_code_validity(&code, &problem.function_signature);
                    let score = if valid { 1.0 } else { 0.0 };

                    SampleResult {
                        sample_id: sample.id.clone(),
                        generated_output: code,
                        expected_output: sample.expected_output.clone(),
                        correct: valid,
                        score,
                        latency_ms,
                        error: None,
                    }
                }
                Err(e) => SampleResult::error(&sample.id, e.to_string()),
            };

            if let Some(ref category) = sample.category {
                category_results
                    .entry(category.clone())
                    .or_default()
                    .push(sample_result.clone());
            }

            results.push(sample_result);
        }

        let statistics = BenchmarkStatistics::from_results(&results);

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
        let code = Self::extract_code(response);
        let signature = sample.expected_output.as_deref().unwrap_or("");

        if Self::check_code_validity(&code, signature) {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, _few_shot_examples: &[DataSample]) -> String {
        format!(
            "Complete the following Python function. Only output the function body, no explanations.\n\n{}\n\n# Your implementation:",
            sample.input
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_humaneval_creation() {
        let benchmark = HumanEvalBenchmark::new();
        assert_eq!(benchmark.id(), "humaneval");
        assert_eq!(benchmark.category(), BenchmarkCategory::Coding);
    }

    #[test]
    fn test_extract_code_markdown() {
        let response = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```\nThat's it!";
        let code = HumanEvalBenchmark::extract_code(response);
        assert!(code.contains("def add"));
    }

    #[test]
    fn test_check_code_validity() {
        let code = "def add(a, b):\n    return a + b";
        let signature = "def add(a: int, b: int) -> int:";
        assert!(HumanEvalBenchmark::check_code_validity(code, signature));

        let invalid_code = "This is not code";
        assert!(!HumanEvalBenchmark::check_code_validity(invalid_code, signature));
    }
}