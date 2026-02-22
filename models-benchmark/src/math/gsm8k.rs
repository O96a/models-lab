//! GSM8K (Grade School Math 8K) Benchmark
//!
//! Tests multi-step mathematical reasoning with word problems.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// GSM8K Benchmark implementation
pub struct GSM8KBenchmark {
    /// Few-shot examples for prompting
    few_shot_examples: Vec<DataSample>,
}

impl GSM8KBenchmark {
    /// Create a new GSM8K benchmark
    pub fn new() -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
        }
    }

    /// Default few-shot examples for GSM8K
    fn default_few_shot_examples() -> Vec<DataSample> {
        vec![
            DataSample::new("gsm8k_fs1",
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")
                .with_expected("18")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_fs2",
                "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?")
                .with_expected("3")
                .with_category("arithmetic"),
        ]
    }

    /// Load sample GSM8K data (for testing)
    fn load_sample_data() -> Vec<DataSample> {
        vec![
            DataSample::new("gsm8k_1",
                "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
                .with_expected("72")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_2",
                "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?")
                .with_expected("10")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_3",
                "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?")
                .with_expected("5")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_4",
                "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?")
                .with_expected("42")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_5",
                "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers are there in Mark's garden altogether?")
                .with_expected("35")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_6",
                "A baker made 125 pounds of sugar. He used 20 pounds of sugar on Monday and 35 pounds of sugar on Tuesday. How many pounds of sugar does the baker have left?")
                .with_expected("70")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_7",
                "Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did he finish the third part?")
                .with_expected("45")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_8",
                "A museum has 150 paintings. On Monday, 28 paintings were moved to storage. On Tuesday, 15 more paintings were moved to storage. How many paintings are still on display?")
                .with_expected("107")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_9",
                "Liza's sweater cost $45. She bought a pair of shoes that cost twice as much as the sweater. How much did Liza spend on the sweater and shoes together?")
                .with_expected("135")
                .with_category("arithmetic"),
            DataSample::new("gsm8k_10",
                "A teacher has 35 students in her class. She gives each student 3 pencils. How many pencils does she give out in total?")
                .with_expected("105")
                .with_category("arithmetic"),
        ]
    }

    /// Extract the numeric answer from a response
    fn extract_number(response: &str) -> Option<f64> {
        // Look for patterns like "The answer is 42" or "#### 42" (GSM8K format)
        let response = response.trim();

        // Try to find "#### <number>" format
        if let Some(pos) = response.find("####") {
            let after = &response[pos + 4..];
            if let Some(num) = Self::parse_first_number(after) {
                return Some(num);
            }
        }

        // Try to find "answer is" patterns
        for pattern in &["answer is", "the answer is", "answer:", "result is", "total is"] {
            if let Some(pos) = response.to_lowercase().find(pattern) {
                let after = &response[pos + pattern.len()..];
                if let Some(num) = Self::parse_first_number(after) {
                    return Some(num);
                }
            }
        }

        // Try to find a dollar amount
        if response.contains('$') {
            let response_no_dollar: String = response.chars().filter(|c| *c != '$').collect();
            if let Some(num) = Self::parse_first_number(&response_no_dollar) {
                return Some(num);
            }
        }

        // Fall back to the last number in the response
        Self::find_last_number(response)
    }

    /// Parse the first number found in a string
    fn parse_first_number(s: &str) -> Option<f64> {
        let s = s.trim();
        let mut num_str = String::new();
        let mut found_digit = false;
        let mut found_decimal = false;

        for c in s.chars() {
            if c.is_ascii_digit() {
                found_digit = true;
                num_str.push(c);
            } else if c == '.' && found_digit && !found_decimal {
                found_decimal = true;
                num_str.push(c);
            } else if c == '-' && num_str.is_empty() {
                num_str.push(c);
            } else if found_digit {
                break;
            }
        }

        if num_str.is_empty() || !found_digit {
            return None;
        }
        num_str.parse().ok()
    }

    /// Find the last number in a string
    fn find_last_number(s: &str) -> Option<f64> {
        let mut last_number: Option<f64> = None;
        let mut current_num = String::new();

        for c in s.chars() {
            if c.is_ascii_digit() || c == '.' || (c == '-' && current_num.is_empty()) {
                current_num.push(c);
            } else if !current_num.is_empty() {
                if let Ok(num) = current_num.parse() {
                    last_number = Some(num);
                }
                current_num.clear();
            }
        }

        // Check the last number
        if !current_num.is_empty() {
            if let Ok(num) = current_num.parse() {
                last_number = Some(num);
            }
        }

        last_number
    }
}

impl Default for GSM8KBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for GSM8KBenchmark {
    fn id(&self) -> &str {
        "gsm8k"
    }

    fn name(&self) -> &str {
        "GSM8K (Grade School Math 8K)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Math
    }

    fn description(&self) -> &str {
        "Multi-step arithmetic reasoning with grade school math word problems"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples = Self::load_sample_data();
        Ok(Dataset::new("GSM8K", samples))
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

        debug!("Running GSM8K benchmark with {} samples", samples.len());

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
        let expected = match &sample.expected_output {
            Some(e) => e.trim(),
            None => return 0.0,
        };

        let expected_num: f64 = match expected.parse() {
            Ok(n) => n,
            Err(_) => return 0.0,
        };

        let extracted_num = match Self::extract_number(response) {
            Some(n) => n,
            None => return 0.0,
        };

        // Allow for small floating point differences
        if (extracted_num - expected_num).abs() < 0.01 {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, few_shot_examples: &[DataSample]) -> String {
        let mut prompt = String::new();

        prompt.push_str("Solve the following math problem step by step. Show your work and give the final answer as a number.\n\n");

        for example in few_shot_examples {
            prompt.push_str(&format!(
                "Question: {}\nAnswer: {}\n\n",
                example.input,
                example.expected_output.as_deref().unwrap_or("")
            ));
        }

        prompt.push_str(&format!("Question: {}\nAnswer:", sample.input));

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsm8k_creation() {
        let benchmark = GSM8KBenchmark::new();
        assert_eq!(benchmark.id(), "gsm8k");
        assert_eq!(benchmark.category(), BenchmarkCategory::Math);
    }

    #[test]
    fn test_extract_number_simple() {
        assert_eq!(GSM8KBenchmark::extract_number("The answer is 42"), Some(42.0));
        assert_eq!(GSM8KBenchmark::extract_number("18"), Some(18.0));
        assert_eq!(GSM8KBenchmark::extract_number("#### 72"), Some(72.0));
    }

    #[test]
    fn test_extract_number_dollar() {
        assert_eq!(GSM8KBenchmark::extract_number("$135"), Some(135.0));
        assert_eq!(GSM8KBenchmark::extract_number("The total is $10.50"), Some(10.5));
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = GSM8KBenchmark::new();
        let sample = DataSample::new("test", "What is 2 + 2?").with_expected("4");

        assert_eq!(benchmark.evaluate_response(&sample, "The answer is 4"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is 5"), 0.0);
        assert_eq!(benchmark.evaluate_response(&sample, "4"), 1.0);
    }
}