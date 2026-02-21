//! MMLU (Massive Multitask Language Understanding) Benchmark

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// MMLU Benchmark implementation
pub struct MMLUBenchmark {
    /// Few-shot examples for prompting
    few_shot_examples: Vec<DataSample>,
}

impl MMLUBenchmark {
    /// Create a new MMLU benchmark
    pub fn new() -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
        }
    }

    /// Default 5-shot examples for MMLU
    fn default_few_shot_examples() -> Vec<DataSample> {
        vec![
            DataSample::new("fs1", "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid")
                .with_expected("B")
                .with_category("geography"),
            DataSample::new("fs2", "Which planet is closest to the Sun?\nA. Venus\nB. Mercury\nC. Mars\nD. Earth")
                .with_expected("B")
                .with_category("astronomy"),
            DataSample::new("fs3", "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6")
                .with_expected("B")
                .with_category("math"),
            DataSample::new("fs4", "Who wrote Romeo and Juliet?\nA. Dickens\nB. Shakespeare\nC. Austen\nD. Twain")
                .with_expected("B")
                .with_category("literature"),
            DataSample::new("fs5", "What is the chemical symbol for water?\nA. O2\nB. CO2\nC. H2O\nD. NaCl")
                .with_expected("C")
                .with_category("chemistry"),
        ]
    }

    /// Load sample MMLU data (for testing)
    fn load_sample_data() -> Vec<DataSample> {
        vec![
            DataSample::new("mmlu_abstract_algebra_1",
                "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 1\nB. 2\nC. 3\nD. 4")
                .with_expected("D")
                .with_category("abstract_algebra"),
            DataSample::new("mmlu_abstract_algebra_2",
                "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the order of p.\nA. 2\nB. 4\nC. 6\nD. 8")
                .with_expected("C")
                .with_category("abstract_algebra"),
            DataSample::new("mmlu_anatomy_1",
                "What is the main function of the mitochondria?\nA. Protein synthesis\nB. Energy production\nC. Cell division\nD. Waste removal")
                .with_expected("B")
                .with_category("anatomy"),
            DataSample::new("mmlu_anatomy_2",
                "Which of the following is NOT a type of white blood cell?\nA. Neutrophil\nB. Erythrocyte\nC. Lymphocyte\nD. Monocyte")
                .with_expected("B")
                .with_category("anatomy"),
            DataSample::new("mmlu_astronomy_1",
                "What is the approximate age of the universe?\nA. 4.6 billion years\nB. 10 billion years\nC. 13.8 billion years\nD. 20 billion years")
                .with_expected("C")
                .with_category("astronomy"),
            DataSample::new("mmlu_astronomy_2",
                "Which type of star is the Sun?\nA. Red dwarf\nB. Yellow dwarf\nC. Blue giant\nD. White dwarf")
                .with_expected("B")
                .with_category("astronomy"),
            DataSample::new("mmlu_business_ethics_1",
                "Which ethical theory focuses on the consequences of actions?\nA. Deontology\nB. Virtue ethics\nC. Utilitarianism\nD. Contractarianism")
                .with_expected("C")
                .with_category("business_ethics"),
            DataSample::new("mmlu_clinical_knowledge_1",
                "What is the most common cause of community-acquired pneumonia?\nA. Staphylococcus aureus\nB. Streptococcus pneumoniae\nC. Haemophilus influenzae\nD. Mycoplasma pneumoniae")
                .with_expected("B")
                .with_category("clinical_knowledge"),
            DataSample::new("mmlu_college_biology_1",
                "What molecule carries genetic information in cells?\nA. RNA\nB. DNA\nC. Protein\nD. Lipid")
                .with_expected("B")
                .with_category("college_biology"),
            DataSample::new("mmlu_college_chemistry_1",
                "What is the pH of a neutral solution at 25°C?\nA. 0\nB. 7\nC. 14\nD. 1")
                .with_expected("B")
                .with_category("college_chemistry"),
        ]
    }

    /// Extract the answer letter from a response
    fn extract_answer(response: &str) -> Option<char> {
        let response = response.trim().to_uppercase();

        // Try to find a single letter answer
        if response.len() == 1 {
            let c = response.chars().next()?;
            if c >= 'A' && c <= 'D' {
                return Some(c);
            }
        }

        // Try to find patterns like "A.", "A)", "(A)", etc.
        for pattern in &["ANSWER IS", "ANSWER:", "OPTION", "CHOICE"] {
            if let Some(pos) = response.find(pattern) {
                let after = &response[pos + pattern.len()..];
                for c in after.chars() {
                    if c >= 'A' && c <= 'D' {
                        return Some(c);
                    }
                }
            }
        }

        // Look for the first letter A-D
        for c in response.chars() {
            if c >= 'A' && c <= 'D' {
                return Some(c);
            }
        }

        None
    }
}

impl Default for MMLUBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for MMLUBenchmark {
    fn id(&self) -> &str {
        "mmlu"
    }

    fn name(&self) -> &str {
        "MMLU (Massive Multitask Language Understanding)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn description(&self) -> &str {
        "A benchmark measuring knowledge across 57 subjects including STEM, humanities, and social sciences"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        // In a real implementation, this would load from files
        // For now, we use sample data
        let samples = Self::load_sample_data();
        Ok(Dataset::new("MMLU", samples))
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

        debug!("Running MMLU benchmark with {} samples", samples.len());

        let mut results = Vec::new();
        let mut category_results: HashMap<String, Vec<SampleResult>> = HashMap::new();

        for sample in &samples {
            let prompt = self.format_prompt(sample, &self.few_shot_examples[..config.num_few_shot.min(5)]);
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
            Some(e) => e.trim().to_uppercase(),
            None => return 0.0,
        };

        let extracted = match Self::extract_answer(response) {
            Some(c) => c.to_string(),
            None => return 0.0,
        };

        if extracted == expected {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, few_shot_examples: &[DataSample]) -> String {
        let mut prompt = String::new();

        // Add few-shot examples
        for example in few_shot_examples {
            prompt.push_str(&format!(
                "Question: {}\nAnswer: {}\n\n",
                example.input,
                example.expected_output.as_deref().unwrap_or("")
            ));
        }

        // Add the actual question
        prompt.push_str(&format!("Question: {}\nAnswer:", sample.input));

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmlu_creation() {
        let benchmark = MMLUBenchmark::new();
        assert_eq!(benchmark.id(), "mmlu");
        assert_eq!(benchmark.category(), BenchmarkCategory::Reasoning);
    }

    #[test]
    fn test_extract_answer_simple() {
        assert_eq!(MMLUBenchmark::extract_answer("A"), Some('A'));
        assert_eq!(MMLUBenchmark::extract_answer("B"), Some('B'));
        assert_eq!(MMLUBenchmark::extract_answer("C"), Some('C'));
        assert_eq!(MMLUBenchmark::extract_answer("D"), Some('D'));
    }

    #[test]
    fn test_extract_answer_complex() {
        assert_eq!(MMLUBenchmark::extract_answer("The answer is A."), Some('A'));
        assert_eq!(MMLUBenchmark::extract_answer("I choose option B"), Some('B'));
        assert_eq!(MMLUBenchmark::extract_answer("Answer: C"), Some('C'));
    }

    #[test]
    fn test_format_prompt() {
        let benchmark = MMLUBenchmark::new();
        let sample = DataSample::new("test", "What is 1+1?\nA. 1\nB. 2\nC. 3\nD. 4")
            .with_expected("B");

        let prompt = benchmark.format_prompt(&sample, &[]);
        assert!(prompt.contains("What is 1+1?"));
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = MMLUBenchmark::new();
        let sample = DataSample::new("test", "Question?\nA. X\nB. Y\nC. Z\nD. W")
            .with_expected("B");

        assert_eq!(benchmark.evaluate_response(&sample, "B"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "A"), 0.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is B"), 1.0);
    }
}
