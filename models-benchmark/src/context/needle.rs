//! Needle in a Haystack Benchmark
//!
//! Tests long-context retrieval capabilities.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// Needle in a Haystack Benchmark implementation
pub struct NeedleInHaystackBenchmark {
    /// Sample contexts with hidden facts
    samples: Vec<NeedleSample>,
}

#[derive(Debug, Clone)]
struct NeedleSample {
    id: String,
    context: String,
    needle: String,  // The fact to find
    question: String,
    answer: String,
}

impl NeedleInHaystackBenchmark {
    pub fn new() -> Self {
        Self {
            samples: Self::generate_samples(),
        }
    }

    fn generate_samples() -> Vec<NeedleSample> {
        vec![
            NeedleSample {
                id: "needle_1".to_string(),
                context: "The quick brown fox jumps over the lazy dog. The secret code is ALPHA-7749. Remember this for later. The dog slept peacefully in the sun. Birds chirped in the trees nearby.".to_string(),
                needle: "ALPHA-7749".to_string(),
                question: "What is the secret code mentioned in the text?".to_string(),
                answer: "ALPHA-7749".to_string(),
            },
            NeedleSample {
                id: "needle_2".to_string(),
                context: "In 2023, the company reported record profits. The CEO's favorite number is 42. The quarterly revenue exceeded expectations by 15%. Market analysts were impressed by the growth trajectory.".to_string(),
                needle: "42".to_string(),
                question: "What is the CEO's favorite number?".to_string(),
                answer: "42".to_string(),
            },
            NeedleSample {
                id: "needle_3".to_string(),
                context: "The recipe calls for flour, sugar, and eggs. The secret ingredient that makes it special is Madagascar vanilla extract. Mix well and bake at 350 degrees for 30 minutes. Let cool before serving.".to_string(),
                needle: "Madagascar vanilla extract".to_string(),
                question: "What is the secret ingredient in the recipe?".to_string(),
                answer: "Madagascar vanilla extract".to_string(),
            },
            NeedleSample {
                id: "needle_4".to_string(),
                context: "The ancient temple was discovered in 1987. Inside, archaeologists found a golden statue weighing exactly 347 kilograms. The statue depicted a deity from a previously unknown civilization. Carbon dating suggested it was over 2000 years old.".to_string(),
                needle: "347 kilograms".to_string(),
                question: "How much did the golden statue weigh?".to_string(),
                answer: "347 kilograms".to_string(),
            },
            NeedleSample {
                id: "needle_5".to_string(),
                context: "The space mission launched on March 15th. The target destination was Kepler-442b. The journey would take approximately 112,000 years at current speeds. Scientists remained optimistic about future propulsion advances.".to_string(),
                needle: "Kepler-442b".to_string(),
                question: "What was the target destination of the space mission?".to_string(),
                answer: "Kepler-442b".to_string(),
            },
        ]
    }
}

impl Default for NeedleInHaystackBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for NeedleInHaystackBenchmark {
    fn id(&self) -> &str {
        "needle_in_haystack"
    }

    fn name(&self) -> &str {
        "Needle in a Haystack"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Context
    }

    fn description(&self) -> &str {
        "Tests the ability to find specific information within longer contexts"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples: Vec<DataSample> = self.samples.iter().map(|s| {
            DataSample::new(&s.id, format!("Context: {}\n\nQuestion: {}", s.context, s.question))
                .with_expected(&s.answer)
                .with_category("retrieval")
        }).collect();

        Ok(Dataset::new("NeedleInHaystack", samples))
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

        debug!("Running Needle in a Haystack benchmark with {} samples", samples.len());

        let mut results = Vec::new();
        let mut category_results: HashMap<String, Vec<SampleResult>> = HashMap::new();

        for sample in &samples {
            let prompt = self.format_prompt(sample, &[]);

            let request = models_core::providers::GenerateRequest::new(&prompt)
                .with_temperature(config.temperature)
                .with_max_tokens(config.max_tokens);

            let start = Instant::now();
            let result = provider.generate(request).await;
            let latency_ms = start.elapsed().as_millis() as u64;

            let sample_result = match result {
                Ok(response) => {
                    let score = self.evaluate_response(sample, &response.text);
                    let correct = score >= 0.9;

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
            Some(e) => e.to_lowercase(),
            None => return 0.0,
        };

        let response_lower = response.to_lowercase();

        if response_lower.contains(&expected) {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, _few_shot_examples: &[DataSample]) -> String {
        format!(
            "Read the following context and answer the question based only on the information provided.\n\n{}\n\nProvide a concise answer:",
            sample.input
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needle_creation() {
        let benchmark = NeedleInHaystackBenchmark::new();
        assert_eq!(benchmark.id(), "needle_in_haystack");
        assert_eq!(benchmark.category(), BenchmarkCategory::Context);
    }
}