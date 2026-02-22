//! AlpacaEval Benchmark
//!
//! Tests instruction following capabilities.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// AlpacaEval Benchmark implementation
pub struct AlpacaEvalBenchmark {
    samples: Vec<InstructionSample>,
}

#[derive(Debug, Clone)]
struct InstructionSample {
    id: String,
    instruction: String,
    category: String,
}

impl AlpacaEvalBenchmark {
    pub fn new() -> Self {
        Self {
            samples: Self::load_samples(),
        }
    }

    fn load_samples() -> Vec<InstructionSample> {
        vec![
            InstructionSample {
                id: "alpaca_1".to_string(),
                instruction: "Write a haiku about programming.".to_string(),
                category: "creative".to_string(),
            },
            InstructionSample {
                id: "alpaca_2".to_string(),
                instruction: "Explain quantum computing in simple terms for a 10-year-old.".to_string(),
                category: "explanation".to_string(),
            },
            InstructionSample {
                id: "alpaca_3".to_string(),
                instruction: "List 5 tips for staying productive while working from home.".to_string(),
                category: "list".to_string(),
            },
            InstructionSample {
                id: "alpaca_4".to_string(),
                instruction: "Convert the following to a formal email: 'Hey, can we meet tomorrow at 3? Got some stuff to discuss.'".to_string(),
                category: "rewriting".to_string(),
            },
            InstructionSample {
                id: "alpaca_5".to_string(),
                instruction: "Write a Python function that reverses a string.".to_string(),
                category: "coding".to_string(),
            },
            InstructionSample {
                id: "alpaca_6".to_string(),
                instruction: "Summarize the following in one sentence: 'The Industrial Revolution was a period of major industrialization that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout the world.'".to_string(),
                category: "summarization".to_string(),
            },
            InstructionSample {
                id: "alpaca_7".to_string(),
                instruction: "What are the main differences between a virus and a bacterium?".to_string(),
                category: "explanation".to_string(),
            },
            InstructionSample {
                id: "alpaca_8".to_string(),
                instruction: "Write a short story about a robot learning to paint.".to_string(),
                category: "creative".to_string(),
            },
            InstructionSample {
                id: "alpaca_9".to_string(),
                instruction: "Provide three pros and three cons of social media.".to_string(),
                category: "list".to_string(),
            },
            InstructionSample {
                id: "alpaca_10".to_string(),
                instruction: "Explain the concept of compound interest with an example.".to_string(),
                category: "explanation".to_string(),
            },
        ]
    }

    /// Check if the response follows the instruction (basic heuristic)
    fn evaluate_instruction_following(instruction: &str, response: &str) -> f64 {
        let response = response.trim();

        // Check for minimum length (non-empty response)
        if response.len() < 10 {
            return 0.0;
        }

        // Check for common issues
        if response.contains("I cannot") || response.contains("I'm unable to") {
            return 0.3; // Partial credit for attempted refusal
        }

        // Check for task-specific completion
        let instruction_lower = instruction.to_lowercase();

        if instruction_lower.contains("list") && instruction_lower.contains("tips") {
            // Count bullet points or numbered items
            let items = response.lines().filter(|l| {
                l.trim().starts_with('-') ||
                l.trim().starts_with('*') ||
                l.trim().starts_with(|c: char| c.is_ascii_digit())
            }).count();
            if items >= 3 { return 1.0; }
            if items >= 1 { return 0.6; }
        }

        if instruction_lower.contains("haiku") {
            // Check for 3-line structure
            let lines: Vec<&str> = response.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.len() == 3 { return 1.0; }
            if lines.len() >= 2 { return 0.7; }
        }

        if instruction_lower.contains("python") && instruction_lower.contains("function") {
            if response.contains("def ") && response.contains("return") {
                return 1.0;
            }
            if response.contains("def ") {
                return 0.7;
            }
        }

        if instruction_lower.contains("one sentence") {
            let sentences = response.matches('.').count();
            if sentences == 1 || (sentences == 0 && response.len() > 20) {
                return 1.0;
            }
            if sentences <= 2 { return 0.7; }
        }

        if instruction_lower.contains("pros") && instruction_lower.contains("cons") {
            if response.to_lowercase().contains("pros") && response.to_lowercase().contains("cons") {
                return 1.0;
            }
        }

        // Default: gave a substantial response
        if response.len() > 50 {
            0.8
        } else if response.len() > 20 {
            0.5
        } else {
            0.3
        }
    }
}

impl Default for AlpacaEvalBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for AlpacaEvalBenchmark {
    fn id(&self) -> &str {
        "alpaca_eval"
    }

    fn name(&self) -> &str {
        "AlpacaEval"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Instruction
    }

    fn description(&self) -> &str {
        "Tests instruction following capabilities across diverse tasks"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples: Vec<DataSample> = self.samples.iter().map(|s| {
            DataSample::new(&s.id, &s.instruction)
                .with_category(&s.category)
        }).collect();

        Ok(Dataset::new("AlpacaEval", samples))
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

        debug!("Running AlpacaEval benchmark with {} samples", samples.len());

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
                    let correct = score >= 0.7;

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
        Self::evaluate_instruction_following(&sample.input, response)
    }

    fn format_prompt(&self, sample: &DataSample, _few_shot_examples: &[DataSample]) -> String {
        sample.input.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpaca_creation() {
        let benchmark = AlpacaEvalBenchmark::new();
        assert_eq!(benchmark.id(), "alpaca_eval");
        assert_eq!(benchmark.category(), BenchmarkCategory::Instruction);
    }

    #[test]
    fn test_haiku_evaluation() {
        let score = AlpacaEvalBenchmark::evaluate_instruction_following(
            "Write a haiku about programming",
            "Code flows like water\nBugs emerge from the shadows\nVictory at last"
        );
        assert!(score >= 0.7);
    }

    #[test]
    fn test_list_evaluation() {
        let score = AlpacaEvalBenchmark::evaluate_instruction_following(
            "List 5 tips for productivity",
            "Here are 5 tips:\n1. Wake up early\n2. Exercise\n3. Plan your day\n4. Take breaks\n5. Stay hydrated"
        );
        assert!(score >= 0.9);
    }
}