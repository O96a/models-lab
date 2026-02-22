//! MT-Bench Benchmark
//!
//! Tests multi-turn conversation capabilities.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, SampleResult,
};

/// MT-Bench Benchmark implementation
pub struct MTBenchBenchmark {
    conversations: Vec<MultiTurnConversation>,
}

#[derive(Debug, Clone)]
struct MultiTurnConversation {
    id: String,
    turns: Vec<(String, String)>, // (question, expected_topic)
    category: String,
}

impl MTBenchBenchmark {
    pub fn new() -> Self {
        Self {
            conversations: Self::load_conversations(),
        }
    }

    fn load_conversations() -> Vec<MultiTurnConversation> {
        vec![
            MultiTurnConversation {
                id: "mtbench_1".to_string(),
                turns: vec![
                    ("What is machine learning?".to_string(), "machine learning".to_string()),
                    ("What are some common algorithms used in it?".to_string(), "algorithms".to_string()),
                    ("How does it differ from deep learning?".to_string(), "deep learning".to_string()),
                ],
                category: "technology".to_string(),
            },
            MultiTurnConversation {
                id: "mtbench_2".to_string(),
                turns: vec![
                    ("I'm planning a trip to Japan. What should I visit?".to_string(), "Japan travel".to_string()),
                    ("What about food recommendations in Tokyo?".to_string(), "Tokyo food".to_string()),
                    ("How much should I budget for a week?".to_string(), "budget".to_string()),
                ],
                category: "travel".to_string(),
            },
            MultiTurnConversation {
                id: "mtbench_3".to_string(),
                turns: vec![
                    ("Can you help me write a resume?".to_string(), "resume".to_string()),
                    ("What skills should I highlight for a software engineer position?".to_string(), "skills".to_string()),
                    ("How do I format my work experience section?".to_string(), "work experience".to_string()),
                ],
                category: "career".to_string(),
            },
            MultiTurnConversation {
                id: "mtbench_4".to_string(),
                turns: vec![
                    ("Explain the water cycle to me.".to_string(), "water cycle".to_string()),
                    ("How does climate change affect it?".to_string(), "climate change".to_string()),
                    ("What can individuals do to help?".to_string(), "individual action".to_string()),
                ],
                category: "science".to_string(),
            },
            MultiTurnConversation {
                id: "mtbench_5".to_string(),
                turns: vec![
                    ("I want to learn guitar. Where should I start?".to_string(), "guitar basics".to_string()),
                    ("What chords should I learn first?".to_string(), "chords".to_string()),
                    ("How long does it typically take to become proficient?".to_string(), "timeline".to_string()),
                ],
                category: "music".to_string(),
            },
        ]
    }

    /// Check if response is relevant to the question and maintains context
    fn evaluate_response_relevance(question: &str, response: &str, expected_topic: &str) -> f64 {
        let response_lower = response.to_lowercase();
        let expected_lower = expected_topic.to_lowercase();

        // Check if response mentions expected topic
        if response_lower.contains(&expected_lower) {
            return 1.0;
        }

        // Check for related terms (simplified)
        let question_lower = question.to_lowercase();
        let question_words: Vec<&str> = question_lower.split_whitespace().collect();
        let response_words: Vec<&str> = response_lower.split_whitespace().collect();

        // Count word overlap
        let overlap = question_words.iter()
            .filter(|w| response_words.contains(w))
            .count();

        let relevance = overlap as f64 / question_words.len().max(1) as f64;

        // Check for minimum response length
        if response.len() < 20 {
            return 0.0;
        }

        if relevance > 0.3 {
            0.8
        } else if response.len() > 50 {
            0.5
        } else {
            0.3
        }
    }
}

impl Default for MTBenchBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for MTBenchBenchmark {
    fn id(&self) -> &str {
        "mt_bench"
    }

    fn name(&self) -> &str {
        "MT-Bench (Multi-turn Benchmark)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Multiturn
    }

    fn description(&self) -> &str {
        "Tests multi-turn conversation and context maintenance abilities"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples: Vec<DataSample> = self.conversations.iter().map(|c| {
            let input = c.turns.iter()
                .map(|(q, _)| q.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            DataSample::new(&c.id, input)
                .with_category(&c.category)
        }).collect();

        Ok(Dataset::new("MT-Bench", samples))
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

        debug!("Running MT-Bench benchmark with {} samples", samples.len());

        let mut results = Vec::new();
        let mut category_results: HashMap<String, Vec<SampleResult>> = HashMap::new();

        for (idx, sample) in samples.iter().enumerate() {
            let conversation = &self.conversations[idx];
            let mut total_score = 0.0;
            let mut turn_outputs = Vec::new();
            let mut context = String::new();
            let mut total_latency = 0u64;

            // Run each turn of the conversation
            for (question, expected_topic) in &conversation.turns {
                let prompt = format!("{}\n\nUser: {}", context, question);
                let request = models_core::providers::GenerateRequest::new(&prompt)
                    .with_temperature(config.temperature)
                    .with_max_tokens(config.max_tokens);

                let start = Instant::now();
                let result = provider.generate(request).await;
                let latency_ms = start.elapsed().as_millis() as u64;
                total_latency += latency_ms;

                match result {
                    Ok(response) => {
                        let score = Self::evaluate_response_relevance(question, &response.text, expected_topic);
                        total_score += score;
                        turn_outputs.push(format!("Q: {}\nA: {}", question, response.text));
                        context = format!("{}\nUser: {}\nAssistant: {}", context, question, response.text);
                    }
                    Err(_) => {
                        total_score += 0.0;
                        turn_outputs.push(format!("Q: {}\nA: [Error]", question));
                    }
                }
            }

            let avg_score = total_score / conversation.turns.len() as f64;

            let sample_result = SampleResult {
                sample_id: sample.id.clone(),
                generated_output: turn_outputs.join("\n\n"),
                expected_output: None,
                correct: avg_score >= 0.7,
                score: avg_score,
                latency_ms: total_latency,
                error: None,
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

    fn evaluate_response(&self, _sample: &DataSample, _response: &str) -> f64 {
        // Multi-turn evaluation is done during run()
        0.0
    }

    fn format_prompt(&self, sample: &DataSample, _few_shot_examples: &[DataSample]) -> String {
        sample.input.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtbench_creation() {
        let benchmark = MTBenchBenchmark::new();
        assert_eq!(benchmark.id(), "mt_bench");
        assert_eq!(benchmark.category(), BenchmarkCategory::Multiturn);
    }

    #[test]
    fn test_relevance_evaluation() {
        let score = MTBenchBenchmark::evaluate_response_relevance(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "machine learning"
        );
        assert!(score >= 0.9);
    }
}