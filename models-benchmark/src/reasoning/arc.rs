//! ARC (AI2 Reasoning Challenge) Benchmark
//!
//! A dataset of science questions designed to test reasoning ability.
//! Contains two splits: ARC-Easy and ARC-Challenge.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, Difficulty, SampleResult,
};

/// ARC variant (Easy or Challenge)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ARCVariant {
    /// ARC-Easy: Questions answered correctly by both retrieval and co-occurrence baselines
    #[default]
    Easy,
    /// ARC-Challenge: Questions that are difficult for baseline models
    Challenge,
}

impl ARCVariant {
    /// Get the display name for this variant
    pub fn display_name(&self) -> &'static str {
        match self {
            ARCVariant::Easy => "ARC-Easy",
            ARCVariant::Challenge => "ARC-Challenge",
        }
    }

    /// Get the variant from a string identifier
    pub fn from_id(id: &str) -> Option<Self> {
        match id.to_lowercase().as_str() {
            "arc-easy" | "easy" => Some(ARCVariant::Easy),
            "arc-challenge" | "challenge" => Some(ARCVariant::Challenge),
            _ => None,
        }
    }
}

/// ARC Benchmark implementation
pub struct ARCBenchmark {
    /// The variant (Easy or Challenge)
    variant: ARCVariant,
    /// Few-shot examples for prompting
    few_shot_examples: Vec<DataSample>,
}

impl ARCBenchmark {
    /// Create a new ARC benchmark with the specified variant
    pub fn new(variant: ARCVariant) -> Self {
        Self {
            variant,
            few_shot_examples: Self::default_few_shot_examples(),
        }
    }

    /// Create a new ARC-Easy benchmark
    pub fn easy() -> Self {
        Self::new(ARCVariant::Easy)
    }

    /// Create a new ARC-Challenge benchmark
    pub fn challenge() -> Self {
        Self::new(ARCVariant::Challenge)
    }

    /// Default 5-shot examples for ARC (science questions)
    fn default_few_shot_examples() -> Vec<DataSample> {
        vec![
            DataSample::new("fs1",
                "What is the process by which plants make their own food using sunlight?\nA. Respiration\nB. Photosynthesis\nC. Fermentation\nD. Digestion")
                .with_expected("B")
                .with_category("biology"),
            DataSample::new("fs2",
                "Which of the following is a conductor of electricity?\nA. Rubber\nB. Glass\nC. Copper\nD. Plastic")
                .with_expected("C")
                .with_category("physics"),
            DataSample::new("fs3",
                "What causes the seasons on Earth?\nA. The distance between Earth and the Sun\nB. The tilt of Earth's axis\nC. The speed of Earth's rotation\nD. The shape of Earth's orbit")
                .with_expected("B")
                .with_category("earth_science"),
            DataSample::new("fs4",
                "Which state of matter has a definite volume but no definite shape?\nA. Solid\nB. Liquid\nC. Gas\nD. Plasma")
                .with_expected("B")
                .with_category("chemistry"),
            DataSample::new("fs5",
                "What is the primary function of the heart in the human body?\nA. To filter blood\nB. To produce blood cells\nC. To pump blood throughout the body\nD. To store oxygen")
                .with_expected("C")
                .with_category("biology"),
        ]
    }

    /// Load sample ARC-Easy data
    fn load_easy_data() -> Vec<DataSample> {
        vec![
            DataSample::new("arc_easy_1",
                "What do plants need to grow?\nA. Rocks and sand\nB. Sunlight, water, and nutrients\nC. Plastic and metal\nD. Ice and snow")
                .with_expected("B")
                .with_category("biology")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_2",
                "Which of these is a source of light?\nA. A mirror\nB. The Moon\nC. A burning candle\nD. A shadow")
                .with_expected("C")
                .with_category("physics")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_3",
                "What is the largest planet in our solar system?\nA. Earth\nB. Mars\nC. Jupiter\nD. Saturn")
                .with_expected("C")
                .with_category("astronomy")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_4",
                "What happens to water when it freezes?\nA. It turns into steam\nB. It turns into ice\nC. It disappears\nD. It turns into a gas")
                .with_expected("B")
                .with_category("chemistry")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_5",
                "Which of the following is a mammal?\nA. A snake\nB. A fish\nC. A dog\nD. A bird")
                .with_expected("C")
                .with_category("biology")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_6",
                "What is the main gas in Earth's atmosphere?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Hydrogen")
                .with_expected("C")
                .with_category("earth_science")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_7",
                "What causes a rainbow to appear?\nA. Reflection of light off clouds\nB. Refraction and dispersion of light through water droplets\nC. Lightning in the sky\nD. Shadows from the sun")
                .with_expected("B")
                .with_category("physics")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_8",
                "Which body system is responsible for transporting blood throughout the body?\nA. Respiratory system\nB. Digestive system\nC. Circulatory system\nD. Nervous system")
                .with_expected("C")
                .with_category("biology")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_9",
                "What type of rock is formed from cooled magma or lava?\nA. Sedimentary\nB. Metamorphic\nC. Igneous\nD. Volcanic")
                .with_expected("C")
                .with_category("earth_science")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_easy_10",
                "What is the unit of electric current?\nA. Volt\nB. Ampere\nC. Ohm\nD. Watt")
                .with_expected("B")
                .with_category("physics")
                .with_difficulty(Difficulty::Easy),
        ]
    }

    /// Load sample ARC-Challenge data (more difficult questions requiring reasoning)
    fn load_challenge_data() -> Vec<DataSample> {
        vec![
            DataSample::new("arc_challenge_1",
                "A student wants to determine if the temperature of water affects how quickly salt dissolves. Which experimental design would best test this?\nA. Add the same amount of salt to different amounts of cold water\nB. Add different amounts of salt to the same amount of water at the same temperature\nC. Add the same amount of salt to water at different temperatures\nD. Add salt to water and stir at different speeds")
                .with_expected("C")
                .with_category("scientific_method")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_2",
                "In a food chain, what role do plants play?\nA. Consumers\nB. Producers\nC. Decomposers\nD. Predators")
                .with_expected("B")
                .with_category("ecology")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_3",
                "A ball is dropped from a height of 2 meters. If air resistance is negligible, approximately how fast will the ball be moving when it hits the ground? (Use g = 10 m/s^2)\nA. 2 m/s\nB. 4 m/s\nC. 6 m/s\nD. 8 m/s")
                .with_expected("C")
                .with_category("physics")
                .with_difficulty(Difficulty::Hard),
            DataSample::new("arc_challenge_4",
                "Why does a boat float while a small nail sinks in water?\nA. The boat is made of lighter materials\nB. The boat displaces more water relative to its weight\nC. The nail is denser than the boat\nD. The boat has air inside")
                .with_expected("B")
                .with_category("physics")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_5",
                "Which of the following would most likely cause a decrease in the population of rabbits in an ecosystem?\nA. An increase in the rabbit food supply\nB. An increase in the number of predators that eat rabbits\nC. A decrease in the number of competing species\nD. An increase in available water")
                .with_expected("B")
                .with_category("ecology")
                .with_difficulty(Difficulty::Easy),
            DataSample::new("arc_challenge_6",
                "In an electric circuit, what happens to the total resistance when resistors are connected in series?\nA. It decreases\nB. It stays the same\nC. It increases\nD. It becomes zero")
                .with_expected("C")
                .with_category("physics")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_7",
                "When a person exercises, their breathing rate increases primarily to:\nA. Increase oxygen intake and remove carbon dioxide\nB. Cool down the body\nC. Increase blood pressure\nD. Reduce water loss")
                .with_expected("A")
                .with_category("biology")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_8",
                "Which of the following best explains why the Moon appears to change shape (phases) over the course of a month?\nA. The Moon moves closer and farther from Earth\nB. Different portions of the Moon's illuminated half become visible from Earth\nC. The Moon's shape actually changes\nD. Clouds cover different parts of the Moon")
                .with_expected("B")
                .with_category("astronomy")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_9",
                "A scientist wants to test the effect of fertilizer on plant growth. What is the independent variable?\nA. The amount of water given to plants\nB. The type of soil used\nC. The amount of fertilizer added\nD. The height of the plants after growth")
                .with_expected("C")
                .with_category("scientific_method")
                .with_difficulty(Difficulty::Medium),
            DataSample::new("arc_challenge_10",
                "Which of these chemical equations is balanced correctly?\nA. H2 + O2 → H2O\nB. 2H2 + O2 → 2H2O\nC. H2 + O2 → 2H2O\nD. 2H2 + 2O2 → 2H2O")
                .with_expected("B")
                .with_category("chemistry")
                .with_difficulty(Difficulty::Hard),
        ]
    }

    /// Extract the answer letter from a response
    /// Handles formats like: "A", "(A)", "Answer: A", "The answer is A", etc.
    fn extract_answer(response: &str) -> Option<char> {
        let response_upper = response.trim().to_uppercase();

        // Single letter answer
        if response_upper.len() == 1 {
            let c = response_upper.chars().next()?;
            if c >= 'A' && c <= 'D' {
                return Some(c);
            }
        }

        // Pattern: "(A)" or "[A]"
        if let Some(first) = response_upper.chars().next() {
            if (first == '(' || first == '[') && response_upper.len() >= 3 {
                let second = response_upper.chars().nth(1)?;
                let third = response_upper.chars().nth(2)?;
                if second >= 'A' && second <= 'D' && (third == ')' || third == ']') {
                    return Some(second);
                }
            }
        }

        // Look for explicit answer patterns
        let patterns = [
            "ANSWER IS ",
            "ANSWER IS: ",
            "ANSWER: ",
            "ANSWER - ",
            "THE ANSWER IS ",
            "THE ANSWER IS: ",
            "THE ANSWER: ",
            "OPTION ",
            "CHOICE ",
            "SELECTED: ",
            "CORRECT ANSWER: ",
            "CORRECT ANSWER IS ",
        ];

        for pattern in &patterns {
            if let Some(pos) = response_upper.find(pattern) {
                let after = &response_upper[pos + pattern.len()..];
                // Look for the first letter A-D in the remainder
                for c in after.chars() {
                    if c >= 'A' && c <= 'D' {
                        return Some(c);
                    }
                }
            }
        }

        // Look for patterns like "A.", "A)", "A " at the start of the response
        if response_upper.len() >= 2 {
            let first_char = response_upper.chars().next()?;
            let second_char = response_upper.chars().nth(1)?;
            if first_char >= 'A' && first_char <= 'D' {
                if second_char == '.' || second_char == ')' || second_char == ' ' {
                    return Some(first_char);
                }
            }
        }

        // Last resort: find the first occurrence of A-D
        for c in response_upper.chars() {
            if c >= 'A' && c <= 'D' {
                return Some(c);
            }
        }

        None
    }
}

impl Default for ARCBenchmark {
    fn default() -> Self {
        Self::new(ARCVariant::Easy)
    }
}

#[async_trait]
impl Benchmark for ARCBenchmark {
    fn id(&self) -> &str {
        match self.variant {
            ARCVariant::Easy => "arc-easy",
            ARCVariant::Challenge => "arc-challenge",
        }
    }

    fn name(&self) -> &str {
        match self.variant {
            ARCVariant::Easy => "ARC-Easy (AI2 Reasoning Challenge)",
            ARCVariant::Challenge => "ARC-Challenge (AI2 Reasoning Challenge)",
        }
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn description(&self) -> &str {
        match self.variant {
            ARCVariant::Easy => {
                "ARC-Easy: Science questions answerable by retrieval-based or word co-occurrence methods"
            }
            ARCVariant::Challenge => {
                "ARC-Challenge: Science questions requiring reasoning beyond simple retrieval"
            }
        }
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples = match self.variant {
            ARCVariant::Easy => Self::load_easy_data(),
            ARCVariant::Challenge => Self::load_challenge_data(),
        };

        let mut dataset = Dataset::new(self.variant.display_name(), samples);
        dataset.metadata.description = Some(format!(
            "{} - Science question answering benchmark",
            self.variant.display_name()
        ));
        dataset.metadata.source_url = Some(
            "https://allenai.org/data/arc".to_string()
        );

        Ok(dataset)
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

        debug!(
            "Running {} benchmark with {} samples",
            self.variant.display_name(),
            samples.len()
        );

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

        // Add instructions
        prompt.push_str("Answer the following multiple choice science question by selecting the correct letter (A, B, C, or D).\n\n");

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
    fn test_arc_creation() {
        let easy = ARCBenchmark::easy();
        assert_eq!(easy.id(), "arc-easy");
        assert_eq!(easy.category(), BenchmarkCategory::Reasoning);

        let challenge = ARCBenchmark::challenge();
        assert_eq!(challenge.id(), "arc-challenge");
        assert_eq!(challenge.category(), BenchmarkCategory::Reasoning);
    }

    #[test]
    fn test_variant_from_id() {
        assert_eq!(ARCVariant::from_id("easy"), Some(ARCVariant::Easy));
        assert_eq!(ARCVariant::from_id("arc-easy"), Some(ARCVariant::Easy));
        assert_eq!(ARCVariant::from_id("challenge"), Some(ARCVariant::Challenge));
        assert_eq!(ARCVariant::from_id("arc-challenge"), Some(ARCVariant::Challenge));
        assert_eq!(ARCVariant::from_id("unknown"), None);
    }

    #[test]
    fn test_extract_answer_simple() {
        assert_eq!(ARCBenchmark::extract_answer("A"), Some('A'));
        assert_eq!(ARCBenchmark::extract_answer("B"), Some('B'));
        assert_eq!(ARCBenchmark::extract_answer("C"), Some('C'));
        assert_eq!(ARCBenchmark::extract_answer("D"), Some('D'));
    }

    #[test]
    fn test_extract_answer_parentheses() {
        assert_eq!(ARCBenchmark::extract_answer("(A)"), Some('A'));
        assert_eq!(ARCBenchmark::extract_answer("[B]"), Some('B'));
        assert_eq!(ARCBenchmark::extract_answer("(C)"), Some('C'));
        assert_eq!(ARCBenchmark::extract_answer("[D]"), Some('D'));
    }

    #[test]
    fn test_extract_answer_patterns() {
        assert_eq!(ARCBenchmark::extract_answer("The answer is A"), Some('A'));
        assert_eq!(ARCBenchmark::extract_answer("Answer: B"), Some('B'));
        assert_eq!(ARCBenchmark::extract_answer("Answer is C"), Some('C'));
        assert_eq!(ARCBenchmark::extract_answer("The correct answer is D"), Some('D'));
        assert_eq!(ARCBenchmark::extract_answer("Correct answer: A"), Some('A'));
    }

    #[test]
    fn test_extract_answer_with_punctuation() {
        assert_eq!(ARCBenchmark::extract_answer("A."), Some('A'));
        assert_eq!(ARCBenchmark::extract_answer("B)"), Some('B'));
        assert_eq!(ARCBenchmark::extract_answer("C "), Some('C'));
    }

    #[test]
    fn test_extract_answer_case_insensitive() {
        assert_eq!(ARCBenchmark::extract_answer("a"), Some('A'));
        assert_eq!(ARCBenchmark::extract_answer("the answer is b"), Some('B'));
        assert_eq!(ARCBenchmark::extract_answer("ANSWER: C"), Some('C'));
    }

    #[test]
    fn test_format_prompt() {
        let benchmark = ARCBenchmark::easy();
        let sample = DataSample::new("test",
            "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid")
            .with_expected("B");

        let prompt = benchmark.format_prompt(&sample, &[]);
        assert!(prompt.contains("What is the capital of France?"));
        assert!(prompt.contains("Answer:"));
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = ARCBenchmark::easy();
        let sample = DataSample::new("test",
            "Question?\nA. X\nB. Y\nC. Z\nD. W")
            .with_expected("B");

        assert_eq!(benchmark.evaluate_response(&sample, "B"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "A"), 0.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is B"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "(B)"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "Answer: C"), 0.0);
    }

    #[test]
    fn test_load_dataset() {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let easy = ARCBenchmark::easy();
        let dataset_easy = rt.block_on(easy.load_dataset()).unwrap();
        assert!(!dataset_easy.is_empty());
        assert_eq!(dataset_easy.name, "ARC-Easy");

        let challenge = ARCBenchmark::challenge();
        let dataset_challenge = rt.block_on(challenge.load_dataset()).unwrap();
        assert!(!dataset_challenge.is_empty());
        assert_eq!(dataset_challenge.name, "ARC-Challenge");
    }
}
