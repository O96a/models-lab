//! BBH (Big Bench Hard) Benchmark
//!
//! BBH is a challenging reasoning benchmark with 23 tasks from Big Bench
//! that require complex multi-step reasoning.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, Difficulty, SampleResult,
};

/// BBH subtasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BbhTask {
    BooleanExpressions,
    CausalJudgment,
    DateUnderstanding,
    DisambiguationQa,
    FormalFallacies,
    GeometricShapes,
    Hyperbaton,
    LogicalDeductionFiveObjects,
    LogicalDeductionSevenObjects,
    LogicalDeductionThreeObjects,
    MovieRecommendation,
    Navigate,
    PenguinsInATable,
    ReasoningAboutColoredObjects,
    RuinNames,
    SalientTranslationErrorDetection,
    Snarks,
    SportsUnderstanding,
    TemporalSequences,
    TrackingShuffledObjectsFiveObjects,
    TrackingShuffledObjectsSevenObjects,
    TrackingShuffledObjectsThreeObjects,
    WebOfLies,
}

impl BbhTask {
    /// Get all available tasks
    pub fn all() -> Vec<BbhTask> {
        vec![
            BbhTask::BooleanExpressions,
            BbhTask::CausalJudgment,
            BbhTask::DateUnderstanding,
            BbhTask::DisambiguationQa,
            BbhTask::FormalFallacies,
            BbhTask::GeometricShapes,
            BbhTask::Hyperbaton,
            BbhTask::LogicalDeductionFiveObjects,
            BbhTask::LogicalDeductionSevenObjects,
            BbhTask::LogicalDeductionThreeObjects,
            BbhTask::MovieRecommendation,
            BbhTask::Navigate,
            BbhTask::PenguinsInATable,
            BbhTask::ReasoningAboutColoredObjects,
            BbhTask::RuinNames,
            BbhTask::SalientTranslationErrorDetection,
            BbhTask::Snarks,
            BbhTask::SportsUnderstanding,
            BbhTask::TemporalSequences,
            BbhTask::TrackingShuffledObjectsFiveObjects,
            BbhTask::TrackingShuffledObjectsSevenObjects,
            BbhTask::TrackingShuffledObjectsThreeObjects,
            BbhTask::WebOfLies,
        ]
    }

    /// Get the task ID string
    pub fn as_str(&self) -> &'static str {
        match self {
            BbhTask::BooleanExpressions => "boolean_expressions",
            BbhTask::CausalJudgment => "causal_judgment",
            BbhTask::DateUnderstanding => "date_understanding",
            BbhTask::DisambiguationQa => "disambiguation_qa",
            BbhTask::FormalFallacies => "formal_fallacies",
            BbhTask::GeometricShapes => "geometric_shapes",
            BbhTask::Hyperbaton => "hyperbaton",
            BbhTask::LogicalDeductionFiveObjects => "logical_deduction_five_objects",
            BbhTask::LogicalDeductionSevenObjects => "logical_deduction_seven_objects",
            BbhTask::LogicalDeductionThreeObjects => "logical_deduction_three_objects",
            BbhTask::MovieRecommendation => "movie_recommendation",
            BbhTask::Navigate => "navigate",
            BbhTask::PenguinsInATable => "penguins_in_a_table",
            BbhTask::ReasoningAboutColoredObjects => "reasoning_about_colored_objects",
            BbhTask::RuinNames => "ruin_names",
            BbhTask::SalientTranslationErrorDetection => "salient_translation_error_detection",
            BbhTask::Snarks => "snarks",
            BbhTask::SportsUnderstanding => "sports_understanding",
            BbhTask::TemporalSequences => "temporal_sequences",
            BbhTask::TrackingShuffledObjectsFiveObjects => "tracking_shuffled_objects_five_objects",
            BbhTask::TrackingShuffledObjectsSevenObjects => "tracking_shuffled_objects_seven_objects",
            BbhTask::TrackingShuffledObjectsThreeObjects => "tracking_shuffled_objects_three_objects",
            BbhTask::WebOfLies => "web_of_lies",
        }
    }

    /// Get human-readable task name
    pub fn name(&self) -> &'static str {
        match self {
            BbhTask::BooleanExpressions => "Boolean Expressions",
            BbhTask::CausalJudgment => "Causal Judgment",
            BbhTask::DateUnderstanding => "Date Understanding",
            BbhTask::DisambiguationQa => "Disambiguation QA",
            BbhTask::FormalFallacies => "Formal Fallacies",
            BbhTask::GeometricShapes => "Geometric Shapes",
            BbhTask::Hyperbaton => "Hyperbaton",
            BbhTask::LogicalDeductionFiveObjects => "Logical Deduction (5 Objects)",
            BbhTask::LogicalDeductionSevenObjects => "Logical Deduction (7 Objects)",
            BbhTask::LogicalDeductionThreeObjects => "Logical Deduction (3 Objects)",
            BbhTask::MovieRecommendation => "Movie Recommendation",
            BbhTask::Navigate => "Navigate",
            BbhTask::PenguinsInATable => "Penguins in a Table",
            BbhTask::ReasoningAboutColoredObjects => "Reasoning about Colored Objects",
            BbhTask::RuinNames => "Ruin Names",
            BbhTask::SalientTranslationErrorDetection => "Salient Translation Error Detection",
            BbhTask::Snarks => "Snarks",
            BbhTask::SportsUnderstanding => "Sports Understanding",
            BbhTask::TemporalSequences => "Temporal Sequences",
            BbhTask::TrackingShuffledObjectsFiveObjects => "Tracking Shuffled Objects (5 Objects)",
            BbhTask::TrackingShuffledObjectsSevenObjects => "Tracking Shuffled Objects (7 Objects)",
            BbhTask::TrackingShuffledObjectsThreeObjects => "Tracking Shuffled Objects (3 Objects)",
            BbhTask::WebOfLies => "Web of Lies",
        }
    }

    /// Get difficulty level for this task
    pub fn difficulty(&self) -> Difficulty {
        match self {
            BbhTask::BooleanExpressions => Difficulty::Medium,
            BbhTask::CausalJudgment => Difficulty::Hard,
            BbhTask::DateUnderstanding => Difficulty::Medium,
            BbhTask::DisambiguationQa => Difficulty::Medium,
            BbhTask::FormalFallacies => Difficulty::Hard,
            BbhTask::GeometricShapes => Difficulty::Hard,
            BbhTask::Hyperbaton => Difficulty::Easy,
            BbhTask::LogicalDeductionFiveObjects => Difficulty::Medium,
            BbhTask::LogicalDeductionSevenObjects => Difficulty::Hard,
            BbhTask::LogicalDeductionThreeObjects => Difficulty::Easy,
            BbhTask::MovieRecommendation => Difficulty::Medium,
            BbhTask::Navigate => Difficulty::Medium,
            BbhTask::PenguinsInATable => Difficulty::Hard,
            BbhTask::ReasoningAboutColoredObjects => Difficulty::Medium,
            BbhTask::RuinNames => Difficulty::Medium,
            BbhTask::SalientTranslationErrorDetection => Difficulty::Hard,
            BbhTask::Snarks => Difficulty::Hard,
            BbhTask::SportsUnderstanding => Difficulty::Medium,
            BbhTask::TemporalSequences => Difficulty::Hard,
            BbhTask::TrackingShuffledObjectsFiveObjects => Difficulty::Medium,
            BbhTask::TrackingShuffledObjectsSevenObjects => Difficulty::Hard,
            BbhTask::TrackingShuffledObjectsThreeObjects => Difficulty::Easy,
            BbhTask::WebOfLies => Difficulty::Hard,
        }
    }
}

impl std::str::FromStr for BbhTask {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "boolean_expressions" => Ok(BbhTask::BooleanExpressions),
            "causal_judgment" => Ok(BbhTask::CausalJudgment),
            "date_understanding" => Ok(BbhTask::DateUnderstanding),
            "disambiguation_qa" => Ok(BbhTask::DisambiguationQa),
            "formal_fallacies" => Ok(BbhTask::FormalFallacies),
            "geometric_shapes" => Ok(BbhTask::GeometricShapes),
            "hyperbaton" => Ok(BbhTask::Hyperbaton),
            "logical_deduction_five_objects" => Ok(BbhTask::LogicalDeductionFiveObjects),
            "logical_deduction_seven_objects" => Ok(BbhTask::LogicalDeductionSevenObjects),
            "logical_deduction_three_objects" => Ok(BbhTask::LogicalDeductionThreeObjects),
            "movie_recommendation" => Ok(BbhTask::MovieRecommendation),
            "navigate" => Ok(BbhTask::Navigate),
            "penguins_in_a_table" => Ok(BbhTask::PenguinsInATable),
            "reasoning_about_colored_objects" => Ok(BbhTask::ReasoningAboutColoredObjects),
            "ruin_names" => Ok(BbhTask::RuinNames),
            "salient_translation_error_detection" => Ok(BbhTask::SalientTranslationErrorDetection),
            "snarks" => Ok(BbhTask::Snarks),
            "sports_understanding" => Ok(BbhTask::SportsUnderstanding),
            "temporal_sequences" => Ok(BbhTask::TemporalSequences),
            "tracking_shuffled_objects_five_objects" => Ok(BbhTask::TrackingShuffledObjectsFiveObjects),
            "tracking_shuffled_objects_seven_objects" => Ok(BbhTask::TrackingShuffledObjectsSevenObjects),
            "tracking_shuffled_objects_three_objects" => Ok(BbhTask::TrackingShuffledObjectsThreeObjects),
            "web_of_lies" => Ok(BbhTask::WebOfLies),
            _ => Err(format!("Unknown BBH task: {}", s)),
        }
    }
}

/// BBH Benchmark implementation
pub struct BbhBenchmark {
    /// Few-shot examples for prompting
    few_shot_examples: Vec<DataSample>,
    /// Whether to use chain-of-thought prompting
    use_cot: bool,
    /// Specific tasks to evaluate (None = all)
    tasks: Option<Vec<BbhTask>>,
}

impl BbhBenchmark {
    /// Create a new BBH benchmark
    pub fn new() -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
            use_cot: true,
            tasks: None,
        }
    }

    /// Create a new BBH benchmark with specific tasks
    pub fn with_tasks(tasks: Vec<BbhTask>) -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
            use_cot: true,
            tasks: Some(tasks),
        }
    }

    /// Create a new BBH benchmark without chain-of-thought
    pub fn without_cot() -> Self {
        Self {
            few_shot_examples: Self::default_few_shot_examples(),
            use_cot: false,
            tasks: None,
        }
    }

    /// Set whether to use chain-of-thought prompting
    pub fn with_cot(mut self, use_cot: bool) -> Self {
        self.use_cot = use_cot;
        self
    }

    /// Default few-shot examples for BBH
    fn default_few_shot_examples() -> Vec<DataSample> {
        vec![
            DataSample::new("fs1", "Q: True or false: If a figure is a rectangle, then it is a square.")
                .with_expected("False")
                .with_category("logical_reasoning"),
            DataSample::new("fs2", "Q: Alice is older than Bob. Bob is older than Charlie. Is Alice older than Charlie?")
                .with_expected("Yes")
                .with_category("logical_deduction"),
        ]
    }

    /// Load sample BBH data (for testing)
    fn load_sample_data() -> Vec<DataSample> {
        vec![
            // Boolean Expressions
            DataSample::new("bbh_boolean_1",
                "Q: True or false: (not ( ( True and False ) ) and ( True and ( ( not False ) ) ))")
                .with_expected("True")
                .with_category("boolean_expressions")
                .with_difficulty(Difficulty::Medium),

            // Causal Judgment
            DataSample::new("bbh_causal_1",
                "Q: How would a violinist feel if their violin string broke during a performance?")
                .with_expected("The violinist would feel embarrassed and frustrated.")
                .with_category("causal_judgment")
                .with_difficulty(Difficulty::Hard),

            // Date Understanding
            DataSample::new("bbh_date_1",
                "Q: Today is March 15, 2024. What is the date 10 days ago in MM/DD/YYYY format?")
                .with_expected("03/05/2024")
                .with_category("date_understanding")
                .with_difficulty(Difficulty::Medium),

            // Disambiguation QA
            DataSample::new("bbh_disambig_1",
                "Q: In the sentence 'The trophy didn't fit into the brown suitcase because it was too large. What was too large?' Answer choices: A) the trophy B) the suitcase")
                .with_expected("A")
                .with_category("disambiguation_qa")
                .with_difficulty(Difficulty::Medium),

            // Formal Fallacies
            DataSample::new("bbh_fallacies_1",
                "Q: 'All roses are flowers. Some flowers fade quickly. Therefore some roses fade quickly.' Is this argument valid?")
                .with_expected("No")
                .with_category("formal_fallacies")
                .with_difficulty(Difficulty::Hard),

            // Geometric Shapes
            DataSample::new("bbh_geo_1",
                "Q: What shape has 4 sides, all equal in length, and 4 right angles?")
                .with_expected("Square")
                .with_category("geometric_shapes")
                .with_difficulty(Difficulty::Easy),

            // Hyperbaton (word order)
            DataSample::new("bbh_hyper_1",
                "Q: Which sentence has the correct word order: A) 'Quickly the fox brown jumped' or B) 'The quick brown fox jumped quickly'?")
                .with_expected("B")
                .with_category("hyperbaton")
                .with_difficulty(Difficulty::Easy),

            // Logical Deduction (3 objects)
            DataSample::new("bbh_logic3_1",
                "Q: Alice, Bob, and Carol are wearing red, blue, and green shirts. Alice is not wearing red. Bob is wearing blue. What color is Carol wearing?")
                .with_expected("Green")
                .with_category("logical_deduction_three_objects")
                .with_difficulty(Difficulty::Easy),

            // Logical Deduction (5 objects)
            DataSample::new("bbh_logic5_1",
                "Q: Five people (A, B, C, D, E) live in five houses of different colors (red, blue, green, yellow, white). A lives in the red house. B lives next to A. C lives in blue. D doesn't live in white. E lives between B and D. Who lives in the white house?")
                .with_expected("E")
                .with_category("logical_deduction_five_objects")
                .with_difficulty(Difficulty::Medium),

            // Logical Deduction (7 objects)
            DataSample::new("bbh_logic7_1",
                "Q: Seven books are arranged on a shelf. Book A is to the left of Book B. Book C is to the right of Book B. Book D is between A and B. Book E is to the left of A. Book F is to the right of C. Book G is between E and A. Which book is leftmost?")
                .with_expected("E")
                .with_category("logical_deduction_seven_objects")
                .with_difficulty(Difficulty::Hard),

            // Movie Recommendation
            DataSample::new("bbh_movie_1",
                "Q: A user likes 'The Matrix', 'Inception', and 'Blade Runner'. Which movie would you recommend: A) The Notebook B) The Terminator?")
                .with_expected("B")
                .with_category("movie_recommendation")
                .with_difficulty(Difficulty::Medium),

            // Navigate
            DataSample::new("bbh_nav_1",
                "Q: If you face north and turn 90 degrees to your right, then 180 degrees to your left, what direction are you facing?")
                .with_expected("West")
                .with_category("navigate")
                .with_difficulty(Difficulty::Medium),

            // Penguins in a Table
            DataSample::new("bbh_penguin_1",
                "Q: Based on the table: Emperor penguins live in Antarctica, are 110cm tall, and weigh 30kg. King penguins live in sub-Antarctic islands, are 90cm tall, and weigh 15kg. Where do King penguins live?")
                .with_expected("Sub-Antarctic islands")
                .with_category("penguins_in_a_table")
                .with_difficulty(Difficulty::Medium),

            // Reasoning about Colored Objects
            DataSample::new("bbh_color_1",
                "Q: On the table, there is a red book, a blue pen, and a green notebook. The red book is to the left of the blue pen. What color is the object to the right of the red book?")
                .with_expected("Blue")
                .with_category("reasoning_about_colored_objects")
                .with_difficulty(Difficulty::Medium),

            // Ruin Names
            DataSample::new("bbh_ruin_1",
                "Q: Which of these is a humorous alteration of a celebrity name: A) Brad Pitt B) Brad Sh Pitt?")
                .with_expected("B")
                .with_category("ruin_names")
                .with_difficulty(Difficulty::Medium),

            // Salient Translation Error Detection
            DataSample::new("bbh_trans_1",
                "Q: Original: 'The cat sat on the mat.' Translation: 'The dog sat on the mat.' What type of error is this? A) Omission B) Addition C) Mistranslation")
                .with_expected("C")
                .with_category("salient_translation_error_detection")
                .with_difficulty(Difficulty::Medium),

            // Snarks (sarcasm detection)
            DataSample::new("bbh_snark_1",
                "Q: Statement: 'Oh great, another meeting that could have been an email.' Is this sarcastic?")
                .with_expected("Yes")
                .with_category("snarks")
                .with_difficulty(Difficulty::Hard),

            // Sports Understanding
            DataSample::new("bbh_sports_1",
                "Q: In basketball, if a player is fouled while shooting and misses, what happens?")
                .with_expected("They get free throws")
                .with_category("sports_understanding")
                .with_difficulty(Difficulty::Medium),

            // Temporal Sequences
            DataSample::new("bbh_temp_1",
                "Q: Event A happened on Monday. Event B happened on Wednesday. Event C happened on Tuesday. Arrange in chronological order.")
                .with_expected("A, C, B")
                .with_category("temporal_sequences")
                .with_difficulty(Difficulty::Medium),

            // Tracking Shuffled Objects (3 objects)
            DataSample::new("bbh_track3_1",
                "Q: A ball, a cup, and a pen are on a table. The ball is swapped with the cup. Then the cup is swapped with the pen. Where is the ball?")
                .with_expected("Where the pen was originally")
                .with_category("tracking_shuffled_objects_three_objects")
                .with_difficulty(Difficulty::Easy),

            // Tracking Shuffled Objects (5 objects)
            DataSample::new("bbh_track5_1",
                "Q: Five items (A, B, C, D, E) are arranged left to right. A and B swap. Then C and D swap. Then B and E swap. What is the position of A?")
                .with_expected("Position 2")
                .with_category("tracking_shuffled_objects_five_objects")
                .with_difficulty(Difficulty::Medium),

            // Tracking Shuffled Objects (7 objects)
            DataSample::new("bbh_track7_1",
                "Q: Seven cards (1-7) are in order. Cards 1 and 7 swap. Then cards 3 and 5 swap. Then cards 2 and 6 swap. Then cards 4 and 7 swap. Where is card 1?")
                .with_expected("Position 4")
                .with_category("tracking_shuffled_objects_seven_objects")
                .with_difficulty(Difficulty::Hard),

            // Web of Lies
            DataSample::new("bbh_lies_1",
                "Q: Person A says: 'Person B is lying.' Person B says: 'Person A is telling the truth.' Is Person A telling the truth?")
                .with_expected("This creates a paradox; neither statement can be consistently true.")
                .with_category("web_of_lies")
                .with_difficulty(Difficulty::Hard),
        ]
    }

    /// Extract the answer from a response
    /// Handles formats like "Answer: X" or just "X"
    fn extract_answer(response: &str) -> String {
        let response = response.trim();

        // Try to find "Answer:" pattern
        if let Some(pos) = response.to_lowercase().find("answer:") {
            let after = &response[pos + 7..];
            // Extract the rest of the line or until a period/newline
            let answer = after
                .lines()
                .next()
                .unwrap_or(after)
                .trim()
                .trim_end_matches('.')
                .trim()
                .to_string();
            if !answer.is_empty() {
                return answer;
            }
        }

        // Try to find "The answer is" pattern
        for pattern in &["the answer is", "answer is", "therefore", "so the answer is"] {
            if let Some(pos) = response.to_lowercase().find(pattern) {
                let after = &response[pos + pattern.len()..];
                let answer = after
                    .trim_start()
                    .trim_start_matches(":")
                    .trim_start()
                    .lines()
                    .next()
                    .unwrap_or(after)
                    .trim()
                    .trim_end_matches('.')
                    .trim()
                    .to_string();
                if !answer.is_empty() && answer.len() < 100 {
                    return answer;
                }
            }
        }

        // If response is short (single line), use it directly
        if !response.contains('\n') && response.len() < 100 {
            return response.to_string();
        }

        // Otherwise, try the last non-empty line
        if let Some(last_line) = response.lines().filter(|l| !l.trim().is_empty()).last() {
            let trimmed = last_line.trim().trim_end_matches('.').trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }

        response.to_string()
    }

    /// Normalize answer for comparison
    fn normalize_answer(answer: &str) -> String {
        answer
            .to_lowercase()
            .trim()
            .replace(".", "")
            .replace(",", "")
            .replace("'", "")
            .replace("\"", "")
            .replace("the ", "")
            .replace(" a ", " ")
            .replace("  ", " ")
            .trim()
            .to_string()
    }

    /// Check if two answers match
    fn answers_match(expected: &str, generated: &str) -> bool {
        let expected_norm = Self::normalize_answer(expected);
        let generated_norm = Self::normalize_answer(generated);

        // Direct match
        if expected_norm == generated_norm {
            return true;
        }

        // Check if generated contains expected (for longer answers)
        if generated_norm.contains(&expected_norm) || expected_norm.contains(&generated_norm) {
            return true;
        }

        // For yes/no answers
        if (expected_norm == "yes" || expected_norm == "true") &&
           (generated_norm.starts_with("yes") || generated_norm.starts_with("true")) {
            return true;
        }
        if (expected_norm == "no" || expected_norm == "false") &&
           (generated_norm.starts_with("no") || generated_norm.starts_with("false")) {
            return true;
        }

        // For option letters (A, B, C, D)
        if expected_norm.len() == 1 && generated_norm.len() >= 1 {
            let first_char = generated_norm.chars().next().unwrap();
            if expected_norm.chars().next().unwrap() == first_char &&
               (first_char >= 'a' && first_char <= 'z' || first_char >= '0' && first_char <= '9') {
                return true;
            }
        }

        false
    }

    /// Get CoT instruction based on whether CoT is enabled
    fn get_cot_instruction(&self) -> &'static str {
        if self.use_cot {
            "Think step by step and explain your reasoning. Then provide your final answer starting with 'Answer:'."
        } else {
            "Provide only the answer without explanation."
        }
    }
}

impl Default for BbhBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for BbhBenchmark {
    fn id(&self) -> &str {
        "bbh"
    }

    fn name(&self) -> &str {
        "BBH (Big Bench Hard)"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }

    fn description(&self) -> &str {
        "A challenging reasoning benchmark with 23 tasks requiring complex multi-step reasoning. Tests logical deduction, causal reasoning, temporal reasoning, and more."
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples = Self::load_sample_data();
        Ok(Dataset::new("BBH", samples))
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

        // Filter by specific tasks if specified
        if let Some(ref tasks) = self.tasks {
            let task_strs: Vec<String> = tasks.iter().map(|t| t.as_str().to_string()).collect();
            samples.retain(|s| {
                s.category
                    .as_ref()
                    .map(|c| task_strs.contains(c))
                    .unwrap_or(false)
            });
        }

        debug!("Running BBH benchmark with {} samples", samples.len());

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
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("use_cot".to_string(), serde_json::Value::Bool(self.use_cot));
                meta.insert("num_tasks".to_string(), serde_json::Value::Number(BbhTask::all().len().into()));
                meta
            },
        })
    }

    fn evaluate_response(&self, sample: &DataSample, response: &str) -> f64 {
        let expected = match &sample.expected_output {
            Some(e) => e.trim(),
            None => return 0.0,
        };

        let extracted = Self::extract_answer(response);

        if Self::answers_match(expected, &extracted) {
            1.0
        } else {
            0.0
        }
    }

    fn format_prompt(&self, sample: &DataSample, few_shot_examples: &[DataSample]) -> String {
        let mut prompt = String::new();

        // Add instruction
        prompt.push_str("You are solving a reasoning problem. ");
        prompt.push_str(self.get_cot_instruction());
        prompt.push_str("\n\n");

        // Add few-shot examples
        for example in few_shot_examples {
            prompt.push_str(&format!(
                "{}\nAnswer: {}\n\n",
                example.input,
                example.expected_output.as_deref().unwrap_or("")
            ));
        }

        // Add the actual question
        prompt.push_str(&sample.input);
        prompt.push('\n');

        // Add answer prefix if using CoT
        if self.use_cot {
            prompt.push_str("Answer:");
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbh_creation() {
        let benchmark = BbhBenchmark::new();
        assert_eq!(benchmark.id(), "bbh");
        assert_eq!(benchmark.category(), BenchmarkCategory::Reasoning);
    }

    #[test]
    fn test_bbh_without_cot() {
        let benchmark = BbhBenchmark::without_cot();
        assert!(!benchmark.use_cot);
    }

    #[test]
    fn test_bbh_with_cot() {
        let benchmark = BbhBenchmark::new().with_cot(true);
        assert!(benchmark.use_cot);
    }

    #[test]
    fn test_extract_answer_simple() {
        assert_eq!(BbhBenchmark::extract_answer("True"), "True");
        assert_eq!(BbhBenchmark::extract_answer("A"), "A");
        assert_eq!(BbhBenchmark::extract_answer("Yes"), "Yes");
    }

    #[test]
    fn test_extract_answer_with_prefix() {
        assert_eq!(BbhBenchmark::extract_answer("The answer is True."), "True");
        assert_eq!(BbhBenchmark::extract_answer("Answer: B"), "B");
        assert_eq!(BbhBenchmark::extract_answer("Therefore, the answer is Square"), "Square");
    }

    #[test]
    fn test_normalize_answer() {
        assert_eq!(BbhBenchmark::normalize_answer("True"), "true");
        assert_eq!(BbhBenchmark::normalize_answer("The Answer."), "answer");
        assert_eq!(BbhBenchmark::normalize_answer("A "), "a");
    }

    #[test]
    fn test_answers_match() {
        assert!(BbhBenchmark::answers_match("True", "True"));
        assert!(BbhBenchmark::answers_match("True", "true"));
        assert!(BbhBenchmark::answers_match("Yes", "yes"));
        assert!(BbhBenchmark::answers_match("A", "A"));
        assert!(BbhBenchmark::answers_match("A", "Answer: A"));
        assert!(!BbhBenchmark::answers_match("True", "False"));
    }

    #[test]
    fn test_task_enum() {
        let tasks = BbhTask::all();
        assert_eq!(tasks.len(), 23);

        // Test a few specific tasks
        assert_eq!(BbhTask::BooleanExpressions.as_str(), "boolean_expressions");
        assert_eq!(BbhTask::WebOfLies.as_str(), "web_of_lies");
        assert_eq!(BbhTask::LogicalDeductionThreeObjects.name(), "Logical Deduction (3 Objects)");
    }

    #[test]
    fn test_task_from_str() {
        assert_eq!(
            "boolean_expressions".parse::<BbhTask>().unwrap(),
            BbhTask::BooleanExpressions
        );
        assert_eq!(
            "web_of_lies".parse::<BbhTask>().unwrap(),
            BbhTask::WebOfLies
        );
        assert!("unknown_task".parse::<BbhTask>().is_err());
    }

    #[test]
    fn test_format_prompt() {
        let benchmark = BbhBenchmark::new();
        let sample = DataSample::new("test", "What is 2+2?")
            .with_expected("4");

        let prompt = benchmark.format_prompt(&sample, &[]);
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("Think step by step"));
        assert!(prompt.contains("Answer:"));
    }

    #[test]
    fn test_format_prompt_no_cot() {
        let benchmark = BbhBenchmark::without_cot();
        let sample = DataSample::new("test", "What is 2+2?")
            .with_expected("4");

        let prompt = benchmark.format_prompt(&sample, &[]);
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("Provide only the answer"));
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = BbhBenchmark::new();
        let sample = DataSample::new("test", "Is the sky blue?")
            .with_expected("Yes");

        assert_eq!(benchmark.evaluate_response(&sample, "Yes"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is Yes."), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "Answer: Yes"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "No"), 0.0);
    }

    #[test]
    fn test_load_sample_data() {
        let samples = BbhBenchmark::load_sample_data();
        assert!(samples.len() >= 15);

        // Check that we have samples from different categories
        let categories: std::collections::HashSet<_> = samples
            .iter()
            .filter_map(|s| s.category.clone())
            .collect();
        assert!(categories.len() > 5); // Should have multiple different tasks
    }

    #[test]
    fn test_with_tasks() {
        let tasks = vec![BbhTask::BooleanExpressions, BbhTask::WebOfLies];
        let benchmark = BbhBenchmark::with_tasks(tasks);
        assert!(benchmark.tasks.is_some());
    }
}
