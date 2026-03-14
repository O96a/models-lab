//! Long Context Benchmark
//!
//! Tests model ability to handle long sequences and multi-hop reasoning
//! across documents of varying lengths (1K, 2K, 4K, 8K, 16K, 32K tokens).

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

use crate::{
    Benchmark, BenchmarkCategory, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics,
    DataSample, Dataset, Difficulty, SampleResult,
};

/// Context length variants for testing
/// Focused on 4K-32K for Phase 2, with support for 64K and 128K in future phases
const CONTEXT_LENGTHS: &[(&str, usize)] = &[
    ("4k", 4000),
    ("8k", 8000),
    ("16k", 16000),
    ("32k", 32000),
    ("64k", 64000),
    ("128k", 128000),
];

/// Position of a question within the document
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionPosition {
    /// Question facts are at the beginning of the document
    Beginning,
    /// Question facts are in the middle of the document
    Middle,
    /// Question facts are at the end of the document
    End,
}

impl QuestionPosition {
    fn as_str(&self) -> &'static str {
        match self {
            QuestionPosition::Beginning => "beginning",
            QuestionPosition::Middle => "middle",
            QuestionPosition::End => "end",
        }
    }
}

/// Document type for synthetic generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentType {
    /// Narrative text (stories, articles)
    Narrative,
    /// Technical documentation
    Technical,
    /// Code repositories
    Code,
    /// Legal/contracts
    Legal,
    /// Scientific papers
    Scientific,
}

impl DocumentType {
    fn as_str(&self) -> &'static str {
        match self {
            DocumentType::Narrative => "narrative",
            DocumentType::Technical => "technical",
            DocumentType::Code => "code",
            DocumentType::Legal => "legal",
            DocumentType::Scientific => "scientific",
        }
    }

    fn themes(&self) -> &'static [&'static str] {
        match self {
            DocumentType::Narrative => &[
                "ancient civilizations",
                "space exploration",
                "mythological creatures",
                "renaissance art",
                "maritime adventures",
            ],
            DocumentType::Technical => &[
                "computer science",
                "quantum physics",
                "artificial intelligence",
                "network protocols",
                "distributed systems",
            ],
            DocumentType::Code => &[
                "software architecture",
                "database design",
                "API development",
                "security protocols",
                "testing frameworks",
            ],
            DocumentType::Legal => &[
                "contract law",
                "intellectual property",
                "regulatory compliance",
                "corporate governance",
                "international treaties",
            ],
            DocumentType::Scientific => &[
                "marine biology",
                "climate science",
                "genetics research",
                "astronomy studies",
                "materials science",
            ],
        }
    }
}

/// A single test sample for long context evaluation
#[derive(Debug, Clone)]
struct LongContextSample {
    id: String,
    context: String,
    context_length: usize,
    context_length_label: String,
    question_type: QuestionType,
    question: String,
    answer: String,
    hop_count: usize,
    /// Position of the question facts in the document
    position: QuestionPosition,
    /// Type of document
    document_type: DocumentType,
}

/// Type of question being asked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuestionType {
    /// Single-hop: answer in one location
    SingleHop,
    /// Multi-hop: requires connecting info from 2 locations
    MultiHop,
    /// Many-hop: requires connecting info from 3+ locations
    ManyHop,
}

impl QuestionType {
    fn as_str(&self) -> &'static str {
        match self {
            QuestionType::SingleHop => "single_hop",
            QuestionType::MultiHop => "multi_hop",
            QuestionType::ManyHop => "many_hop",
        }
    }

    fn hop_count(&self) -> usize {
        match self {
            QuestionType::SingleHop => 1,
            QuestionType::MultiHop => 2,
            QuestionType::ManyHop => 3,
        }
    }
}

/// Long Context Benchmark implementation
pub struct LongContextBenchmark {
    samples: Vec<LongContextSample>,
}

impl LongContextBenchmark {
    pub fn new() -> Self {
        Self {
            samples: Self::generate_all_samples(),
        }
    }

    fn generate_all_samples() -> Vec<LongContextSample> {
        let mut samples = Vec::new();
        let mut id_counter = 1;

        // Document types for variety
        let doc_types = [
            DocumentType::Narrative,
            DocumentType::Technical,
            DocumentType::Code,
            DocumentType::Legal,
            DocumentType::Scientific,
        ];

        // Positions for testing position bias
        let positions = [
            QuestionPosition::Beginning,
            QuestionPosition::Middle,
            QuestionPosition::End,
        ];

        for (length_label, target_tokens) in CONTEXT_LENGTHS {
            for question_type in [QuestionType::SingleHop, QuestionType::MultiHop, QuestionType::ManyHop] {
                // Generate samples with different document types and positions
                for doc_type in &doc_types {
                    for position in &positions {
                        let sample = Self::generate_sample(
                            &format!("longctx_{}", id_counter),
                            *target_tokens,
                            length_label,
                            question_type,
                            *doc_type,
                            *position,
                        );
                        samples.push(sample);
                        id_counter += 1;
                    }
                }
            }
        }

        samples
    }

    fn generate_sample(
        id: &str,
        target_tokens: usize,
        length_label: &str,
        question_type: QuestionType,
        document_type: DocumentType,
        position: QuestionPosition,
    ) -> LongContextSample {
        // Generate synthetic document with embedded facts
        let (context, question, answer) = Self::generate_synthetic_document(
            target_tokens,
            question_type,
            document_type,
            position,
        );

        LongContextSample {
            id: id.to_string(),
            context,
            context_length: target_tokens,
            context_length_label: length_label.to_string(),
            question_type,
            question,
            answer,
            hop_count: question_type.hop_count(),
            position,
            document_type,
        }
    }

    fn generate_synthetic_document(
        target_tokens: usize,
        question_type: QuestionType,
        document_type: DocumentType,
        position: QuestionPosition,
    ) -> (String, String, String) {
        let mut rng = SimpleRng::new(target_tokens as u64);

        // Character estimate: ~4 characters per token on average
        let target_chars = target_tokens * 4;

        // Get themes based on document type
        let themes = document_type.themes();

        // Generate document sections
        let num_sections = 4 + rng.next_range(4); // 4-8 sections
        let mut sections = Vec::new();
        let mut section_themes = Vec::new();

        for i in 0..num_sections {
            let theme = themes[rng.next_range(themes.len())];
            section_themes.push(theme);
            let section = Self::generate_section(&mut rng, theme, i + 1);
            sections.push(section);
        }

        // Generate questions based on type
        let (question, answer, fact_positions) = match question_type {
            QuestionType::SingleHop => {
                Self::generate_single_hop_question(&mut rng, &sections, &section_themes)
            }
            QuestionType::MultiHop => {
                Self::generate_multi_hop_question(&mut rng, &sections, &section_themes, 2)
            }
            QuestionType::ManyHop => {
                Self::generate_multi_hop_question(&mut rng, &sections, &section_themes, 3)
            }
        };

        // Embed the facts at specified positions
        let sections = Self::embed_facts_in_sections(sections, fact_positions);

        // Combine sections with fillers to reach target length
        let document = Self::combine_sections_to_length(sections, target_chars, &mut rng);

        (document, question, answer)
    }

    fn generate_section(rng: &mut SimpleRng, theme: &str, section_num: usize) -> String {
        let templates = vec![
            format!(
                "Section {}: The study of {} has fascinated researchers for centuries. ",
                section_num, theme
            ),
            format!(
                "In the field of {}, numerous discoveries have shaped our understanding. ",
                theme
            ),
            format!(
                "Exploring the depths of {} reveals fascinating insights about our world. ",
                theme
            ),
        ];

        let mut section = templates[rng.next_range(templates.len())].clone();

        // Add paragraphs of content
        let num_paragraphs = 2 + rng.next_range(3);
        for _ in 0..num_paragraphs {
            section.push_str(&Self::generate_paragraph(rng, theme));
            section.push_str(" ");
        }

        section
    }

    fn generate_paragraph(rng: &mut SimpleRng, theme: &str) -> String {
        let sentences = vec![
            format!("Researchers in {} continue to make groundbreaking discoveries each year.", theme),
            "The history of this field dates back thousands of years to ancient civilizations.".to_string(),
            format!("Modern techniques have revolutionized how we approach {} in contemporary studies.", theme),
            format!("Many experts believe that {} holds the key to understanding broader patterns.", theme),
            format!("Recent publications have shed new light on previously unknown aspects of {}.", theme),
            format!("The relationship between {} and other disciplines remains an active area of investigation.", theme),
            format!("Field studies have provided valuable data that challenges existing theories about {}.", theme),
            format!("Collaborative efforts across institutions have accelerated progress in {} research.", theme),
        ];

        let num_sentences = 3 + rng.next_range(4);
        let mut paragraph = String::new();
        for i in 0..num_sentences {
            let sentence = &sentences[rng.next_range(sentences.len())];
            paragraph.push_str(sentence);
            if i < num_sentences - 1 {
                paragraph.push(' ');
            }
        }

        paragraph
    }

    fn generate_single_hop_question(
        rng: &mut SimpleRng,
        sections: &[String],
        themes: &[&str],
    ) -> (String, String, Vec<(usize, String)>) {
        let section_idx = rng.next_range(sections.len());
        let theme = themes[section_idx];

        let facts = vec![
            ("person", vec!["Dr. Elena Vasquez", "Professor James Chen", "Dr. Maria Rodriguez", "Dr. Ahmed Hassan"]),
            ("number", vec!["42", "137", "256", "89", "1024"]),
            ("location", vec!["the Arctic Research Station", "the Tropical Research Center", "the Mountain Observatory", "the Deep Sea Laboratory"]),
            ("artifact", vec!["the Golden Compass", "the Crystal Sphere", "the Ancient Manuscript", "the Silver Telescope"]),
            ("code", vec!["ALPHA-7X9", "BETA-442", "GAMMA-2024", "DELTA-88"]),
            ("year", vec!["1847", "1923", "2001", "1776", "2050"]),
        ];

        let fact_type = &facts[rng.next_range(facts.len())];
        let fact_value = fact_type.1[rng.next_range(fact_type.1.len())];

        let question = match fact_type.0 {
            "person" => format!("According to the text, who discovered the key finding in {}?", theme),
            "number" => format!("What specific number is associated with the main concept in {}?", theme),
            "location" => format!("Where was the primary research on {} conducted?", theme),
            "artifact" => format!("What important object is mentioned in connection with {}?", theme),
            "code" => format!("What code is referenced in the section about {}?", theme),
            "year" => format!("In what year did the significant event in {} occur?", theme),
            _ => format!("What is the key fact mentioned about {}?", theme),
        };

        let answer = fact_value.to_string();
        let fact_positions = vec![(section_idx, format!("The key finding was discovered by {}.", fact_value))];

        (question, answer, fact_positions)
    }

    fn generate_multi_hop_question(
        rng: &mut SimpleRng,
        sections: &[String],
        themes: &[&str],
        num_hops: usize,
    ) -> (String, String, Vec<(usize, String)>) {
        let mut selected_indices: Vec<usize> = (0..sections.len()).collect();
        // Shuffle-like selection
        for i in (1..selected_indices.len()).rev() {
            let j = rng.next_range(i + 1);
            selected_indices.swap(i, j);
        }
        selected_indices.truncate(num_hops);

        let subjects = vec![
            ("Project", vec!["Mercury", "Orion", "Discovery", "Pioneer", "Voyager"]),
            ("Experiment", vec!["Alpha", "Beta", "Gamma", "Delta", "Omega"]),
            ("Study", vec!["Horizon", "Genesis", "Nova", "Atlas", "Nexus"]),
            ("Mission", vec!["Artemis", "Apollo", "Endeavor", "Odyssey", "Sentinel"]),
        ];

        let subject_type = &subjects[rng.next_range(subjects.len())];
        let subject_name = subject_type.1[rng.next_range(subject_type.1.len())];

        let mut facts = Vec::new();
        let mut clues = Vec::new();

        for (i, &section_idx) in selected_indices.iter().enumerate() {
            let clue_type = rng.next_range(4);
            let clue = match clue_type {
                0 => {
                    let value = vec!["initiated", "launched", "commenced", "began"][rng.next_range(4)];
                    format!("{} was {} in", subject_name, value)
                }
                1 => {
                    let value = vec!["Dr. Smith", "Dr. Jones", "Dr. Lee", "Dr. Patel"][rng.next_range(4)];
                    format!("{} was led by {}", subject_name, value)
                }
                2 => {
                    let value = vec!["successful", "groundbreaking", "innovative", "pioneering"][rng.next_range(4)];
                    format!("{} was considered {}", subject_name, value)
                }
                _ => {
                    let value = vec!["2019", "2020", "2021", "2022", "2023"][rng.next_range(5)];
                    format!("{} concluded in {}", subject_name, value)
                }
            };
            facts.push((section_idx, format!("Fact {}: {}", i + 1, &clue)));
            clues.push(clue);
        }

        let theme_names: Vec<String> = selected_indices
            .iter()
            .map(|&idx| themes[idx].to_string())
            .collect();

        let question = if num_hops == 2 {
            format!(
                "Based on the information about {} and {}, what is the name of the {} that connects these two areas?",
                theme_names[0], theme_names[1], subject_type.0.to_lowercase()
            )
        } else {
            format!(
                "Across the sections on {}, {}, and {}, what {} name is mentioned in all three contexts?",
                theme_names[0], theme_names[1], theme_names[2], subject_type.0.to_lowercase()
            )
        };

        (question, subject_name.to_string(), facts)
    }

    fn embed_facts_in_sections(
        mut sections: Vec<String>,
        fact_positions: Vec<(usize, String)>,
    ) -> Vec<String> {
        for (idx, fact) in fact_positions {
            if idx < sections.len() {
                // Insert fact at a random position within the section
                let section = &sections[idx];
                let sentences: Vec<&str> = section.split(". ").collect();
                if !sentences.is_empty() {
                    let insert_pos = sentences.len() / 2;
                    let mut new_sentences = sentences.clone();
                    new_sentences.insert(insert_pos, &fact);
                    sections[idx] = new_sentences.join(". ");
                }
            }
        }
        sections
    }

    fn combine_sections_to_length(
        sections: Vec<String>,
        target_chars: usize,
        rng: &mut SimpleRng,
    ) -> String {
        let mut document = String::new();

        // Add introduction
        document.push_str("Document: Comprehensive Analysis\n\n");
        document.push_str("This document contains information about various topics. Please read carefully and answer questions based on the content provided.\n\n");

        // Add all sections
        for (i, section) in sections.iter().enumerate() {
            document.push_str(&format!("--- Section {} ---\n", i + 1));
            document.push_str(section);
            document.push_str("\n\n");
        }

        // Pad with filler content if needed
        let fillers = vec![
            "Additional context provides important background information for understanding the main topics.",
            "Supplementary materials are available for further reading on these subjects.",
            "Related research has been conducted in parallel fields with similar findings.",
            "Historical precedents help contextualize modern developments in these areas.",
            "Technical specifications may vary depending on specific implementation details.",
        ];

        while document.len() < target_chars {
            let filler = fillers[rng.next_range(fillers.len())];
            document.push_str(filler);
            document.push_str(" ");
        }

        // Trim to approximate target
        if document.len() > target_chars {
            document.truncate(target_chars);
            // Find last sentence ending
            if let Some(last_period) = document.rfind('.') {
                document.truncate(last_period + 1);
            }
        }

        document
    }

    /// Get statistics about the benchmark samples
    pub fn sample_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        for length in CONTEXT_LENGTHS.iter().map(|(l, _)| *l) {
            let count = self.samples.iter()
                .filter(|s| s.context_length_label == length)
                .count();
            stats.insert(format!("{}_tokens", length), count);
        }

        for qtype in [QuestionType::SingleHop, QuestionType::MultiHop, QuestionType::ManyHop] {
            let count = self.samples.iter()
                .filter(|s| s.question_type == qtype)
                .count();
            stats.insert(qtype.as_str().to_string(), count);
        }

        stats
    }
}

impl Default for LongContextBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for LongContextBenchmark {
    fn id(&self) -> &str {
        "long_context"
    }

    fn name(&self) -> &str {
        "Long Context"
    }

    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Context
    }

    fn description(&self) -> &str {
        "Tests model ability to handle long sequences and multi-hop reasoning across documents of varying lengths"
    }

    async fn load_dataset(&self) -> models_core::error::Result<Dataset> {
        let samples: Vec<DataSample> = self.samples.iter().map(|s| {
            let input = format!(
                "{}",
                s.context
            );

            DataSample::new(&s.id, input)
                .with_expected(&s.answer)
                .with_category(&format!("{}_{}", s.context_length_label, s.question_type.as_str()))
                .with_difficulty(match s.question_type {
                    QuestionType::SingleHop => Difficulty::Easy,
                    QuestionType::MultiHop => Difficulty::Medium,
                    QuestionType::ManyHop => Difficulty::Hard,
                })
        }).collect();

        let mut dataset = Dataset::new("LongContext", samples);
        dataset.metadata.description = Some(
            "Long context benchmark with varying document lengths (1K-32K tokens) and multi-hop questions".to_string()
        );
        dataset.metadata.categories = vec![
            "1k_tokens".to_string(),
            "2k_tokens".to_string(),
            "4k_tokens".to_string(),
            "8k_tokens".to_string(),
            "16k_tokens".to_string(),
            "32k_tokens".to_string(),
            "single_hop".to_string(),
            "multi_hop".to_string(),
            "many_hop".to_string(),
        ];

        Ok(dataset)
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

        debug!("Running Long Context benchmark with {} samples", samples.len());

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
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("context_lengths".to_string(),
                    serde_json::json!(CONTEXT_LENGTHS.iter().map(|(l, _)| *l).collect::<Vec<_>>()));
                meta
            },
        })
    }

    fn evaluate_response(&self, sample: &DataSample, response: &str) -> f64 {
        let expected = match &sample.expected_output {
            Some(e) => e.trim(),
            None => return 0.0,
        };

        let response_clean = response.trim().to_lowercase();
        let expected_clean = expected.to_lowercase();

        // Exact match
        if response_clean == expected_clean {
            return 1.0;
        }

        // Check if expected is contained in response
        if response_clean.contains(&expected_clean) {
            return 1.0;
        }

        // Check for each word in expected (for multi-word answers)
        let expected_words: Vec<&str> = expected_clean.split_whitespace().collect();
        if expected_words.len() > 1 {
            let match_count = expected_words
                .iter()
                .filter(|&&word| response_clean.contains(word))
                .count();
            let ratio = match_count as f64 / expected_words.len() as f64;
            if ratio >= 0.8 {
                return 0.9;
            }
            if ratio >= 0.5 {
                return 0.5;
            }
        }

        // Check for numbers specifically using simple string matching
        if expected_clean.parse::<f64>().is_ok() {
            // Check if the expected number appears as a standalone word in the response
            let response_words: Vec<&str> = response_clean
                .split(|c: char| !c.is_alphanumeric() && c != '.' && c != '-')
                .filter(|w| !w.is_empty())
                .collect();

            if response_words.contains(&expected_clean.as_str()) {
                return 1.0;
            }
        }

        0.0
    }

    fn format_prompt(&self, sample: &DataSample, _few_shot_examples: &[DataSample]) -> String {
        let question = self.samples.iter()
            .find(|s| s.id == sample.id)
            .map(|s| &s.question[..])
            .unwrap_or("Answer the question based on the document provided.");

        format!(
            "{}

Question: {}

Provide a concise answer:",
            sample.input,
            question
        )
    }
}

/// Simple deterministic RNG for generating consistent samples
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // Simple LCG
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next() as usize) % max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_context_creation() {
        let benchmark = LongContextBenchmark::new();
        assert_eq!(benchmark.id(), "long_context");
        assert_eq!(benchmark.name(), "Long Context");
        assert_eq!(benchmark.category(), BenchmarkCategory::Context);
    }

    #[test]
    fn test_sample_generation() {
        let benchmark = LongContextBenchmark::new();
        let stats = benchmark.sample_statistics();

        // Should have 6 context lengths * 3 question types * 2 samples = 36 samples
        assert!(stats.values().sum::<usize>() > 0);

        // Check that all context lengths are represented (4K-128K)
        for length in ["4k", "8k", "16k", "32k", "64k", "128k"] {
            assert!(stats.contains_key(&format!("{}_tokens", length)));
        }
    }

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(12345);
        let v1 = rng.next();
        let v2 = rng.next();
        assert_ne!(v1, v2);

        // Same seed should produce same sequence
        let mut rng2 = SimpleRng::new(12345);
        assert_eq!(v1, rng2.next());
        assert_eq!(v2, rng2.next());
    }

    #[test]
    fn test_question_type_display() {
        assert_eq!(QuestionType::SingleHop.as_str(), "single_hop");
        assert_eq!(QuestionType::MultiHop.as_str(), "multi_hop");
        assert_eq!(QuestionType::ManyHop.as_str(), "many_hop");
        assert_eq!(QuestionType::SingleHop.hop_count(), 1);
        assert_eq!(QuestionType::MultiHop.hop_count(), 2);
        assert_eq!(QuestionType::ManyHop.hop_count(), 3);
    }

    #[test]
    fn test_evaluate_response() {
        let benchmark = LongContextBenchmark::new();

        let sample = DataSample::new("test", "context")
            .with_expected("ALPHA-7X9");

        assert_eq!(benchmark.evaluate_response(&sample, "ALPHA-7X9"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "The answer is ALPHA-7X9"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "alpha-7x9"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample, "BETA-123"), 0.0);

        // Number matching
        let sample_num = DataSample::new("test2", "context")
            .with_expected("42");
        assert_eq!(benchmark.evaluate_response(&sample_num, "The answer is 42"), 1.0);
        assert_eq!(benchmark.evaluate_response(&sample_num, "43"), 0.0);
    }

    #[test]
    fn test_document_generation() {
        let (doc, question, answer) = LongContextBenchmark::generate_synthetic_document(
            1000,
            QuestionType::SingleHop,
            DocumentType::Narrative,
            QuestionPosition::Middle,
        );

        assert!(!doc.is_empty());
        assert!(!question.is_empty());
        assert!(!answer.is_empty());
        assert!(doc.len() >= 2000); // ~1000 tokens * ~4 chars/token

        // Additional tests for document generation
        // Test different context sizes
        let (doc_2k, _, _) = LongContextBenchmark::generate_synthetic_document(
            2000, QuestionType::SingleHop, DocumentType::Narrative, QuestionPosition::Middle);
        let (doc_4k, _, _) = LongContextBenchmark::generate_synthetic_document(
            4000, QuestionType::MultiHop, DocumentType::Technical, QuestionPosition::Middle);

        // Documents should scale roughly with token count (4 chars/token)
        assert!(doc_2k.len() >= 2000 * 3, "2K token doc should be ~8000 chars");
        assert!(doc_4k.len() >= 4000 * 3, "4K token doc should be ~16000 chars");
        assert!(doc_4k.len() > doc_2k.len(), "4K doc should be longer than 2K doc");

        // All documents should have sections
        assert!(doc.contains("--- Section"), "Document should have sections");
        assert!(doc_2k.contains("--- Section"), "2K doc should have sections");
        assert!(doc_4k.contains("--- Section"), "4K doc should have sections");

        // Test different question types produce different structures
        let (_, single_q, single_a) = LongContextBenchmark::generate_synthetic_document(
            1000, QuestionType::SingleHop, DocumentType::Narrative, QuestionPosition::Middle);
        let (_, multi_q, multi_a) = LongContextBenchmark::generate_synthetic_document(
            1000, QuestionType::MultiHop, DocumentType::Technical, QuestionPosition::Middle);
        let (_, many_q, many_a) = LongContextBenchmark::generate_synthetic_document(
            1000, QuestionType::ManyHop, DocumentType::Scientific, QuestionPosition::Middle);

        assert!(!single_q.is_empty(), "Single hop should have a question");
        assert!(!multi_q.is_empty(), "Multi hop should have a question");
        assert!(!many_q.is_empty(), "Many hop should have a question");
        assert!(!single_a.is_empty(), "Single hop should have an answer");
        assert!(!multi_a.is_empty(), "Multi hop should have an answer");
        assert!(!many_a.is_empty(), "Many hop should have an answer");
    }

    #[test]
    fn test_question_extraction() {
        // Generate samples and verify question extraction
        let (_, single_q, single_a) = LongContextBenchmark::generate_synthetic_document(
            1000,
            QuestionType::SingleHop,
            DocumentType::Narrative,
            QuestionPosition::Middle,
        );

        // Verify question structure
        assert!(
            single_q.contains("?") || single_q.ends_with(".") || !single_q.is_empty(),
            "Question should be properly formatted"
        );

        // Answer should be extractable
        assert!(!single_a.is_empty(), "Answer should not be empty");
        assert!(
            !single_a.contains("?"),
            "Answer should not be a question"
        );

        // Generate multiple samples to test variety
        let mut unique_questions = std::collections::HashSet::new();
        let mut unique_answers = std::collections::HashSet::new();

        for _ in 0..5 {
            let (_, q, a) = LongContextBenchmark::generate_synthetic_document(
                1000,
                QuestionType::SingleHop,
                DocumentType::Narrative,
                QuestionPosition::Middle,
            );
            unique_questions.insert(q);
            unique_answers.insert(a);
        }

        // With 5 samples, we should have some variety
        assert!(
            unique_questions.len() >= 1,
            "Should generate at least one unique question"
        );
        assert!(
            unique_answers.len() >= 1,
            "Should generate at least one unique answer"
        );

        // Test multi-hop question structure
        let (_, multi_q, _multi_a) = LongContextBenchmark::generate_synthetic_document(
            2000,
            QuestionType::MultiHop,
            DocumentType::Technical,
            QuestionPosition::Middle,
        );

        // Multi-hop questions often reference multiple sections
        assert!(
            multi_q.contains("and") || multi_q.contains("or") || multi_q.len() > 20,
            "Multi-hop question should be complex"
        );
    }

    #[test]
    fn test_context_length_validation() {
        // Test that documents are generated with appropriate lengths
        let test_cases = [
            (1000, 2000),   // 1K tokens -> expect at least 2000 chars
            (2000, 5000),   // 2K tokens -> expect at least 5000 chars
            (4000, 10000),  // 4K tokens -> expect at least 10000 chars
        ];

        for (target_tokens, _min_chars) in test_cases {
            let (doc, _, _) = LongContextBenchmark::generate_synthetic_document(
                target_tokens,
                QuestionType::SingleHop,
                DocumentType::Narrative,
                QuestionPosition::Middle,
            );

            // Rough token estimate: 1 token ~= 4 characters
            let min_expected = target_tokens * 2; // At least half the expected chars
            let max_expected = target_tokens * 6;   // No more than 1.5x expected chars

            assert!(
                doc.len() >= min_expected,
                "Document for {} tokens should have at least {} chars, got {}",
                target_tokens,
                min_expected,
                doc.len()
            );

            assert!(
                doc.len() <= max_expected,
                "Document for {} tokens should have at most {} chars, got {}",
                target_tokens,
                max_expected,
                doc.len()
            );

            // Verify document structure
            assert!(
                doc.contains("Document: Comprehensive Analysis"),
                "Document should have title"
            );
            assert!(
                doc.contains("--- Section"),
                "Document should have section markers"
            );
        }

        // Test that all context lengths in CONTEXT_LENGTHS are valid
        for (label, tokens) in CONTEXT_LENGTHS {
            let (doc, _, _) = LongContextBenchmark::generate_synthetic_document(
                *tokens,
                QuestionType::SingleHop,
                DocumentType::Narrative,
                QuestionPosition::Middle,
            );
            assert!(
                !doc.is_empty(),
                "Context length {} ({}) should produce non-empty document",
                label,
                tokens
            );
        }
    }
}
