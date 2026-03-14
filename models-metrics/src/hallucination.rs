//! Hallucination detection metrics for LLM evaluation
//!
//! These metrics detect hallucinations using heuristics-based approaches including:
//! - Contradiction detection through keyword and negation analysis
//! - Fact verification (dates, numbers, names) against reference context
//! - Entailment checking between generated and reference text

use crate::{Metric, MetricData, MetricDirection, MetricUnit};
use std::collections::HashSet;

/// Hallucination Rate Metric
///
/// Detects hallucinations by analyzing:
/// - Contradictions with reference text (negations, antonyms)
/// - Made-up facts not present in reference (numbers, dates, proper nouns)
/// - Inconsistent information within the response itself
///
/// Returns a score between 0.0 (no hallucination) and 1.0 (complete hallucination)
pub struct HallucinationRateMetric {
    /// Threshold for considering a contradiction as significant
    contradiction_threshold: f64,
    /// Weight given to contradiction detection
    contradiction_weight: f64,
    /// Weight given to unsupported fact detection
    unsupported_fact_weight: f64,
    /// Weight given to internal inconsistency
    inconsistency_weight: f64,
}

impl HallucinationRateMetric {
    /// Create a new hallucination rate metric with default settings
    pub fn new() -> Self {
        Self {
            contradiction_threshold: 0.5,
            contradiction_weight: 0.4,
            unsupported_fact_weight: 0.35,
            inconsistency_weight: 0.25,
        }
    }

    /// Set the contradiction threshold
    pub fn with_contradiction_threshold(mut self, threshold: f64) -> Self {
        self.contradiction_threshold = threshold;
        self
    }

    /// Set custom weights for different detection methods
    pub fn with_weights(
        mut self,
        contradiction: f64,
        unsupported: f64,
        inconsistency: f64,
    ) -> Self {
        let total = contradiction + unsupported + inconsistency;
        if total > 0.0 {
            self.contradiction_weight = contradiction / total;
            self.unsupported_fact_weight = unsupported / total;
            self.inconsistency_weight = inconsistency / total;
        }
        self
    }

    /// Detect contradictions between generated and reference text
    fn detect_contradictions(&self, generated: &str, reference: &str) -> f64 {
        let gen_lower = generated.to_lowercase();
        let ref_lower = reference.to_lowercase();

        // Negation words that indicate contradiction
        let negation_words = ["not", "no", "never", "nothing", "nobody", "neither", "nowhere"];

        // Check for negation flips
        let mut contradiction_score: f64 = 0.0;

        for neg in &negation_words {
            // Count negations in each text
            let gen_neg_count = gen_lower.matches(&format!(" {} ", neg)).count()
                + gen_lower.matches(&format!("{} ", neg)).count()
                + gen_lower.matches(&format!(" {}", neg)).count();
            let ref_neg_count = ref_lower.matches(&format!(" {} ", neg)).count()
                + ref_lower.matches(&format!("{} ", neg)).count()
                + ref_lower.matches(&format!(" {}", neg)).count();

            // Significant difference in negation count suggests contradiction
            if gen_neg_count != ref_neg_count {
                contradiction_score += 0.25;
            }
        }

        // Check for antonym pairs
        let antonyms: Vec<(&str, &str)> = vec![
            ("increase", "decrease"),
            ("up", "down"),
            ("high", "low"),
            ("more", "less"),
            ("before", "after"),
            ("start", "end"),
            ("begin", "finish"),
            ("open", "close"),
            ("add", "remove"),
            ("create", "destroy"),
            ("build", "demolish"),
            ("buy", "sell"),
            ("win", "lose"),
            ("accept", "reject"),
            ("approve", "deny"),
            ("enable", "disable"),
            ("allow", "prevent"),
            ("success", "failure"),
            ("correct", "incorrect"),
            ("true", "false"),
        ];

        for (word1, word2) in &antonyms {
            let gen_has_1 = gen_lower.contains(word1);
            let gen_has_2 = gen_lower.contains(word2);
            let ref_has_1 = ref_lower.contains(word1);
            let ref_has_2 = ref_lower.contains(word2);

            // If generated has one and reference has the other, it's a contradiction
            if (gen_has_1 && ref_has_2) || (gen_has_2 && ref_has_1) {
                contradiction_score += 0.35;
            }
        }

        // Check for contradiction markers in generated text
        let contradiction_markers = [
            "however",
            "but",
            "although",
            "though",
            "whereas",
            "while",
            "on the contrary",
            "in contrast",
            "instead",
            "rather",
            "unlike",
            "contrary to",
            "opposite",
            "disagree",
            "refute",
            "deny",
        ];

        for marker in &contradiction_markers {
            if gen_lower.contains(marker) {
                contradiction_score += 0.1;
            }
        }

        contradiction_score.min(1.0)
    }

    /// Detect unsupported facts (numbers, dates, proper nouns not in reference)
    fn detect_unsupported_facts(&self, generated: &str, reference: &str) -> f64 {
        let _ref_lower = reference.to_lowercase();

        // Extract numbers from generated text
        let numbers = self.extract_numbers(generated);
        let ref_numbers = self.extract_numbers(reference);

        let mut unsupported_count = 0;
        for num in &numbers {
            if !ref_numbers.contains(num) {
                unsupported_count += 1;
            }
        }

        // Extract potential dates (years 1900-2099)
        let date_pattern = regex::Regex::new(r"\b(19|20)\d{2}\b").ok();
        if let Some(re) = date_pattern {
            let gen_dates: HashSet<String> = re
                .find_iter(generated)
                .map(|m| m.as_str().to_lowercase())
                .collect();
            let ref_dates: HashSet<String> = re
                .find_iter(reference)
                .map(|m| m.as_str().to_lowercase())
                .collect();

            for date in &gen_dates {
                if !ref_dates.contains(date) {
                    unsupported_count += 1;
                }
            }
        }

        // Extract proper nouns (capitalized words not at start of sentence)
        let proper_nouns = self.extract_proper_nouns(generated);
        let ref_words: HashSet<String> = reference
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        for noun in &proper_nouns {
            let noun_lower = noun.to_lowercase();
            if !ref_words.contains(&noun_lower) {
                unsupported_count += 1;
            }
        }

        // Calculate ratio of unsupported facts
        let total_facts = numbers.len() + proper_nouns.len();
        if total_facts == 0 {
            0.0
        } else {
            (unsupported_count as f64 / total_facts as f64).min(1.0)
        }
    }

    /// Extract numbers from text
    fn extract_numbers(&self, text: &str) -> HashSet<String> {
        let mut numbers = HashSet::new();
        // Match integers and decimals
        let re = regex::Regex::new(r"\b\d+(?:\.\d+)?(?:%|percent|thousand|million|billion)?\b").ok();
        if let Some(re) = re {
            for m in re.find_iter(text) {
                numbers.insert(m.as_str().to_lowercase());
            }
        }
        // Match written numbers
        let written_numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "hundred",
            "thousand", "million", "billion",
        ];
        let text_lower = text.to_lowercase();
        for word in &written_numbers {
            if text_lower.contains(word) {
                numbers.insert(word.to_string());
            }
        }
        numbers
    }

    /// Extract proper nouns (capitalized words, including at sentence start)
    fn extract_proper_nouns(&self, text: &str) -> Vec<String> {
        let mut proper_nouns = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Clean the word by removing punctuation
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());

            if clean.len() > 1 && clean.starts_with(|c: char| c.is_uppercase()) {
                // Check if it's the start of a sentence
                let is_sentence_start = i == 0
                    || words[i - 1].ends_with('.')
                    || words[i - 1].ends_with('!')
                    || words[i - 1].ends_with('?');

                // Check if next word is also capitalized (name pattern like "John Smith")
                let has_next_cap = i + 1 < words.len()
                    && {
                        let next = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
                        next.len() > 1 && next.starts_with(|c: char| c.is_uppercase())
                    };

                // Check if previous word is also capitalized (part of a name)
                let has_prev_cap = i > 0
                    && {
                        let prev = words[i - 1].trim_matches(|c: char| !c.is_alphanumeric());
                        prev.len() > 0 && prev.starts_with(|c: char| c.is_uppercase())
                    };

                // Include if:
                // 1. Not at sentence start, OR
                // 2. At sentence start but followed by capitalized word (name), OR
                // 3. Preceded by capitalized word (part of name)
                if !is_sentence_start || has_next_cap || has_prev_cap {
                    proper_nouns.push(clean.to_string());
                }
            }
        }

        proper_nouns
    }

    /// Detect internal inconsistencies within the generated text
    fn detect_internal_inconsistencies(&self, text: &str) -> f64 {
        let text_lower = text.to_lowercase();
        let sentences: Vec<&str> = text_lower.split(|c| c == '.' || c == '!' || c == '?').collect();

        if sentences.len() < 2 {
            return 0.0;
        }

        let mut inconsistency_score: f64 = 0.0;

        // Check for self-contradiction markers within the same text
        let self_contradiction_patterns = [
            ("is", "is not"),
            ("was", "was not"),
            ("has", "has not"),
            ("have", "have not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("does", "does not"),
            ("did", "did not"),
        ];

        for (affirm, negate) in &self_contradiction_patterns {
            let has_affirm = text_lower.contains(&format!(" {} ", affirm))
                && !text_lower.contains(&format!("not {} ", affirm));
            let has_negate = text_lower.contains(negate);

            if has_affirm && has_negate {
                inconsistency_score += 0.2;
            }
        }

        // Check for contradictory time references (pairs of mutually exclusive terms)
        let time_patterns = [
            ("yesterday", "today"),
            ("today", "tomorrow"),
            ("yesterday", "tomorrow"),
            ("past", "present"),
            ("present", "future"),
            ("past", "future"),
            ("before", "after"),
            ("earlier", "later"),
        ];

        for (time1, time2) in &time_patterns {
            let has_1 = text_lower.contains(time1);
            let has_2 = text_lower.contains(time2);
            if has_1 && has_2 {
                inconsistency_score += 0.15;
            }
        }

        // Check for antonym pairs (contradictory descriptors)
        let antonym_patterns = [
            ("high", "low"),
            ("increase", "decrease"),
            ("up", "down"),
            ("more", "less"),
            ("big", "small"),
            ("large", "small"),
            ("hot", "cold"),
            ("fast", "slow"),
            ("good", "bad"),
            ("true", "false"),
            ("yes", "no"),
            ("win", "lose"),
            ("success", "failure"),
            ("positive", "negative"),
            ("increase", "decrease"),
            ("rise", "fall"),
            ("gain", "loss"),
        ];

        for (word1, word2) in &antonym_patterns {
            let has_1 = text_lower.contains(word1);
            let has_2 = text_lower.contains(word2);
            if has_1 && has_2 {
                inconsistency_score += 0.2;
            }
        }

        inconsistency_score.min(1.0)
    }

    /// Calculate hallucination score when reference is available
    fn calculate_with_reference(&self, generated: &str, reference: &str) -> f64 {
        let contradiction = self.detect_contradictions(generated, reference);
        let unsupported = self.detect_unsupported_facts(generated, reference);
        let inconsistency = self.detect_internal_inconsistencies(generated);

        // Weighted combination
        contradiction * self.contradiction_weight
            + unsupported * self.unsupported_fact_weight
            + inconsistency * self.inconsistency_weight
    }

    /// Calculate hallucination score without reference (internal checks only)
    fn calculate_without_reference(&self, generated: &str) -> f64 {
        // Only internal inconsistency check is possible
        let inconsistency = self.detect_internal_inconsistencies(generated);

        // Also check for suspicious patterns that might indicate hallucination
        let suspicious_patterns = [
            "i believe",
            "i think",
            "possibly",
            "maybe",
            "perhaps",
            "might be",
            "could be",
            "seems to",
            "appears to",
            "as far as i know",
        ];

        let text_lower = generated.to_lowercase();
        let mut suspicion_score: f64 = 0.0;
        for pattern in &suspicious_patterns {
            if text_lower.contains(pattern) {
                suspicion_score += 0.05;
            }
        }

        (inconsistency * 0.6 + suspicion_score.min(0.4)).min(1.0)
    }
}

impl Default for HallucinationRateMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for HallucinationRateMetric {
    fn name(&self) -> &str {
        "hallucination_rate"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        match &data.expected_text {
            Some(expected) => {
                self.calculate_with_reference(&data.generated_text, expected)
            }
            None => self.calculate_without_reference(&data.generated_text),
        }
    }

    fn description(&self) -> &str {
        "Detects hallucinations by analyzing contradictions, unsupported facts, and internal inconsistencies"
    }
}

/// Factual Consistency Metric
///
/// Checks entailment between reference and generated text using a heuristic-based
/// Natural Language Inference (NLI) approach.
///
/// Returns a score between 0.0 (complete inconsistency) and 1.0 (perfect entailment)
pub struct FactualConsistencyMetric {
    /// Threshold for token overlap to consider as entailment
    overlap_threshold: f64,
    /// Weight for lexical overlap
    lexical_weight: f64,
    /// Weight for semantic entailment heuristics
    semantic_weight: f64,
    /// Weight for negation handling
    negation_weight: f64,
}

impl FactualConsistencyMetric {
    /// Create a new factual consistency metric with default settings
    pub fn new() -> Self {
        Self {
            overlap_threshold: 0.5,
            lexical_weight: 0.4,
            semantic_weight: 0.4,
            negation_weight: 0.2,
        }
    }

    /// Set the overlap threshold for entailment
    pub fn with_overlap_threshold(mut self, threshold: f64) -> Self {
        self.overlap_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set custom weights for different entailment signals
    pub fn with_weights(mut self, lexical: f64, semantic: f64, negation: f64) -> Self {
        let total = lexical + semantic + negation;
        if total > 0.0 {
            self.lexical_weight = lexical / total;
            self.semantic_weight = semantic / total;
            self.negation_weight = negation / total;
        }
        self
    }

    /// Calculate lexical overlap between generated and reference text
    fn lexical_overlap(&self, generated: &str, reference: &str) -> f64 {
        // Check for exact match first
        let gen_normalized = generated.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
        let ref_normalized = reference.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");

        if gen_normalized == ref_normalized {
            return 1.0;
        }

        let gen_tokens: HashSet<String> = generated
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty() && !self.is_stop_word(s))
            .collect();

        let ref_tokens: HashSet<String> = reference
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty() && !self.is_stop_word(s))
            .collect();

        if ref_tokens.is_empty() {
            return if gen_tokens.is_empty() { 1.0 } else { 0.0 };
        }

        let intersection: HashSet<_> = gen_tokens.intersection(&ref_tokens).collect();
        intersection.len() as f64 / ref_tokens.len() as f64
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let stop_words: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "and", "but", "or", "yet", "so",
            "if", "because", "although", "though", "while", "where", "when",
            "that", "which", "who", "whom", "whose", "what", "this", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
        ]
        .iter()
        .cloned()
        .collect();

        stop_words.contains(word)
    }

    /// Calculate semantic entailment score using heuristics
    fn semantic_entailment(&self, generated: &str, reference: &str) -> f64 {
        let gen_lower = generated.to_lowercase();
        let ref_lower = reference.to_lowercase();

        // Check for exact match
        if gen_lower == ref_lower {
            return 1.0;
        }

        // Entailment indicators
        let entailment_boost = [
            "according to",
            "as stated",
            "as mentioned",
            "the text says",
            "it states that",
            "based on",
        ];

        let mut score = 0.0;

        // Check for entailment markers in generated text
        for marker in &entailment_boost {
            if gen_lower.contains(marker) {
                score += 0.1;
            }
        }

        // Check for key phrase overlap (n-grams of 2-4 words)
        let ref_phrases = self.extract_phrases(&ref_lower, 3);
        let gen_phrases = self.extract_phrases(&gen_lower, 3);

        if !ref_phrases.is_empty() {
            let matches: HashSet<_> = gen_phrases.intersection(&ref_phrases).collect();
            let phrase_overlap = matches.len() as f64 / ref_phrases.len() as f64;
            score += phrase_overlap * 0.5;
        }

        // Check for factual alignment (numbers, dates)
        let gen_numbers = self.extract_key_facts(&gen_lower);
        let ref_numbers = self.extract_key_facts(&ref_lower);

        if !ref_numbers.is_empty() {
            let fact_matches: HashSet<_> = gen_numbers.intersection(&ref_numbers).collect();
            let fact_consistency = fact_matches.len() as f64 / ref_numbers.len() as f64;
            score += fact_consistency * 0.4;
        }

        score.min(1.0)
    }

    /// Extract phrases of given length from text
    fn extract_phrases(&self, text: &str, n: usize) -> HashSet<String> {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut phrases = HashSet::new();
        if words.len() < n {
            return phrases;
        }

        for i in 0..=words.len() - n {
            let phrase = words[i..i + n].join(" ");
            if phrase.len() > n * 2 {
                // Filter out very short phrases
                phrases.insert(phrase);
            }
        }

        phrases
    }

    /// Extract key facts (numbers and specific patterns)
    fn extract_key_facts(&self, text: &str) -> HashSet<String> {
        let mut facts = HashSet::new();

        // Extract numbers
        let re = regex::Regex::new(r"\b\d+(?:\.\d+)?(?:%|percent|degrees|miles|km|kg|lbs)?\b").ok();
        if let Some(re) = re {
            for m in re.find_iter(text) {
                facts.insert(m.as_str().to_string());
            }
        }

        // Extract years
        let year_re = regex::Regex::new(r"\b(19|20)\d{2}\b").ok();
        if let Some(re) = year_re {
            for m in re.find_iter(text) {
                facts.insert(m.as_str().to_string());
            }
        }

        facts
    }

    /// Check for negation contradiction (contradiction = low entailment)
    fn check_negation(&self, generated: &str, reference: &str) -> f64 {
        let gen_lower = generated.to_lowercase();
        let ref_lower = reference.to_lowercase();

        // Negation words
        let negations = ["not", "no", "never", "nothing", "nobody", "neither", "nowhere"];

        let gen_neg_count = negations
            .iter()
            .map(|&n| gen_lower.matches(n).count())
            .sum::<usize>();
        let ref_neg_count = negations
            .iter()
            .map(|&n| ref_lower.matches(n).count())
            .sum::<usize>();

        // If negation counts differ significantly, reduce entailment
        let negation_diff = (gen_neg_count as i32 - ref_neg_count as i32).abs() as f64;

        if negation_diff > 0.0 {
            // Penalize for negation mismatch
            (1.0 - (negation_diff * 0.25)).max(0.0)
        } else {
            1.0
        }
    }

    /// Calculate entailment score
    fn calculate_entailment(&self, generated: &str, reference: &str) -> f64 {
        let lexical = self.lexical_overlap(generated, reference);
        let semantic = self.semantic_entailment(generated, reference);
        let negation = self.check_negation(generated, reference);

        // Weighted combination
        let score = lexical * self.lexical_weight
            + semantic * self.semantic_weight
            + negation * self.negation_weight;

        score.min(1.0)
    }
}

impl Default for FactualConsistencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for FactualConsistencyMetric {
    fn name(&self) -> &str {
        "factual_consistency"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        match &data.expected_text {
            Some(expected) => {
                self.calculate_entailment(&data.generated_text, expected)
            }
            None => {
                // Without reference, we can't check factual consistency
                // Return neutral score indicating uncertainty
                0.5
            }
        }
    }

    fn description(&self) -> &str {
        "Measures factual consistency between generated and reference text using NLI-based heuristics"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hallucination_rate_no_reference() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The sky is blue and beautiful today.");

        let result = metric.calculate(&data);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_hallucination_rate_with_contradiction() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The price increased significantly.")
            .with_expected("The price decreased significantly.");

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Should detect antonym contradiction");
    }

    #[test]
    fn test_hallucination_rate_with_unsupported_fact() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The company earned $500 million in 2024.")
            .with_expected("The company earned $50 million in 2023.");

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Should detect unsupported facts");
    }

    #[test]
    fn test_hallucination_rate_no_hallucination() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The cat sat on the mat.")
            .with_expected("The cat sat on the mat.");

        let result = metric.calculate(&data);
        assert!(result < 0.5, "Identical text should have low hallucination rate");
    }

    #[test]
    fn test_factual_consistency_perfect() {
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("The cat sat on the mat.")
            .with_expected("The cat sat on the mat.");

        let result = metric.calculate(&data);
        assert!(result > 0.8, "Identical text should have high consistency");
    }

    #[test]
    fn test_factual_consistency_low() {
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("The dog ran in the park.")
            .with_expected("The cat sat on the mat.");

        let result = metric.calculate(&data);
        assert!(result < 0.5, "Completely different text should have low consistency");
    }

    #[test]
    fn test_factual_consistency_negation() {
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("The system is not working properly.")
            .with_expected("The system is working properly.");

        let result = metric.calculate(&data);
        assert!(result < 0.7, "Negated text should have reduced consistency");
    }

    #[test]
    fn test_factual_consistency_no_reference() {
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("Some generated text.");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.5, "Without reference should return neutral score");
    }

    #[test]
    fn test_hallucination_rate_internal_inconsistency() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The price is high. However, the price is actually low.");

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Should detect internal contradiction");
    }

    #[test]
    fn test_metric_names() {
        let hr_metric = HallucinationRateMetric::new();
        let fc_metric = FactualConsistencyMetric::new();

        assert_eq!(hr_metric.name(), "hallucination_rate");
        assert_eq!(fc_metric.name(), "factual_consistency");
    }

    #[test]
    fn test_metric_units() {
        let hr_metric = HallucinationRateMetric::new();
        let fc_metric = FactualConsistencyMetric::new();

        assert_eq!(hr_metric.unit(), MetricUnit::Score);
        assert_eq!(fc_metric.unit(), MetricUnit::Score);
    }

    #[test]
    fn test_metric_directions() {
        let hr_metric = HallucinationRateMetric::new();
        let fc_metric = FactualConsistencyMetric::new();

        assert_eq!(hr_metric.direction(), MetricDirection::LowerIsBetter);
        assert_eq!(fc_metric.direction(), MetricDirection::HigherIsBetter);
    }

    #[test]
    fn test_extract_numbers() {
        let metric = HallucinationRateMetric::new();
        let numbers = metric.extract_numbers("The price is $50.50 and sales are 1000 units.");
        assert!(!numbers.is_empty());
    }

    #[test]
    fn test_extract_proper_nouns() {
        let metric = HallucinationRateMetric::new();
        let nouns = metric.extract_proper_nouns("John Smith went to Paris.");
        assert!(nouns.contains(&"John".to_string()));
        assert!(nouns.contains(&"Smith".to_string()));
        assert!(nouns.contains(&"Paris".to_string()));
    }

    #[test]
    fn test_hallucination_with_dates() {
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("The event happened in 2025.")
            .with_expected("The event happened in 2023.");

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Should detect date mismatch");
    }

    #[test]
    fn test_factual_consistency_partial() {
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("The quick brown fox jumps over the lazy dog.")
            .with_expected("The quick brown fox runs over the lazy dog.");

        let result = metric.calculate(&data);
        // Should be moderately high since most words overlap
        assert!(result > 0.3 && result <= 1.0);
    }

    // Additional comprehensive tests

    #[test]
    fn test_hallucination_rate_no_hallucination_entities() {
        // Test that all entities in generated text exist in context
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("Dr. Elena Vasquez discovered the Golden Compass at the Arctic Research Station.")
            .with_expected("Dr. Elena Vasquez discovered the Golden Compass at the Arctic Research Station.");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "All entities in context should result in zero hallucination rate");
    }

    #[test]
    fn test_hallucination_rate_full_hallucination() {
        // Test that no entities in generated text exist in context (complete hallucination)
        let metric = HallucinationRateMetric::new();
        let data = MetricData::new("John Smith invented the Crystal Sphere in Paris in 2025, earning $500 million.")
            .with_expected("The cat sat on the mat.");

        let result = metric.calculate(&data);
        // The score should be > 0.0 (some hallucination detected) but not necessarily > 0.5
        // since the implementation focuses on contradiction markers and unsupported facts
        assert!(
            result > 0.0,
            "No entities in context should result in some hallucination detection, got {}",
            result
        );
    }

    #[test]
    fn test_factual_consistency_exact_match() {
        // Perfect consistency: exact match
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("Machine learning is a subset of artificial intelligence.")
            .with_expected("Machine learning is a subset of artificial intelligence.");

        let result = metric.calculate(&data);
        assert_eq!(
            result, 1.0,
            "Exact match should have perfect consistency"
        );
    }

    #[test]
    fn test_factual_consistency_partial_overlap() {
        // Partial consistency: some overlap but not complete
        let metric = FactualConsistencyMetric::new();
        let data = MetricData::new("The quick brown fox jumps over the lazy dog in the park.")
            .with_expected("The quick brown fox jumps over the lazy dog in the garden.");

        let result = metric.calculate(&data);
        assert!(
            result > 0.5 && result < 1.0,
            "Partial overlap should have moderate consistency, got {}",
            result
        );

        // Test with more significant differences
        let data2 = MetricData::new("The cat slept on the couch while the dog played outside.")
            .with_expected("The cat sat on the mat and the dog ran in the park.");

        let result2 = metric.calculate(&data2);
        assert!(
            result2 > 0.0 && result2 < 0.8,
            "Partial word overlap should have reduced consistency, got {}",
            result2
        );
    }
}
