//! Accuracy metrics for LLM evaluation

use crate::{Metric, MetricData, MetricDirection, MetricUnit};
use std::collections::HashMap;

/// Exact match metric
pub struct ExactMatchMetric {
    case_sensitive: bool,
    normalize_whitespace: bool,
}

impl ExactMatchMetric {
    /// Create a new exact match metric with default settings
    pub fn new() -> Self {
        Self {
            case_sensitive: false,
            normalize_whitespace: true,
        }
    }

    /// Create with case sensitivity
    pub fn case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    /// Create with whitespace normalization
    pub fn normalize_whitespace(mut self, normalize: bool) -> Self {
        self.normalize_whitespace = normalize;
        self
    }

    fn normalize(&self, text: &str) -> String {
        let mut result = text.to_string();
        if self.normalize_whitespace {
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        if !self.case_sensitive {
            result = result.to_lowercase();
        }
        result.trim().to_string()
    }
}

impl Default for ExactMatchMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for ExactMatchMetric {
    fn name(&self) -> &str {
        "exact_match"
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
                let normalized_generated = self.normalize(&data.generated_text);
                let normalized_expected = self.normalize(expected);
                if normalized_generated == normalized_expected {
                    1.0
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    fn description(&self) -> &str {
        "Binary score indicating if generated text exactly matches expected text"
    }
}

/// F1 score metric (token-level)
pub struct F1ScoreMetric {
    case_sensitive: bool,
}

impl F1ScoreMetric {
    pub fn new() -> Self {
        Self {
            case_sensitive: false,
        }
    }

    pub fn case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        text.split_whitespace()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn calculate_f1(&self, generated: &[String], expected: &[String]) -> f64 {
        if generated.is_empty() && expected.is_empty() {
            return 1.0;
        }
        if generated.is_empty() || expected.is_empty() {
            return 0.0;
        }

        // Count occurrences
        let mut generated_counts: HashMap<&str, usize> = HashMap::new();
        for token in generated {
            *generated_counts.entry(token.as_str()).or_insert(0) += 1;
        }

        let mut expected_counts: HashMap<&str, usize> = HashMap::new();
        for token in expected {
            *expected_counts.entry(token.as_str()).or_insert(0) += 1;
        }

        // Calculate true positives
        let mut true_positives = 0usize;
        for (token, count) in &expected_counts {
            let gen_count = generated_counts.get(token).copied().unwrap_or(0);
            true_positives += count.min(&gen_count);
        }

        // Precision: TP / (TP + FP) = TP / generated_tokens
        let precision = if generated.is_empty() {
            0.0
        } else {
            true_positives as f64 / generated.len() as f64
        };

        // Recall: TP / (TP + FN) = TP / expected_tokens
        let recall = if expected.is_empty() {
            0.0
        } else {
            true_positives as f64 / expected.len() as f64
        };

        // F1 = 2 * precision * recall / (precision + recall)
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

impl Default for F1ScoreMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for F1ScoreMetric {
    fn name(&self) -> &str {
        "f1_score"
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
                let generated_tokens = self.tokenize(&data.generated_text);
                let expected_tokens = self.tokenize(expected);
                self.calculate_f1(&generated_tokens, &expected_tokens)
            }
            None => 0.0,
        }
    }

    fn description(&self) -> &str {
        "Token-level F1 score between generated and expected text"
    }
}

// ============================================================================
// BLEU Score Metric
// ============================================================================

/// BLEU (Bilingual Evaluation Understudy) score metric
pub struct BleuMetric {
    max_n: usize,
}

impl BleuMetric {
    /// Create a new BLEU metric with default settings (BLEU-4)
    pub fn new() -> Self {
        Self { max_n: 4 }
    }

    /// Create BLEU-N metric
    pub fn with_max_n(max_n: usize) -> Self {
        Self { max_n }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn get_ngrams(&self, tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
        let mut ngrams = HashMap::new();
        if tokens.len() < n {
            return ngrams;
        }
        for i in 0..=tokens.len() - n {
            let ngram: Vec<String> = tokens[i..i + n].to_vec();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }
        ngrams
    }

    fn calculate_bleu(&self, generated: &[String], expected: &[String]) -> f64 {
        if generated.is_empty() || expected.is_empty() {
            return 0.0;
        }

        // Calculate brevity penalty
        let bp = if generated.len() >= expected.len() {
            1.0
        } else {
            (1.0 - expected.len() as f64 / generated.len() as f64).exp()
        };

        // Calculate n-gram precisions
        let mut precisions = Vec::new();
        for n in 1..=self.max_n {
            let gen_ngrams = self.get_ngrams(generated, n);
            let exp_ngrams = self.get_ngrams(expected, n);

            if gen_ngrams.is_empty() || exp_ngrams.is_empty() {
                precisions.push(0.0);
                continue;
            }

            let mut clipped_count = 0usize;
            let mut total_count = 0usize;

            for (ngram, count) in &gen_ngrams {
                let max_ref_count = exp_ngrams.get(ngram).copied().unwrap_or(0);
                clipped_count += count.min(&max_ref_count);
                total_count += count;
            }

            let precision = if total_count > 0 {
                clipped_count as f64 / total_count as f64
            } else {
                0.0
            };
            precisions.push(precision);
        }

        // Calculate geometric mean of precisions
        let log_precisions: f64 = precisions.iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p.ln())
            .sum();

        let num_non_zero = precisions.iter().filter(|&&p| p > 0.0).count();

        if num_non_zero == 0 {
            return 0.0;
        }

        let avg_log_precision = log_precisions / num_non_zero as f64;
        bp * avg_log_precision.exp()
    }
}

impl Default for BleuMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for BleuMetric {
    fn name(&self) -> &str {
        "bleu"
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
                let generated_tokens = self.tokenize(&data.generated_text);
                let expected_tokens = self.tokenize(expected);
                self.calculate_bleu(&generated_tokens, &expected_tokens)
            }
            None => 0.0,
        }
    }

    fn description(&self) -> &str {
        "BLEU score for machine translation quality evaluation"
    }
}

// ============================================================================
// ROUGE Score Metric
// ============================================================================

/// ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score metric
pub struct RougeMetric {
    rouge_type: RougeType,
}

/// Type of ROUGE metric
#[derive(Debug, Clone, Copy)]
pub enum RougeType {
    /// ROUGE-1: Unigram overlap
    Rouge1,
    /// ROUGE-2: Bigram overlap
    Rouge2,
    /// ROUGE-L: Longest common subsequence
    RougeL,
}

impl RougeMetric {
    /// Create ROUGE-1 metric
    pub fn rouge1() -> Self {
        Self { rouge_type: RougeType::Rouge1 }
    }

    /// Create ROUGE-2 metric
    pub fn rouge2() -> Self {
        Self { rouge_type: RougeType::Rouge2 }
    }

    /// Create ROUGE-L metric
    pub fn rouge_l() -> Self {
        Self { rouge_type: RougeType::RougeL }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn get_ngrams(&self, tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
        let mut ngrams = HashMap::new();
        if tokens.len() < n {
            return ngrams;
        }
        for i in 0..=tokens.len() - n {
            let ngram: Vec<String> = tokens[i..i + n].to_vec();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }
        ngrams
    }

    fn lcs_length(&self, a: &[String], b: &[String]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        dp[m][n]
    }

    fn calculate_rouge_n(&self, generated: &[String], expected: &[String], n: usize) -> f64 {
        let gen_ngrams = self.get_ngrams(generated, n);
        let exp_ngrams = self.get_ngrams(expected, n);

        if exp_ngrams.is_empty() {
            return 0.0;
        }

        let mut overlap = 0usize;
        for (ngram, count) in &exp_ngrams {
            let gen_count = gen_ngrams.get(ngram).copied().unwrap_or(0);
            overlap += count.min(&gen_count);
        }

        let total_expected: usize = exp_ngrams.values().sum();
        overlap as f64 / total_expected as f64
    }

    fn calculate_rouge_l(&self, generated: &[String], expected: &[String]) -> f64 {
        if expected.is_empty() {
            return 0.0;
        }

        let lcs_len = self.lcs_length(generated, expected);
        lcs_len as f64 / expected.len() as f64
    }
}

impl Metric for RougeMetric {
    fn name(&self) -> &str {
        match self.rouge_type {
            RougeType::Rouge1 => "rouge_1",
            RougeType::Rouge2 => "rouge_2",
            RougeType::RougeL => "rouge_l",
        }
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
                let generated_tokens = self.tokenize(&data.generated_text);
                let expected_tokens = self.tokenize(expected);

                match self.rouge_type {
                    RougeType::Rouge1 => self.calculate_rouge_n(&generated_tokens, &expected_tokens, 1),
                    RougeType::Rouge2 => self.calculate_rouge_n(&generated_tokens, &expected_tokens, 2),
                    RougeType::RougeL => self.calculate_rouge_l(&generated_tokens, &expected_tokens),
                }
            }
            None => 0.0,
        }
    }

    fn description(&self) -> &str {
        match self.rouge_type {
            RougeType::Rouge1 => "ROUGE-1: Unigram recall between generated and expected text",
            RougeType::Rouge2 => "ROUGE-2: Bigram recall between generated and expected text",
            RougeType::RougeL => "ROUGE-L: Longest common subsequence recall between generated and expected text",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_match() {
        let metric = ExactMatchMetric::new();
        let data = MetricData::new("Hello World")
            .with_expected("hello world");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_exact_match_no_match() {
        let metric = ExactMatchMetric::new();
        let data = MetricData::new("Hello")
            .with_expected("World");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_exact_match_no_expected() {
        let metric = ExactMatchMetric::new();
        let data = MetricData::new("Hello");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f1_score_perfect() {
        let metric = F1ScoreMetric::new();
        let data = MetricData::new("the cat sat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_f1_score_partial() {
        let metric = F1ScoreMetric::new();
        let data = MetricData::new("the cat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert!(result >= 0.0 && result <= 1.0);
        assert!(result < 1.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_f1_score_empty() {
        let metric = F1ScoreMetric::new();
        let data = MetricData::new("")
            .with_expected("");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_bleu_score() {
        let metric = BleuMetric::new();
        let data = MetricData::new("the cat sat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert!(result > 0.9); // Perfect match should score high
    }

    #[test]
    fn test_bleu_partial() {
        let metric = BleuMetric::new();
        let data = MetricData::new("the cat on mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_rouge_1() {
        let metric = RougeMetric::rouge1();
        let data = MetricData::new("the cat sat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_rouge_2() {
        let metric = RougeMetric::rouge2();
        let data = MetricData::new("the cat sat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_rouge_l() {
        let metric = RougeMetric::rouge_l();
        let data = MetricData::new("the cat sat on the mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_rouge_partial() {
        let metric = RougeMetric::rouge1();
        let data = MetricData::new("cat mat")
            .with_expected("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert!(result > 0.0 && result < 1.0);
    }
}
