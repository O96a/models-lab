//! Accuracy metrics for LLM evaluation

use crate::{Metric, MetricData, MetricDirection, MetricUnit};

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
        let mut generated_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for token in generated {
            *generated_counts.entry(token.as_str()).or_insert(0) += 1;
        }

        let mut expected_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
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
        // Just verify it's between 0 and 1 and not perfect
        assert!(result >= 0.0 && result <= 1.0);
        assert!(result < 1.0); // Not perfect match
        assert!(result > 0.0); // Some overlap
    }

    #[test]
    fn test_f1_score_empty() {
        let metric = F1ScoreMetric::new();
        let data = MetricData::new("")
            .with_expected("");

        let result = metric.calculate(&data);
        assert_eq!(result, 1.0);
    }
}
