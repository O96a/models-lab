//! Drift detection metrics for LLM evaluation
//!
//! These metrics measure how much model outputs vary across multiple runs
//! or under different temperature settings, helping detect output instability.

use crate::{Metric, MetricData, MetricDirection, MetricUnit};
use std::collections::HashSet;

/// Output drift metric
///
/// Measures semantic similarity variance across multiple runs of the same prompt.
/// Returns a drift score between 0.0 (no drift/high consistency) and 1.0 (high drift/low consistency).
pub struct OutputDriftMetric {
    ngram_size: usize,
    case_sensitive: bool,
}

impl OutputDriftMetric {
    /// Create a new output drift metric with default settings
    pub fn new() -> Self {
        Self {
            ngram_size: 2,
            case_sensitive: false,
        }
    }

    /// Set the n-gram size for similarity calculation
    pub fn with_ngram_size(mut self, n: usize) -> Self {
        self.ngram_size = n.max(1);
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    /// Tokenize text into words
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

    /// Generate n-grams from tokens
    fn get_ngrams(&self, tokens: &[String]) -> HashSet<Vec<String>> {
        let mut ngrams = HashSet::new();
        if tokens.len() < self.ngram_size {
            return ngrams;
        }
        for i in 0..=tokens.len() - self.ngram_size {
            let ngram: Vec<String> = tokens[i..i + self.ngram_size].to_vec();
            ngrams.insert(ngram);
        }
        ngrams
    }

    /// Calculate Jaccard similarity between two texts using n-grams
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f64 {
        let tokens1 = self.tokenize(text1);
        let tokens2 = self.tokenize(text2);

        // Handle edge cases
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0; // Both empty = identical
        }
        if tokens1.is_empty() || tokens2.is_empty() {
            return 0.0; // One empty, one not = completely different
        }

        // If texts are identical, return 1.0 immediately (optimization)
        if text1 == text2 {
            return 1.0;
        }

        let ngrams1 = self.get_ngrams(&tokens1);
        let ngrams2 = self.get_ngrams(&tokens2);

        if ngrams1.is_empty() && ngrams2.is_empty() {
            // Both have no n-grams (texts too short)
            // Fall back to exact match of tokens
            return if tokens1 == tokens2 { 1.0 } else { 0.0 };
        }

        // Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        let intersection: HashSet<_> = ngrams1.intersection(&ngrams2).collect();
        let union: HashSet<_> = ngrams1.union(&ngrams2).collect();

        if union.is_empty() {
            return 0.0;
        }

        intersection.len() as f64 / union.len() as f64
    }

    /// Calculate pairwise similarities between all outputs
    fn calculate_pairwise_similarities(&self, outputs: &[String]) -> Vec<f64> {
        let mut similarities = Vec::new();
        let n = outputs.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.calculate_similarity(&outputs[i], &outputs[j]);
                similarities.push(sim);
            }
        }

        similarities
    }

    /// Extract multiple outputs from metadata
    fn extract_outputs_from_metadata(&self, data: &MetricData) -> Vec<String> {
        // Try to get drift_outputs from metadata
        if let Some(outputs_value) = data.metadata.get("drift_outputs") {
            if let Some(outputs_array) = outputs_value.as_array() {
                return outputs_array
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }
        }

        // Fallback: just use the current generated text
        vec![data.generated_text.clone()]
    }
}

impl Default for OutputDriftMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for OutputDriftMetric {
    fn name(&self) -> &str {
        "output_drift"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        let outputs = self.extract_outputs_from_metadata(data);

        // Need at least 2 outputs to calculate drift
        if outputs.len() < 2 {
            return 0.0; // No drift with single output
        }

        // Calculate pairwise similarities
        let similarities = self.calculate_pairwise_similarities(&outputs);

        if similarities.is_empty() {
            return 0.0;
        }

        // Calculate mean similarity
        let mean_similarity = similarities.iter().sum::<f64>() / similarities.len() as f64;

        // Drift = 1.0 - mean_similarity
        // High similarity (close to 1.0) = low drift (close to 0.0)
        // Low similarity (close to 0.0) = high drift (close to 1.0)
        let drift = 1.0 - mean_similarity;

        // Clamp to [0.0, 1.0]
        drift.clamp(0.0, 1.0)
    }

    fn description(&self) -> &str {
        "Measures semantic similarity variance across multiple runs of the same prompt. \
         Returns 0.0 (no drift/high consistency) to 1.0 (high drift/low consistency). \
         Requires 'drift_outputs' metadata with array of output strings."
    }
}

/// Temperature variance metric
///
/// Measures consistency when running with different temperature settings.
/// Compares outputs at temperatures 0.0, 0.5, and 1.0 to measure divergence.
/// Returns a variance score between 0.0 (no variance/high consistency) and 1.0 (high variance/low consistency).
pub struct TemperatureVarianceMetric {
    ngram_size: usize,
    case_sensitive: bool,
}

/// Temperature-output pair for metadata
type TemperatureOutput = (f64, String);

impl TemperatureVarianceMetric {
    /// Create a new temperature variance metric with default settings
    pub fn new() -> Self {
        Self {
            ngram_size: 2,
            case_sensitive: false,
        }
    }

    /// Set the n-gram size for similarity calculation
    pub fn with_ngram_size(mut self, n: usize) -> Self {
        self.ngram_size = n.max(1);
        self
    }

    /// Set case sensitivity
    pub fn case_sensitive(mut self, sensitive: bool) -> Self {
        self.case_sensitive = sensitive;
        self
    }

    /// Tokenize text into words
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

    /// Generate n-grams from tokens
    fn get_ngrams(&self, tokens: &[String]) -> HashSet<Vec<String>> {
        let mut ngrams = HashSet::new();
        if tokens.len() < self.ngram_size {
            return ngrams;
        }
        for i in 0..=tokens.len() - self.ngram_size {
            let ngram: Vec<String> = tokens[i..i + self.ngram_size].to_vec();
            ngrams.insert(ngram);
        }
        ngrams
    }

    /// Calculate Jaccard similarity between two texts using n-grams
    fn calculate_similarity(&self, text1: &str, text2: &str) -> f64 {
        let tokens1 = self.tokenize(text1);
        let tokens2 = self.tokenize(text2);

        // Handle edge cases
        if tokens1.is_empty() && tokens2.is_empty() {
            return 1.0;
        }
        if tokens1.is_empty() || tokens2.is_empty() {
            return 0.0;
        }

        // If texts are identical, return 1.0 immediately
        if text1 == text2 {
            return 1.0;
        }

        let ngrams1 = self.get_ngrams(&tokens1);
        let ngrams2 = self.get_ngrams(&tokens2);

        if ngrams1.is_empty() && ngrams2.is_empty() {
            return if tokens1 == tokens2 { 1.0 } else { 0.0 };
        }

        // Calculate Jaccard similarity
        let intersection: HashSet<_> = ngrams1.intersection(&ngrams2).collect();
        let union: HashSet<_> = ngrams1.union(&ngrams2).collect();

        if union.is_empty() {
            return 0.0;
        }

        intersection.len() as f64 / union.len() as f64
    }

    /// Extract temperature-output pairs from metadata
    fn extract_temperature_outputs(&self, data: &MetricData) -> Vec<TemperatureOutput> {
        // Try to get temperature_outputs from metadata
        if let Some(outputs_value) = data.metadata.get("temperature_outputs") {
            if let Some(outputs_array) = outputs_value.as_array() {
                return outputs_array
                    .iter()
                    .filter_map(|v| {
                        if let Some(obj) = v.as_object() {
                            let temp = obj.get("temperature")?.as_f64()?;
                            let output = obj.get("output")?.as_str()?;
                            Some((temp, output.to_string()))
                        } else if let Some(arr) = v.as_array() {
                            // Handle array format: [temperature, output]
                            if arr.len() >= 2 {
                                let temp = arr[0].as_f64()?;
                                let output = arr[1].as_str()?;
                                Some((temp, output.to_string()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }

        Vec::new()
    }

    /// Calculate variance based on temperature range
    fn calculate_temperature_variance(&self, temp_outputs: &[TemperatureOutput]) -> f64 {
        if temp_outputs.len() < 2 {
            return 0.0;
        }

        // Sort by temperature
        let mut sorted = temp_outputs.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compare outputs between temperature extremes (min and max temp)
        let min_temp_output = &sorted.first().unwrap().1;
        let max_temp_output = &sorted.last().unwrap().1;
        let extreme_similarity = self.calculate_similarity(min_temp_output, max_temp_output);

        // Compare adjacent temperature outputs
        let mut adjacent_similarities = Vec::new();
        for i in 1..sorted.len() {
            let sim = self.calculate_similarity(&sorted[i - 1].1, &sorted[i].1);
            adjacent_similarities.push(sim);
        }

        // Calculate mean similarity across all comparisons
        let mean_adjacent_sim = if !adjacent_similarities.is_empty() {
            adjacent_similarities.iter().sum::<f64>() / adjacent_similarities.len() as f64
        } else {
            1.0
        };

        // Weighted combination: emphasize extreme differences
        // 60% weight on extreme comparison, 40% on adjacent comparisons
        let mean_similarity = 0.6 * extreme_similarity + 0.4 * mean_adjacent_sim;

        // Variance = 1.0 - mean_similarity
        let variance = 1.0 - mean_similarity;

        variance.clamp(0.0, 1.0)
    }
}

impl Default for TemperatureVarianceMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for TemperatureVarianceMetric {
    fn name(&self) -> &str {
        "temperature_variance"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        let temp_outputs = self.extract_temperature_outputs(data);

        if temp_outputs.len() < 2 {
            // Try to use drift_outputs as fallback
            let outputs = self.extract_outputs_from_metadata(data);
            if outputs.len() >= 2 {
                // Calculate pairwise variance as fallback
                let similarities = self.calculate_pairwise_similarities(&outputs);
                if similarities.is_empty() {
                    return 0.0;
                }
                let mean_sim = similarities.iter().sum::<f64>() / similarities.len() as f64;
                return (1.0 - mean_sim).clamp(0.0, 1.0);
            }
            return 0.0;
        }

        self.calculate_temperature_variance(&temp_outputs)
    }

    fn description(&self) -> &str {
        "Measures consistency when running with different temperature settings (0.0, 0.5, 1.0). \
         Returns 0.0 (no variance/high consistency) to 1.0 (high variance/low consistency). \
         Requires 'temperature_outputs' metadata with array of {temperature, output} objects."
    }
}

impl TemperatureVarianceMetric {
    /// Helper to extract drift_outputs (used as fallback)
    fn extract_outputs_from_metadata(&self, data: &MetricData) -> Vec<String> {
        if let Some(outputs_value) = data.metadata.get("drift_outputs") {
            if let Some(outputs_array) = outputs_value.as_array() {
                return outputs_array
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
            }
        }
        vec![data.generated_text.clone()]
    }

    /// Helper for calculating pairwise similarities (used as fallback)
    fn calculate_pairwise_similarities(&self, outputs: &[String]) -> Vec<f64> {
        let mut similarities = Vec::new();
        let n = outputs.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.calculate_similarity(&outputs[i], &outputs[j]);
                similarities.push(sim);
            }
        }

        similarities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_drift_identical_outputs() {
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "the cat sat on the mat",
            "the cat sat on the mat",
            "the cat sat on the mat",
        ];

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "Identical outputs should have 0 drift");
    }

    #[test]
    fn test_output_drift_different_outputs() {
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
        ];

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Different outputs should have some drift");
        assert!(result <= 1.0, "Drift should be at most 1.0");
    }

    #[test]
    fn test_output_drift_partial_similarity() {
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "the cat sat on the mat",
            "the cat sat on the rug", // Similar but not identical
            "the cat sat on the mat", // Back to original
        ];

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Partial similarity should have some drift");
        assert!(result < 1.0, "Partial similarity should not have max drift");
    }

    #[test]
    fn test_output_drift_single_output() {
        let metric = OutputDriftMetric::new();
        let outputs = vec!["the cat sat on the mat"];

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "Single output should have 0 drift");
    }

    #[test]
    fn test_output_drift_no_metadata() {
        let metric = OutputDriftMetric::new();
        let data = MetricData::new("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "No metadata should return 0 drift");
    }

    #[test]
    fn test_temperature_variance_no_metadata() {
        let metric = TemperatureVarianceMetric::new();
        let data = MetricData::new("the cat sat on the mat");

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "No metadata should return 0 variance");
    }

    #[test]
    fn test_temperature_variance_consistent_outputs() {
        let metric = TemperatureVarianceMetric::new();
        let temp_outputs = serde_json::json!([
            {"temperature": 0.0, "output": "the cat sat on the mat"},
            {"temperature": 0.5, "output": "the cat sat on the mat"},
            {"temperature": 1.0, "output": "the cat sat on the mat"},
        ]);

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("temperature_outputs", temp_outputs);

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "Identical outputs across temps should have 0 variance");
    }

    #[test]
    fn test_temperature_variance_divergent_outputs() {
        let metric = TemperatureVarianceMetric::new();
        let temp_outputs = serde_json::json!([
            {"temperature": 0.0, "output": "the cat sat on the mat"},
            {"temperature": 0.5, "output": "the feline rested on the carpet"},
            {"temperature": 1.0, "output": "a dog ran through the field"},
        ]);

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("temperature_outputs", temp_outputs);

        let result = metric.calculate(&data);
        assert!(result > 0.0, "Divergent outputs should have variance");
        assert!(result <= 1.0, "Variance should be at most 1.0");
    }

    #[test]
    fn test_temperature_variance_array_format() {
        let metric = TemperatureVarianceMetric::new();
        // Test array format: [[temp, output], ...]
        let temp_outputs = serde_json::json!([
            [0.0, "the cat sat on the mat"],
            [0.5, "the cat sat on the mat"],
            [1.0, "the cat sat on the mat"],
        ]);

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("temperature_outputs", temp_outputs);

        let result = metric.calculate(&data);
        assert_eq!(result, 0.0, "Array format should work for identical outputs");
    }

    #[test]
    fn test_temperature_variance_fallback_to_drift_outputs() {
        let metric = TemperatureVarianceMetric::new();
        let outputs = vec![
            "the cat sat on the mat",
            "the cat sat on the rug",
        ];

        let data = MetricData::new("the cat sat on the mat")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let result = metric.calculate(&data);
        // Should use drift_outputs as fallback
        assert!(result >= 0.0, "Fallback should produce valid result");
    }

    #[test]
    fn test_ngram_similarity() {
        let metric = OutputDriftMetric::new();

        // Identical texts
        let sim = metric.calculate_similarity("hello world", "hello world");
        assert_eq!(sim, 1.0);

        // Completely different texts
        let sim = metric.calculate_similarity("hello world", "foo bar baz");
        assert_eq!(sim, 0.0);

        // Partial overlap
        let sim = metric.calculate_similarity("the cat sat", "the cat stood");
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_metric_name() {
        let metric1 = OutputDriftMetric::new();
        assert_eq!(metric1.name(), "output_drift");

        let metric2 = TemperatureVarianceMetric::new();
        assert_eq!(metric2.name(), "temperature_variance");
    }

    #[test]
    fn test_metric_direction() {
        let metric1 = OutputDriftMetric::new();
        assert_eq!(metric1.direction(), MetricDirection::LowerIsBetter);

        let metric2 = TemperatureVarianceMetric::new();
        assert_eq!(metric2.direction(), MetricDirection::LowerIsBetter);
    }

    // Additional comprehensive tests

    #[test]
    fn test_output_drift_identical() {
        // Same text = 1.0 consistency = 0.0 drift
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "machine learning is a subset of artificial intelligence",
            "machine learning is a subset of artificial intelligence",
            "machine learning is a subset of artificial intelligence",
        ];

        let data = MetricData::new("machine learning is a subset of artificial intelligence")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let drift = metric.calculate(&data);
        assert_eq!(
            drift, 0.0,
            "Identical outputs should have 0.0 drift (1.0 consistency), got {}",
            drift
        );

        // Verify consistency calculation directly
        let consistency = 1.0 - drift;
        assert_eq!(
            consistency, 1.0,
            "Consistency should be 1.0 for identical outputs"
        );
    }

    #[test]
    fn test_output_drift_completely_different() {
        // Different text = 0.0 consistency = 1.0 drift
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms process data efficiently",
            "quantum physics explores subatomic particle behavior",
        ];

        let data = MetricData::new("the quick brown fox jumps over the lazy dog")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let drift = metric.calculate(&data);
        assert!(
            drift > 0.8,
            "Completely different outputs should have drift close to 1.0, got {}",
            drift
        );

        // Verify consistency is low
        let consistency = 1.0 - drift;
        assert!(
            consistency < 0.2,
            "Consistency should be close to 0.0 for completely different outputs, got {}",
            consistency
        );
    }

    #[test]
    fn test_output_drift_partial() {
        // Some overlap = moderate drift
        let metric = OutputDriftMetric::new();
        let outputs = vec![
            "the cat sat on the mat and looked outside",
            "the cat sat on the mat and slept peacefully",
            "the cat sat on the mat and watched the birds",
        ];

        let data = MetricData::new("the cat sat on the mat and looked outside")
            .with_metadata("drift_outputs", serde_json::json!(outputs));

        let drift = metric.calculate(&data);
        assert!(
            drift > 0.0 && drift < 0.8,
            "Partial overlap should have moderate drift (0.0 < drift < 0.8), got {}",
            drift
        );

        // Verify consistency is moderate
        let consistency = 1.0 - drift;
        assert!(
            consistency > 0.2 && consistency < 1.0,
            "Partial overlap should have moderate consistency (0.2 < consistency < 1.0), got {}",
            consistency
        );

        // Test with another partial case
        let outputs2 = vec![
            "machine learning models are trained on data",
            "deep learning models are trained on large datasets",
            "neural network models are trained on training data",
        ];

        let data2 = MetricData::new("machine learning models are trained on data")
            .with_metadata("drift_outputs", serde_json::json!(outputs2));

        let drift2 = metric.calculate(&data2);
        assert!(
            drift2 > 0.0 && drift2 < 0.9,
            "Semantic overlap should have moderate drift, got {}",
            drift2
        );
    }
}
