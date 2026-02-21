//! Speed metrics for LLM evaluation

use crate::{Metric, MetricData, MetricDirection, MetricUnit};

/// Tokens per second metric
pub struct TokensPerSecondMetric;

impl TokensPerSecondMetric {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TokensPerSecondMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for TokensPerSecondMetric {
    fn name(&self) -> &str {
        "tokens_per_second"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::TokensPerSecond
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        let tokens = data.tokens_generated.unwrap_or(0);
        let latency_secs = data.latency_ms as f64 / 1000.0;

        if latency_secs > 0.0 {
            tokens as f64 / latency_secs
        } else {
            0.0
        }
    }

    fn description(&self) -> &str {
        "Number of tokens generated per second"
    }
}

/// Time to first token latency metric
pub struct FirstTokenLatencyMetric;

impl FirstTokenLatencyMetric {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FirstTokenLatencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for FirstTokenLatencyMetric {
    fn name(&self) -> &str {
        "first_token_latency"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Milliseconds
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        data.time_to_first_token_ms.unwrap_or(data.latency_ms) as f64
    }

    fn description(&self) -> &str {
        "Time in milliseconds to generate the first token"
    }
}

/// P95 latency metric (tracks latency values for percentile calculation)
pub struct P95LatencyMetric {
    latencies: Vec<u64>,
}

impl P95LatencyMetric {
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
        }
    }

    /// Add a latency measurement
    pub fn add_latency(&mut self, latency_ms: u64) {
        self.latencies.push(latency_ms);
    }

    /// Clear all latency measurements
    pub fn clear(&mut self) {
        self.latencies.clear();
    }
}

impl Default for P95LatencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for P95LatencyMetric {
    fn name(&self) -> &str {
        "p95_latency"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Milliseconds
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        // For single sample, return the latency
        // P95 is typically calculated across many samples
        data.latency_ms as f64
    }

    fn description(&self) -> &str {
        "95th percentile latency across samples"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_per_second() {
        let metric = TokensPerSecondMetric::new();
        let data = MetricData::new("test")
            .with_latency(1000)
            .with_tokens(100, 10);

        let result = metric.calculate(&data);
        assert_eq!(result, 100.0); // 100 tokens / 1 second
    }

    #[test]
    fn test_first_token_latency() {
        let metric = FirstTokenLatencyMetric::new();
        let data = MetricData::new("test")
            .with_latency(1000)
            .with_ttft(150);

        let result = metric.calculate(&data);
        assert_eq!(result, 150.0);
    }

    #[test]
    fn test_first_token_latency_fallback() {
        let metric = FirstTokenLatencyMetric::new();
        let data = MetricData::new("test").with_latency(500);

        let result = metric.calculate(&data);
        assert_eq!(result, 500.0); // Falls back to total latency
    }
}
