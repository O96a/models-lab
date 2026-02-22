//! Efficiency metrics for LLM evaluation

use crate::{Metric, MetricData, MetricDirection, MetricUnit};

/// Cost per token metric
///
/// Calculates the cost per token based on token counts and a cost rate.
/// Requires `cost_per_1k_tokens` in metadata or uses a default rate.
pub struct CostPerTokenMetric {
    /// Cost per 1000 tokens in dollars
    cost_per_1k_tokens: f64,
}

impl CostPerTokenMetric {
    /// Create a new cost per token metric with default rate
    /// Default: $0.002 per 1K tokens (approximate GPT-3.5-turbo rate)
    pub fn new() -> Self {
        Self {
            cost_per_1k_tokens: 0.002,
        }
    }

    /// Create with custom cost per 1K tokens
    pub fn with_cost(cost_per_1k: f64) -> Self {
        Self {
            cost_per_1k_tokens: cost_per_1k,
        }
    }
}

impl Default for CostPerTokenMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for CostPerTokenMetric {
    fn name(&self) -> &str {
        "cost_per_token"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Custom // dollars per token
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        // Check for custom cost in metadata
        let cost_per_1k = data.metadata.get("cost_per_1k_tokens")
            .and_then(|v| v.as_f64())
            .unwrap_or(self.cost_per_1k_tokens);

        // Calculate cost based on total tokens (prompt + generated)
        let prompt_tokens = data.prompt_tokens.unwrap_or(0) as f64;
        let generated_tokens = data.tokens_generated.unwrap_or(0) as f64;
        let total_tokens = prompt_tokens + generated_tokens;

        if total_tokens == 0.0 {
            return 0.0;
        }

        // Cost per token = (cost_per_1k / 1000)
        (cost_per_1k / 1000.0) * total_tokens
    }

    fn description(&self) -> &str {
        "Total cost in dollars for the API call based on token usage"
    }
}

/// Total cost metric
///
/// Calculates the total cost based on separate prompt and completion token rates.
pub struct TotalCostMetric {
    /// Cost per 1K prompt tokens
    prompt_cost_per_1k: f64,
    /// Cost per 1K completion tokens
    completion_cost_per_1k: f64,
}

impl TotalCostMetric {
    /// Create with default GPT-3.5-turbo rates
    pub fn new() -> Self {
        Self {
            prompt_cost_per_1k: 0.0015,
            completion_cost_per_1k: 0.002,
        }
    }

    /// Create with custom rates
    pub fn with_rates(prompt_cost: f64, completion_cost: f64) -> Self {
        Self {
            prompt_cost_per_1k: prompt_cost,
            completion_cost_per_1k: completion_cost,
        }
    }
}

impl Default for TotalCostMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for TotalCostMetric {
    fn name(&self) -> &str {
        "total_cost"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Custom // dollars
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        let prompt_tokens = data.prompt_tokens.unwrap_or(0) as f64;
        let completion_tokens = data.tokens_generated.unwrap_or(0) as f64;

        let prompt_cost = (prompt_tokens / 1000.0) * self.prompt_cost_per_1k;
        let completion_cost = (completion_tokens / 1000.0) * self.completion_cost_per_1k;

        prompt_cost + completion_cost
    }

    fn description(&self) -> &str {
        "Total API cost in dollars with separate prompt and completion rates"
    }
}

/// Quality per dollar metric
///
/// Calculates a quality score per dollar spent.
/// Requires a quality score (accuracy, f1, etc.) in metadata.
pub struct QualityPerDollarMetric {
    /// Cost per 1K tokens
    cost_per_1k_tokens: f64,
}

impl QualityPerDollarMetric {
    pub fn new() -> Self {
        Self {
            cost_per_1k_tokens: 0.002,
        }
    }

    pub fn with_cost(cost_per_1k: f64) -> Self {
        Self {
            cost_per_1k_tokens: cost_per_1k,
        }
    }
}

impl Default for QualityPerDollarMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for QualityPerDollarMetric {
    fn name(&self) -> &str {
        "quality_per_dollar"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Custom // quality points per dollar
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        // Get quality score from metadata (e.g., accuracy, f1)
        let quality = data.metadata.get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let prompt_tokens = data.prompt_tokens.unwrap_or(0) as f64;
        let generated_tokens = data.tokens_generated.unwrap_or(0) as f64;
        let total_tokens = prompt_tokens + generated_tokens;

        if total_tokens == 0.0 {
            return 0.0;
        }

        let cost = (total_tokens / 1000.0) * self.cost_per_1k_tokens;

        if cost == 0.0 {
            return 0.0;
        }

        quality / cost
    }

    fn description(&self) -> &str {
        "Quality score per dollar spent (requires quality_score in metadata)"
    }
}

/// Latency efficiency metric
///
/// Tokens generated per second.
pub struct LatencyEfficiencyMetric;

impl LatencyEfficiencyMetric {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LatencyEfficiencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for LatencyEfficiencyMetric {
    fn name(&self) -> &str {
        "latency_efficiency"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::TokensPerSecond
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        let tokens = data.tokens_generated.unwrap_or(0) as f64;
        let latency_seconds = data.latency_ms as f64 / 1000.0;

        if latency_seconds == 0.0 {
            return 0.0;
        }

        tokens / latency_seconds
    }

    fn description(&self) -> &str {
        "Number of tokens generated per second"
    }
}

/// Cost efficiency metric
///
/// Calculates the cost for 1% of quality improvement.
pub struct CostEfficiencyMetric {
    cost_per_1k_tokens: f64,
}

impl CostEfficiencyMetric {
    pub fn new() -> Self {
        Self {
            cost_per_1k_tokens: 0.002,
        }
    }

    pub fn with_cost(cost_per_1k: f64) -> Self {
        Self {
            cost_per_1k_tokens: cost_per_1k,
        }
    }
}

impl Default for CostEfficiencyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for CostEfficiencyMetric {
    fn name(&self) -> &str {
        "cost_efficiency"
    }

    fn unit(&self) -> MetricUnit {
        MetricUnit::Custom // cost per quality point
    }

    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }

    fn calculate(&self, data: &MetricData) -> f64 {
        // Get quality score from metadata
        let quality = data.metadata.get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let prompt_tokens = data.prompt_tokens.unwrap_or(0) as f64;
        let generated_tokens = data.tokens_generated.unwrap_or(0) as f64;
        let total_tokens = prompt_tokens + generated_tokens;

        let cost = (total_tokens / 1000.0) * self.cost_per_1k_tokens;

        if quality == 0.0 {
            return 0.0;
        }

        cost / quality
    }

    fn description(&self) -> &str {
        "Cost in dollars per percentage point of quality (requires quality_score in metadata)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_per_token() {
        let metric = CostPerTokenMetric::with_cost(1.0); // $1 per 1K tokens
        let data = MetricData::new("test")
            .with_tokens(500, 500); // 1000 total tokens

        let result = metric.calculate(&data);
        assert!((result - 1.0).abs() < 0.001); // Should be $1
    }

    #[test]
    fn test_total_cost() {
        let metric = TotalCostMetric::with_rates(1.0, 2.0);
        let data = MetricData::new("test")
            .with_tokens(500, 1000); // 500 generated, 1K prompt

        let result = metric.calculate(&data);
        // $1 for prompt (1K * $1/1K) + $1 for completion (500 * $2/1K) = $2
        assert!((result - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_efficiency() {
        let metric = LatencyEfficiencyMetric::new();
        let data = MetricData::new("test")
            .with_latency(1000) // 1 second
            .with_tokens(100, 0); // 100 tokens

        let result = metric.calculate(&data);
        assert_eq!(result, 100.0); // 100 tokens per second
    }

    #[test]
    fn test_quality_per_dollar() {
        let metric = QualityPerDollarMetric::with_cost(1.0);
        let data = MetricData::new("test")
            .with_tokens(1000, 0)
            .with_metadata("quality_score".to_string(), serde_json::json!(0.8));

        let result = metric.calculate(&data);
        // Quality 0.8 / cost $1 = 0.8
        assert!((result - 0.8).abs() < 0.001);
    }
}