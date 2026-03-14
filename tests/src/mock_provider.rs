//! Mock provider for integration testing
//!
//! This module provides a mock implementation of the ModelProvider trait
//! that can be used for testing without requiring external API calls.

use async_trait::async_trait;
use models_core::{
    ChunkStream, EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse,
    HealthStatus, ModelInfo, ModelProvider, TokenUsage, FinishReason,
};
use models_core::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for mock provider behavior
#[derive(Debug, Clone)]
pub struct MockProviderConfig {
    /// Model name to report
    pub model_name: String,
    /// Fixed latency in milliseconds
    pub latency_ms: u64,
    /// Whether to simulate failures
    pub should_fail: bool,
    /// Failure rate (0.0 - 1.0)
    pub failure_rate: f64,
    /// Default response when no patterns match
    pub default_response: String,
    /// Prompt tokens to report
    pub prompt_tokens: u64,
    /// Completion tokens to report
    pub completion_tokens: u64,
}

impl Default for MockProviderConfig {
    fn default() -> Self {
        Self {
            model_name: "mock-model".to_string(),
            latency_ms: 100,
            should_fail: false,
            failure_rate: 0.0,
            default_response: "This is a mock response.".to_string(),
            prompt_tokens: 50,
            completion_tokens: 100,
        }
    }
}

impl MockProviderConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// Set the latency
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Set failure mode
    pub fn with_failure(mut self, rate: f64) -> Self {
        self.should_fail = rate > 0.0;
        self.failure_rate = rate;
        self
    }

    /// Set default response
    pub fn with_default_response(mut self, response: impl Into<String>) -> Self {
        self.default_response = response.into();
        self
    }

    /// Set token counts
    pub fn with_tokens(mut self, prompt: u64, completion: u64) -> Self {
        self.prompt_tokens = prompt;
        self.completion_tokens = completion;
        self
    }
}

/// A mock model provider for testing
pub struct MockProvider {
    config: MockProviderConfig,
    /// Response patterns: (pattern, response) pairs
    response_patterns: Vec<(String, String)>,
    /// Call history for verification
    call_history: Arc<Mutex<Vec<GenerateRequest>>>,
    /// Call count for failure simulation
    call_count: Arc<Mutex<u64>>,
}

impl MockProvider {
    /// Create a new mock provider with default config
    pub fn new() -> Self {
        Self::with_config(MockProviderConfig::default())
    }

    /// Create a new mock provider with custom config
    pub fn with_config(config: MockProviderConfig) -> Self {
        Self {
            config,
            response_patterns: Vec::new(),
            call_history: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Add a response pattern
    /// When the prompt contains the pattern, return the given response
    pub fn add_pattern(&mut self, pattern: impl Into<String>, response: impl Into<String>) -> &mut Self {
        self.response_patterns.push((pattern.into(), response.into()));
        self
    }

    /// Add multiple patterns at once
    pub fn with_patterns(mut self, patterns: Vec<(String, String)>) -> Self {
        self.response_patterns = patterns;
        self
    }

    /// Get the call history
    pub fn call_history(&self) -> Vec<GenerateRequest> {
        self.call_history.lock().unwrap().clone()
    }

    /// Clear the call history
    pub fn clear_history(&self) {
        self.call_history.lock().unwrap().clear();
    }

    /// Get the call count
    pub fn call_count(&self) -> u64 {
        *self.call_count.lock().unwrap()
    }

    /// Find a matching response based on patterns
    fn find_response(&self, prompt: &str) -> String {
        for (pattern, response) in &self.response_patterns {
            if prompt.to_lowercase().contains(&pattern.to_lowercase()) {
                return response.clone();
            }
        }
        self.config.default_response.clone()
    }

    /// Create a provider that simulates coding tasks
    pub fn coding_assistant() -> Self {
        let mut provider = Self::new();
        provider
            .add_pattern("sum", "def sum_list(numbers):\n    return sum(numbers)")
            .add_pattern("max", "def find_max(numbers):\n    return max(numbers) if numbers else None")
            .add_pattern("palindrome", "def is_palindrome(s):\n    return s == s[::-1]")
            .add_pattern("reverse", "def reverse_string(s):\n    return s[::-1]")
            .add_pattern("factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)")
            .add_pattern("vowels", "def count_vowels(s):\n    return sum(1 for c in s if c.lower() in 'aeiou')")
            .add_pattern("even", "def is_even(n):\n    return n % 2 == 0")
            .add_pattern("second largest", "def second_largest(numbers):\n    return sorted(set(numbers))[-2]")
            .add_pattern("duplicates", "def remove_duplicates(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]")
            .add_pattern("sum of digits", "def sum_digits(n):\n    return sum(int(d) for d in str(n))");
        provider
    }

    /// Create a provider that simulates reasoning tasks
    pub fn reasoning_assistant() -> Self {
        let mut provider = Self::new();
        provider
            .add_pattern("What is", "Based on the information provided, the answer is:")
            .add_pattern("Which of", "The correct answer is: A")
            .add_pattern("Explain", "Let me explain this step by step:")
            .add_pattern("Why", "The reason is because of the following factors:")
            .add_pattern("How", "Here's how this works:");
        provider
    }

    /// Create a provider with perfect accuracy (always correct)
    pub fn perfect_coder() -> Self {
        let mut provider = Self::new();
        provider
            .add_pattern("sum of all numbers", "def sum_list(numbers):\n    return sum(numbers)")
            .add_pattern("largest element", "def find_max(numbers):\n    return max(numbers) if numbers else None")
            .add_pattern("palindrome", "def is_palindrome(s):\n    return s == s[::-1]")
            .add_pattern("vowels", "def count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)")
            .add_pattern("reverse", "def reverse_string(s):\n    return s[::-1]")
            .add_pattern("factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)")
            .add_pattern("even", "def is_even(n):\n    return n % 2 == 0")
            .add_pattern("second largest", "def second_largest(numbers):\n    unique_nums = list(set(numbers))\n    if len(unique_nums) < 2:\n        return None\n    unique_nums.sort(reverse=True)\n    return unique_nums[1]")
            .add_pattern("duplicates", "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result")
            .add_pattern("sum of digits", "def sum_digits(n):\n    return sum(int(digit) for digit in str(abs(n)))")
            .add_pattern("remove duplicate", "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result");
        provider
    }

    /// Create a provider with poor accuracy (mostly incorrect)
    pub fn poor_coder() -> Self {
        let mut provider = Self::new();
        provider
            .add_pattern("sum", "def sum_list(numbers):\n    return 0  # TODO: implement")
            .add_pattern("max", "def find_max(numbers):\n    return numbers[0]")
            .add_pattern("palindrome", "def is_palindrome(s):\n    return True")
            .add_pattern("vowels", "def count_vowels(s):\n    return len(s)")
            .add_pattern("reverse", "def reverse_string(s):\n    return s")
            .add_pattern("factorial", "def factorial(n):\n    return n")
            .add_pattern("even", "def is_even(n):\n    return True")
            .add_pattern("second largest", "def second_largest(numbers):\n    return max(numbers)")
            .add_pattern("duplicates", "def remove_duplicates(lst):\n    return lst")
            .add_pattern("sum of digits", "def sum_digits(n):\n    return n");
        provider
    }

    /// Create a high-latency provider for performance testing
    pub fn slow_provider(delay_ms: u64) -> Self {
        let config = MockProviderConfig::new()
            .with_latency(delay_ms);
        Self::with_config(config)
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // Record the call
        self.call_history.lock().unwrap().push(request.clone());

        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        // Check if we should fail
        if self.config.should_fail && rand::random::<f64>() < self.config.failure_rate {
            return Err(models_core::Error::Provider(
                "Mock provider simulated failure".to_string()
            ));
        }

        // Simulate latency
        if self.config.latency_ms > 0 {
            sleep(Duration::from_millis(self.config.latency_ms)).await;
        }

        // Generate response based on patterns
        let text = self.find_response(&request.prompt);

        let mut response = GenerateResponse::new(text, &self.config.model_name)
            .with_latency(self.config.latency_ms)
            .with_finish_reason(FinishReason::Stop);

        // Add token usage if configured
        if self.config.prompt_tokens > 0 || self.config.completion_tokens > 0 {
            response = response.with_tokens(TokenUsage::new(
                self.config.prompt_tokens,
                self.config.completion_tokens,
            ));
        }

        Ok(response)
    }

    async fn generate_stream(&self, _request: GenerateRequest) -> Result<ChunkStream> {
        // For simplicity, just return an empty stream in the mock
        // In a real implementation, this would return a stream of chunks
        Err(models_core::Error::Provider(
            "Streaming not supported in mock provider".to_string()
        ))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        // Return a mock embedding vector
        let embedding = vec![0.1; 384]; // 384-dimensional mock embedding
        let mut response = EmbedResponse::new(embedding, &self.config.model_name);
        response.token_count = Some(request.text.len() as u64 / 4);
        Ok(response)
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        Ok(ModelInfo::new(model_id, &self.config.model_name)
            .with_family("mock")
            .with_parameters("7B")
            .with_context_length(4096))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        if self.config.should_fail {
            Ok(HealthStatus::unhealthy("Mock provider is configured to fail"))
        } else {
            Ok(HealthStatus::healthy("Mock provider is healthy")
                .with_latency(self.config.latency_ms))
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo::new(&self.config.model_name, &self.config.model_name)
                .with_family("mock")
                .with_parameters("7B"),
        ])
    }

    fn provider_name(&self) -> &str {
        "mock"
    }

    fn default_model(&self) -> Option<&str> {
        Some(&self.config.model_name)
    }
}

/// Builder for creating complex mock providers
pub struct MockProviderBuilder {
    config: MockProviderConfig,
    patterns: Vec<(String, String)>,
}

impl MockProviderBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MockProviderConfig::default(),
            patterns: Vec::new(),
        }
    }

    /// Set the model name
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.config.model_name = name.into();
        self
    }

    /// Set the latency
    pub fn latency(mut self, ms: u64) -> Self {
        self.config.latency_ms = ms;
        self
    }

    /// Set failure mode
    pub fn should_fail(mut self, rate: f64) -> Self {
        self.config.should_fail = rate > 0.0;
        self.config.failure_rate = rate;
        self
    }

    /// Set default response
    pub fn default_response(mut self, response: impl Into<String>) -> Self {
        self.config.default_response = response.into();
        self
    }

    /// Add a response pattern
    pub fn pattern(mut self, pattern: impl Into<String>, response: impl Into<String>) -> Self {
        self.patterns.push((pattern.into(), response.into()));
        self
    }

    /// Set token counts
    pub fn tokens(mut self, prompt: u64, completion: u64) -> Self {
        self.config.prompt_tokens = prompt;
        self.config.completion_tokens = completion;
        self
    }

    /// Build the mock provider
    pub fn build(self) -> MockProvider {
        MockProvider::with_config(self.config).with_patterns(self.patterns)
    }
}

impl Default for MockProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_basic() {
        let provider = MockProvider::new();
        let request = GenerateRequest::new("Hello, world!");

        let response = provider.generate(request).await.unwrap();

        assert_eq!(response.text, "This is a mock response.");
        assert_eq!(response.model_name, "mock-model");
        assert!(response.tokens_used.is_some());
    }

    #[tokio::test]
    async fn test_mock_provider_with_pattern() {
        let mut provider = MockProvider::new();
        provider.add_pattern("hello", "Hello, user!");

        let request = GenerateRequest::new("Say hello to me");
        let response = provider.generate(request).await.unwrap();

        assert_eq!(response.text, "Hello, user!");
    }

    #[tokio::test]
    async fn test_mock_provider_coding_assistant() {
        let provider = MockProvider::coding_assistant();

        let request = GenerateRequest::new("Write a function to calculate the sum");
        let response = provider.generate(request).await.unwrap();

        assert!(response.text.contains("sum_list"));
    }

    #[tokio::test]
    async fn test_mock_provider_call_count() {
        let provider = MockProvider::new();

        assert_eq!(provider.call_count(), 0);

        let request = GenerateRequest::new("Test");
        provider.generate(request.clone()).await.unwrap();

        assert_eq!(provider.call_count(), 1);

        provider.generate(request).await.unwrap();
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_provider_health_check() {
        let provider = MockProvider::new();
        let health = provider.health_check().await.unwrap();

        assert!(health.healthy);
        assert_eq!(health.message, "Mock provider is healthy");
    }

    #[tokio::test]
    async fn test_mock_provider_builder() {
        let provider = MockProviderBuilder::new()
            .model_name("custom-model")
            .latency(500)
            .pattern("test", "response")
            .tokens(10, 20)
            .build();

        assert_eq!(provider.call_count(), 0);

        let request = GenerateRequest::new("This is a test");
        let response = provider.generate(request).await.unwrap();

        assert_eq!(response.text, "response");
        assert_eq!(response.model_name, "custom-model");
        assert_eq!(response.latency_ms, 500);
    }

    #[tokio::test]
    async fn test_mock_provider_list_models() {
        let provider = MockProvider::new();
        let models = provider.list_models().await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "mock-model");
    }

    #[tokio::test]
    async fn test_mock_provider_embed() {
        let provider = MockProvider::new();
        let request = EmbedRequest::new("Test text");
        let response = provider.embed(request).await.unwrap();

        assert_eq!(response.embedding.len(), 384);
        assert_eq!(response.model, "mock-model");
    }
}
