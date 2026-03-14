//! Integration tests for Models Lab
//!
//! This crate provides integration tests for end-to-end functionality
//! including CLI commands, API endpoints, benchmarks, and report generation.

use async_trait::async_trait;
use models_core::{
    providers::{
        GenerateRequest, GenerateResponse, EmbedRequest, EmbedResponse,
        HealthStatus, ModelInfo, ModelProvider, TokenUsage, FinishReason,
        ChunkStream, StreamChunk,
    },
    error::Result,
};
use std::collections::HashMap;
use std::sync::Mutex;

/// Mock provider for testing that returns predetermined responses
pub struct MockProvider {
    /// Name of the provider
    name: String,
    /// Default model to use
    default_model: String,
    /// Predetermined responses to return
    responses: Mutex<Vec<String>>,
    /// Current response index
    response_index: Mutex<usize>,
    /// Whether to simulate errors
    simulate_errors: bool,
    /// Latency to simulate (in ms)
    simulated_latency_ms: u64,
}

impl MockProvider {
    /// Create a new mock provider with default settings
    pub fn new() -> Self {
        Self {
            name: "mock".to_string(),
            default_model: "mock-model".to_string(),
            responses: Mutex::new(vec!["Mock response".to_string()]),
            response_index: Mutex::new(0),
            simulate_errors: false,
            simulated_latency_ms: 1,
        }
    }

    /// Create a mock provider with a specific name and model
    pub fn with_name(mut self, name: impl Into<String>, model: impl Into<String>) -> Self {
        self.name = name.into();
        self.default_model = model.into();
        self
    }

    /// Set predetermined responses to return
    pub fn with_responses(mut self, responses: Vec<String>) -> Self {
        self.responses = Mutex::new(responses);
        self
    }

    /// Set a single response for all requests
    pub fn with_response(self, response: impl Into<String>) -> Self {
        self.with_responses(vec![response.into()])
    }

    /// Configure error simulation
    pub fn with_errors(mut self, simulate: bool) -> Self {
        self.simulate_errors = simulate;
        self
    }

    /// Configure simulated latency
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.simulated_latency_ms = latency_ms;
        self
    }

    /// Get the next response from the queue
    fn next_response(&self) -> String {
        let mut index = self.response_index.lock().unwrap();
        let responses = self.responses.lock().unwrap();
        let response = responses.get(*index % responses.len())
            .cloned()
            .unwrap_or_else(|| "Default mock response".to_string());
        *index += 1;
        response
    }

    /// Create a provider configured for MBPP benchmark
    pub fn for_mbpp() -> Self {
        // Responses that would pass MBPP tests
        let responses = vec![
            "```python\ndef sum_list(numbers):\n    return sum(numbers)\n```".to_string(),
            "```python\ndef find_max(numbers):\n    return max(numbers) if numbers else None\n```".to_string(),
            "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```".to_string(),
            "```python\ndef count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)\n```".to_string(),
            "```python\ndef reverse_string(s):\n    return s[::-1]\n```".to_string(),
        ];
        Self::new()
            .with_name("mock", "test-model")
            .with_responses(responses)
            .with_latency(5)
    }

    /// Create a provider configured for Long Context benchmark
    pub fn for_long_context() -> Self {
        // Responses with expected answers from LongContext samples
        let responses = vec![
            "Dr. Elena Vasquez".to_string(),
            "ALPHA-7X9".to_string(),
            "Orion".to_string(),
            "42".to_string(),
            "1923".to_string(),
            "the Arctic Research Station".to_string(),
        ];
        Self::new()
            .with_name("mock", "test-model")
            .with_responses(responses)
            .with_latency(10)
    }

    /// Create a provider that simulates a coding-capable model
    pub fn coding_model() -> Self {
        Self::new()
            .with_name("mock", "coding-model")
            .with_response("```python\ndef solution():\n    pass\n```")
            .with_latency(2)
    }

    /// Create a provider that simulates a general purpose model
    pub fn general_model() -> Self {
        Self::new()
            .with_name("mock", "general-model")
            .with_response("This is a test response.")
            .with_latency(1)
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
        // Simulate latency
        if self.simulated_latency_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.simulated_latency_ms)).await;
        }

        if self.simulate_errors {
            return Err(models_core::Error::Provider("Simulated error".to_string()));
        }

        let text = self.next_response();
        Ok(GenerateResponse::new(text, &self.default_model)
            .with_tokens(TokenUsage::new(10, 20))
            .with_latency(self.simulated_latency_ms))
    }

    async fn generate_stream(&self, _request: GenerateRequest) -> Result<models_core::providers::ChunkStream> {
        use futures::stream;
        use models_core::providers::StreamChunk;

        let text = self.next_response();
        let chunk = StreamChunk {
            text,
            done: true,
            tokens_used: Some(TokenUsage::new(10, 20)),
            finish_reason: Some(FinishReason::Stop),
        };

        let stream = Box::pin(stream::iter(vec![Ok(chunk)]));
        Ok(stream as models_core::providers::ChunkStream)
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        // Return a simple mock embedding
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let model = request.model.unwrap_or_else(|| self.default_model.clone());
        let mut response = EmbedResponse::new(embedding, model);
        response.token_count = Some(10);
        Ok(response)
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let mut info = ModelInfo::new(model_id, format!("Mock {}", model_id));
        info.context_length = Some(8192);
        Ok(info)
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        Ok(HealthStatus::healthy("Mock provider is healthy"))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo::new(&self.default_model, format!("Mock {}", self.default_model)),
        ])
    }

    fn provider_name(&self) -> &str {
        &self.name
    }

    fn default_model(&self) -> Option<&str> {
        Some(&self.default_model)
    }
}

/// Test utilities for common operations
pub mod test_utils {
    use models_benchmark::{BenchmarkRegistry, BenchmarkConfig};
    use models_core::{EvaluationConfig, EvaluationOrchestrator, InMemoryStorage};
    use std::sync::Arc;

    /// Create a test benchmark registry with built-in benchmarks
    pub fn create_test_registry() -> BenchmarkRegistry {
        BenchmarkRegistry::with_builtin()
    }

    /// Create a default test evaluation config
    pub fn create_test_eval_config() -> EvaluationConfig {
        EvaluationConfig {
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 0,
            max_samples_per_benchmark: Some(2), // Use minimal samples for fast tests
            benchmarks: vec!["mbpp".to_string()],
            save_results: false,
            detailed_report: true,
        }
    }

    /// Create a default test benchmark config
    pub fn create_test_benchmark_config() -> BenchmarkConfig {
        BenchmarkConfig {
            max_samples: Some(2), // Use minimal samples for fast tests
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 0,
            stop_sequences: vec![],
            seed: None,
            categories: None,
        }
    }

    /// Create an evaluation orchestrator with in-memory storage
    pub fn create_test_orchestrator() -> EvaluationOrchestrator {
        let storage = Arc::new(InMemoryStorage::new());
        EvaluationOrchestrator::new(storage)
    }

    /// Assert that a report has valid structure
    pub fn assert_valid_report(report: &models_core::EvaluationReport) {
        assert!(!report.model_name.is_empty(), "Report should have model name");
        assert!(!report.provider_name.is_empty(), "Report should have provider name");
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0, "Overall score should be between 0 and 1");
        assert!(!report.benchmark_results.is_empty() || report.config.benchmarks.is_empty(), "Should have benchmark results");
    }

    /// Assert that a benchmark result has valid structure
    pub fn assert_valid_benchmark_result(result: &models_benchmark::BenchmarkResult) {
        assert!(!result.benchmark_id.is_empty(), "Result should have benchmark ID");
        assert!(!result.model_name.is_empty(), "Result should have model name");
        assert!(result.statistics.total_samples > 0, "Should have evaluated samples");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_generate() {
        let provider = MockProvider::new()
            .with_response("Test response");

        let request = GenerateRequest::new("Test prompt");
        let response = provider.generate(request).await.unwrap();

        assert_eq!(response.text, "Test response");
        assert_eq!(response.model_name, "mock-model");
    }

    #[tokio::test]
    async fn test_mock_provider_multiple_responses() {
        let provider = MockProvider::new()
            .with_responses(vec![
                "Response 1".to_string(),
                "Response 2".to_string(),
            ]);

        let request = GenerateRequest::new("Test prompt");

        let response1 = provider.generate(request.clone()).await.unwrap();
        let response2 = provider.generate(request.clone()).await.unwrap();
        let response3 = provider.generate(request).await.unwrap(); // Cycles back

        assert_eq!(response1.text, "Response 1");
        assert_eq!(response2.text, "Response 2");
        assert_eq!(response3.text, "Response 1"); // Cycles
    }

    #[tokio::test]
    async fn test_mock_provider_health() {
        let provider = MockProvider::new();
        let health = provider.health_check().await.unwrap();
        assert!(health.healthy);
    }

    #[test]
    fn test_mock_provider_mbpp_config() {
        let provider = MockProvider::for_mbpp();
        assert_eq!(provider.provider_name(), "mock");
        assert_eq!(provider.default_model(), Some("test-model"));
    }

    #[test]
    fn test_mock_provider_long_context_config() {
        let provider = MockProvider::for_long_context();
        assert_eq!(provider.provider_name(), "mock");
        assert_eq!(provider.default_model(), Some("test-model"));
    }
}