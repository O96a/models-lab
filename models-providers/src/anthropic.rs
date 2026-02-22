//! Anthropic provider adapter
//!
//! Implements the `ModelProvider` trait for Anthropic's Claude API.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::debug;

use models_core::providers::{
    ChunkStream, EmbedRequest, EmbedResponse, FinishReason, GenerateRequest, GenerateResponse,
    HealthStatus, ModelInfo, ModelProvider, StreamChunk, TokenUsage,
};
use models_core::error::{Error, Result};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_MODEL: &str = "claude-3-5-sonnet-20241022";
const DEFAULT_TIMEOUT: u64 = 120;
const ANTHROPIC_VERSION: &str = "2023-06-01";

// ============================================================================
// Anthropic API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicContent>,
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct AnthropicModelsResponse {
    data: Vec<AnthropicModelData>,
}

#[derive(Debug, Deserialize)]
struct AnthropicModelData {
    id: String,
    #[serde(rename = "type")]
    model_type: String,
    display_name: String,
}

// ============================================================================
// Anthropic Provider
// ============================================================================

/// Anthropic provider
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
    timeout: Duration,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_options(api_key, None, None, None)
    }

    /// Create a provider with full options
    pub fn with_options(
        api_key: impl Into<String>,
        default_model: Option<&str>,
        base_url: Option<&str>,
        timeout_seconds: Option<u64>,
    ) -> Result<Self> {
        let api_key = api_key.into();
        let timeout = Duration::from_secs(timeout_seconds.unwrap_or(DEFAULT_TIMEOUT));

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Provider(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
            default_model: default_model.unwrap_or(DEFAULT_MODEL).to_string(),
            timeout,
        })
    }

    /// Set the default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Build headers for the request
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.api_key.parse().unwrap(),
        );
        headers.insert(
            "anthropic-version",
            ANTHROPIC_VERSION.parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }

    /// Convert GenerateRequest to Anthropic request
    fn to_anthropic_request(&self, request: GenerateRequest, model: &str) -> AnthropicRequest {
        let messages = vec![AnthropicMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        }];

        AnthropicRequest {
            model: model.to_string(),
            max_tokens: request.max_tokens.unwrap_or(4096),
            messages,
            system: request.system_prompt.clone(),
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            stop_sequences: if request.stop_sequences.as_ref().map_or(true, |s| s.is_empty()) {
                None
            } else {
                request.stop_sequences.clone()
            },
        }
    }

    /// Convert Anthropic response to GenerateResponse
    fn from_anthropic_response(&self, response: AnthropicResponse, latency_ms: u64) -> GenerateResponse {
        let text = response
            .content
            .iter()
            .filter(|c| c.content_type == "text")
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("stop_sequence") => FinishReason::Stop,
            _ => FinishReason::Unknown,
        };

        let tokens_used = TokenUsage::new(
            response.usage.input_tokens,
            response.usage.output_tokens,
        );

        GenerateResponse::new(text, response.model)
            .with_tokens(tokens_used)
            .with_latency(latency_ms)
            .with_finish_reason(finish_reason)
    }

    /// Get the model to use
    fn get_model(&self, request: &GenerateRequest) -> String {
        if let Some(model) = request.extra_params.get("model") {
            if let Some(model_str) = model.as_str() {
                return model_str.to_string();
            }
        }
        self.default_model.clone()
    }
}

trait WithTokens {
    fn with_tokens(self, tokens: TokenUsage) -> Self;
}

impl WithTokens for GenerateResponse {
    fn with_tokens(mut self, tokens: TokenUsage) -> Self {
        self.tokens_used = Some(tokens);
        self
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.get_model(&request);
        let anthropic_request = self.to_anthropic_request(request, &model);

        debug!("AnthropicProvider: generating with model {}", model);
        let start = Instant::now();

        let response = self.client
            .post(format!("{}/messages", self.base_url))
            .headers(self.build_headers())
            .json(&anthropic_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("Anthropic request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "Anthropic API error: {} - {}",
                status, body
            )));
        }

        let chat_response: AnthropicResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("Anthropic response parse error: {}", e))
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        Ok(self.from_anthropic_response(chat_response, latency_ms))
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<ChunkStream> {
        // For now, return a non-streaming response wrapped as a stream
        // TODO: Implement proper streaming using Anthropic's streaming API
        let response = self.generate(request).await?;

        let chunk = StreamChunk {
            text: response.text,
            done: true,
            tokens_used: response.tokens_used,
            finish_reason: Some(response.finish_reason),
        };

        let stream = futures::stream::once(async move { Ok(chunk) });
        Ok(Box::pin(stream))
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResponse> {
        // Anthropic doesn't have a dedicated embeddings API
        // Return an error suggesting alternative approaches
        Err(Error::Provider(
            "Anthropic does not provide an embeddings API. \
             Consider using a different provider for embeddings."
                .to_string(),
        ))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        // Anthropic doesn't have a model info endpoint, return static info
        let (family, context_length) = if model_id.contains("claude-3-5") {
            ("claude-3.5".to_string(), Some(200000))
        } else if model_id.contains("claude-3") {
            ("claude-3".to_string(), Some(200000))
        } else {
            ("claude".to_string(), None)
        };

        Ok(ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            family: Some(family),
            parameters: None,
            quantization: None,
            context_length,
            size_bytes: None,
            capabilities: vec!["chat".to_string(), "vision".to_string()],
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        // Anthropic doesn't have a health endpoint, so we check if we can list models
        // or just verify the API key format
        let start = Instant::now();

        if self.api_key.is_empty() {
            return Ok(HealthStatus::unhealthy("API key is not configured")
                .with_latency(0));
        }

        // Simple check - API key should start with "sk-ant-"
        let is_valid_format = self.api_key.starts_with("sk-ant-");

        let latency_ms = start.elapsed().as_millis() as u64;

        if is_valid_format {
            Ok(HealthStatus::healthy("Anthropic API key configured")
                .with_latency(latency_ms))
        } else {
            Ok(HealthStatus::unhealthy("Anthropic API key format may be invalid")
                .with_latency(latency_ms))
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Anthropic doesn't have a list models endpoint
        // Return known models statically
        Ok(vec![
            ModelInfo {
                id: "claude-3-5-sonnet-20241022".to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                family: Some("claude-3.5".to_string()),
                parameters: None,
                quantization: None,
                context_length: Some(200000),
                size_bytes: None,
                capabilities: vec!["chat".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "claude-3-5-haiku-20241022".to_string(),
                name: "Claude 3.5 Haiku".to_string(),
                family: Some("claude-3.5".to_string()),
                parameters: None,
                quantization: None,
                context_length: Some(200000),
                size_bytes: None,
                capabilities: vec!["chat".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "claude-3-opus-20240229".to_string(),
                name: "Claude 3 Opus".to_string(),
                family: Some("claude-3".to_string()),
                parameters: None,
                quantization: None,
                context_length: Some(200000),
                size_bytes: None,
                capabilities: vec!["chat".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "claude-3-sonnet-20240229".to_string(),
                name: "Claude 3 Sonnet".to_string(),
                family: Some("claude-3".to_string()),
                parameters: None,
                quantization: None,
                context_length: Some(200000),
                size_bytes: None,
                capabilities: vec!["chat".to_string(), "vision".to_string()],
            },
            ModelInfo {
                id: "claude-3-haiku-20240307".to_string(),
                name: "Claude 3 Haiku".to_string(),
                family: Some("claude-3".to_string()),
                parameters: None,
                quantization: None,
                context_length: Some(200000),
                size_bytes: None,
                capabilities: vec!["chat".to_string(), "vision".to_string()],
            },
        ])
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn default_model(&self) -> Option<&str> {
        Some(&self.default_model)
    }
}

/// Create an Anthropic provider from configuration
pub fn create_anthropic_provider(
    api_key: &str,
    default_model: Option<&str>,
    base_url: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<AnthropicProvider> {
    AnthropicProvider::with_options(api_key, default_model, base_url, timeout_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AnthropicProvider::new("sk-ant-test-key");
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_with_options() {
        let provider = AnthropicProvider::with_options(
            "sk-ant-test-key",
            Some("claude-3-opus-20240229"),
            None,
            Some(60),
        );
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.default_model(), Some("claude-3-opus-20240229"));
    }

    #[test]
    fn test_default_model() {
        let provider = AnthropicProvider::new("sk-ant-test-key").unwrap();
        assert_eq!(provider.default_model(), Some(DEFAULT_MODEL));
    }
}