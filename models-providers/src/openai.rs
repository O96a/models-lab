//! OpenAI provider adapter
//!
//! Implements the `ModelProvider` trait for OpenAI's API.

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

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TIMEOUT: u64 = 120;

// ============================================================================
// OpenAI API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: OpenAIMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModelData>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelData {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct OpenAIEmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedData>,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedData {
    embedding: Vec<f64>,
    index: u32,
}

// ============================================================================
// OpenAI Provider
// ============================================================================

/// OpenAI provider
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    organization: Option<String>,
    base_url: String,
    default_model: String,
    timeout: Duration,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::with_options(api_key, None, None, None, None)
    }

    /// Create a provider with full options
    pub fn with_options(
        api_key: impl Into<String>,
        organization: Option<&str>,
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
            organization: organization.map(|s| s.to_string()),
            base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
            default_model: default_model.unwrap_or("gpt-4o-mini").to_string(),
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
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(ref org) = self.organization {
            headers.insert(
                "OpenAI-Organization",
                org.parse().unwrap(),
            );
        }
        headers
    }

    /// Convert GenerateRequest to OpenAI request
    fn to_openai_request(&self, request: GenerateRequest, model: &str) -> OpenAIChatRequest {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system_prompt {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        OpenAIChatRequest {
            model: model.to_string(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            seed: request.seed,
            stream: false,
        }
    }

    /// Convert OpenAI response to GenerateResponse
    fn from_openai_response(&self, response: OpenAIChatResponse, latency_ms: u64) -> GenerateResponse {
        let choice = response.choices.first();
        let (text, finish_reason) = match choice {
            Some(c) => {
                let reason = match c.finish_reason.as_deref() {
                    Some("stop") => FinishReason::Stop,
                    Some("length") => FinishReason::Length,
                    Some("content_filter") => FinishReason::ContentFilter,
                    _ => FinishReason::Unknown,
                };
                (c.message.content.clone(), reason)
            }
            None => (String::new(), FinishReason::Error),
        };

        let tokens_used = response.usage.map(|u| TokenUsage::new(
            u.prompt_tokens,
            u.completion_tokens,
        ));

        GenerateResponse::new(text, response.model)
            .with_tokens_opt(tokens_used)
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

trait WithTokensOpt {
    fn with_tokens_opt(self, tokens: Option<TokenUsage>) -> Self;
}

impl WithTokensOpt for GenerateResponse {
    fn with_tokens_opt(mut self, tokens: Option<TokenUsage>) -> Self {
        self.tokens_used = tokens;
        self
    }
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.get_model(&request);
        let openai_request = self.to_openai_request(request, &model);

        debug!("OpenAIProvider: generating with model {}", model);
        let start = Instant::now();

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&openai_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("OpenAI request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "OpenAI API error: {} - {}",
                status, body
            )));
        }

        let chat_response: OpenAIChatResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("OpenAI response parse error: {}", e))
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        Ok(self.from_openai_response(chat_response, latency_ms))
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<ChunkStream> {
        // For now, return a non-streaming response wrapped as a stream
        // TODO: Implement proper streaming using OpenAI's streaming API
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

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        let model = request.model.unwrap_or_else(|| "text-embedding-3-small".to_string());

        debug!("OpenAIProvider: embedding with model {}", model);

        let embed_request = OpenAIEmbedRequest {
            model: model.clone(),
            input: request.text,
        };

        let response = self.client
            .post(format!("{}/embeddings", self.base_url))
            .headers(self.build_headers())
            .json(&embed_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("OpenAI embed request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "OpenAI embed API error: {} - {}",
                status, body
            )));
        }

        let embed_response: OpenAIEmbedResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("OpenAI embed response parse error: {}", e))
        })?;

        let embedding = embed_response
            .data
            .first()
            .map(|d| d.embedding.clone())
            .unwrap_or_default();

        Ok(EmbedResponse::new(embedding, embed_response.model)
            .with_token_count_opt(Some(embed_response.usage.total_tokens)))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        // OpenAI doesn't have a specific model info endpoint for individual models
        // We return basic info based on the model ID
        Ok(ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            family: Some(extract_family(model_id)),
            parameters: None,
            quantization: None,
            context_length: None,
            size_bytes: None,
            capabilities: vec!["chat".to_string()],
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();

        let response = self.client
            .get(format!("{}/models", self.base_url))
            .headers(self.build_headers())
            .timeout(Duration::from_secs(10))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status().is_success() => {
                Ok(HealthStatus::healthy("OpenAI API is reachable")
                    .with_latency(latency_ms))
            }
            Ok(resp) => {
                Ok(HealthStatus::unhealthy(format!(
                    "OpenAI API returned status {}",
                    resp.status()
                ))
                .with_latency(latency_ms))
            }
            Err(e) => {
                Ok(HealthStatus::unhealthy(format!(
                    "OpenAI API unreachable: {}",
                    e
                ))
                .with_latency(latency_ms))
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let response = self.client
            .get(format!("{}/models", self.base_url))
            .headers(self.build_headers())
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("OpenAI list models error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "OpenAI list models API error: {} - {}",
                status, body
            )));
        }

        let models_response: OpenAIModelsResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("OpenAI models response parse error: {}", e))
        })?;

        let models: Vec<ModelInfo> = models_response
            .data
            .into_iter()
            .filter(|m| {
                // Filter to only show chat models
                m.id.contains("gpt") || m.id.contains("chat")
            })
            .map(|m| {
                let family = extract_family(&m.id);
                ModelInfo {
                    id: m.id.clone(),
                    name: m.id,
                    family: Some(family),
                    parameters: None,
                    quantization: None,
                    context_length: None,
                    size_bytes: None,
                    capabilities: vec!["chat".to_string()],
                }
            })
            .collect();

        Ok(models)
    }

    fn provider_name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> Option<&str> {
        Some(&self.default_model)
    }
}

/// Helper to extract model family from ID
fn extract_family(model_id: &str) -> String {
    if model_id.contains("gpt-4") {
        "gpt-4".to_string()
    } else if model_id.contains("gpt-3.5") {
        "gpt-3.5".to_string()
    } else if model_id.contains("o1") {
        "o1".to_string()
    } else if model_id.contains("o3") {
        "o3".to_string()
    } else {
        "unknown".to_string()
    }
}

/// Helper trait for optional token count
trait WithTokenCountOpt {
    fn with_token_count_opt(self, count: Option<u64>) -> Self;
}

impl WithTokenCountOpt for EmbedResponse {
    fn with_token_count_opt(mut self, count: Option<u64>) -> Self {
        self.token_count = count;
        self
    }
}

/// Create an OpenAI provider from configuration
pub fn create_openai_provider(
    api_key: &str,
    organization: Option<&str>,
    default_model: Option<&str>,
    base_url: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<OpenAIProvider> {
    OpenAIProvider::with_options(
        api_key,
        organization,
        default_model,
        base_url,
        timeout_seconds,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-api-key");
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_with_options() {
        let provider = OpenAIProvider::with_options(
            "test-api-key",
            Some("org-123"),
            Some("gpt-4"),
            None,
            Some(60),
        );
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.default_model(), Some("gpt-4"));
    }

    #[test]
    fn test_extract_family() {
        assert_eq!(extract_family("gpt-4o-mini"), "gpt-4");
        assert_eq!(extract_family("gpt-3.5-turbo"), "gpt-3.5");
        assert_eq!(extract_family("o1-preview"), "o1");
    }
}
