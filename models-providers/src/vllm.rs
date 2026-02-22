//! vLLM provider adapter
//!
//! Implements the `ModelProvider` trait for vLLM servers using the OpenAI-compatible API.

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

const DEFAULT_BASE_URL: &str = "http://localhost:8000";
const DEFAULT_TIMEOUT: u64 = 120;

// ============================================================================
// vLLM API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct VllmChatRequest {
    model: String,
    messages: Vec<VllmMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct VllmMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct VllmChatResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<VllmChoice>,
    usage: Option<VllmUsage>,
}

#[derive(Debug, Deserialize)]
struct VllmChoice {
    index: u32,
    message: VllmMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VllmMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct VllmUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct VllmModelsResponse {
    data: Vec<VllmModelData>,
}

#[derive(Debug, Deserialize)]
struct VllmModelData {
    id: String,
    object: String,
    owned_by: Option<String>,
}

// ============================================================================
// vLLM Provider
// ============================================================================

/// vLLM provider (OpenAI-compatible API)
pub struct VllmProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    default_model: String,
    timeout: Duration,
}

impl VllmProvider {
    /// Create a new vLLM provider
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Self::with_options(base_url, None, None, None)
    }

    /// Create a provider with full options
    pub fn with_options(
        base_url: impl Into<String>,
        api_key: Option<&str>,
        default_model: Option<&str>,
        timeout_seconds: Option<u64>,
    ) -> Result<Self> {
        let timeout = Duration::from_secs(timeout_seconds.unwrap_or(DEFAULT_TIMEOUT));

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Provider(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            base_url: base_url.into(),
            api_key: api_key.map(|s| s.to_string()),
            default_model: default_model.unwrap_or("").to_string(),
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
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(ref api_key) = self.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", api_key).parse().unwrap(),
            );
        }
        headers
    }

    /// Convert GenerateRequest to vLLM request
    fn to_vllm_request(&self, request: GenerateRequest, model: &str) -> VllmChatRequest {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system_prompt {
            messages.push(VllmMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        messages.push(VllmMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        VllmChatRequest {
            model: model.to_string(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            top_k: request.top_k,
            stop: request.stop_sequences.clone(),
            seed: request.seed,
            stream: false,
        }
    }

    /// Convert vLLM response to GenerateResponse
    fn from_vllm_response(&self, response: VllmChatResponse, latency_ms: u64) -> GenerateResponse {
        let choice = response.choices.first();
        let (text, finish_reason) = match choice {
            Some(c) => {
                let reason = match c.finish_reason.as_deref() {
                    Some("stop") => FinishReason::Stop,
                    Some("length") => FinishReason::Length,
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
        if !self.default_model.is_empty() {
            self.default_model.clone()
        } else {
            // vLLM requires a model to be specified
            "default".to_string()
        }
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
impl ModelProvider for VllmProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.get_model(&request);
        let vllm_request = self.to_vllm_request(request, &model);

        debug!("VllmProvider: generating with model {}", model);
        let start = Instant::now();

        let response = self.client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(self.build_headers())
            .json(&vllm_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("vLLM request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "vLLM API error: {} - {}",
                status, body
            )));
        }

        let chat_response: VllmChatResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("vLLM response parse error: {}", e))
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        Ok(self.from_vllm_response(chat_response, latency_ms))
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<ChunkStream> {
        // For now, return a non-streaming response wrapped as a stream
        // TODO: Implement proper streaming using vLLM's streaming API
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
        // vLLM supports embeddings via the /v1/embeddings endpoint
        let model = request.model.unwrap_or_else(|| self.default_model.clone());

        debug!("VllmProvider: embedding with model {}", model);

        let embed_request = serde_json::json!({
            "model": model,
            "input": request.text,
        });

        let response = self.client
            .post(format!("{}/v1/embeddings", self.base_url))
            .headers(self.build_headers())
            .json(&embed_request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("vLLM embed request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "vLLM embed API error: {} - {}",
                status, body
            )));
        }

        #[derive(Debug, Deserialize)]
        struct VllmEmbedResponse {
            data: Vec<VllmEmbedData>,
            model: String,
            usage: VllmUsage,
        }

        #[derive(Debug, Deserialize)]
        struct VllmEmbedData {
            embedding: Vec<f64>,
            index: u32,
        }

        let embed_response: VllmEmbedResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("vLLM embed response parse error: {}", e))
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
        Ok(ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            family: None,
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
            .get(format!("{}/v1/models", self.base_url))
            .headers(self.build_headers())
            .timeout(Duration::from_secs(10))
            .send()
            .await;

        let latency_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) if resp.status().is_success() => {
                Ok(HealthStatus::healthy("vLLM server is reachable")
                    .with_latency(latency_ms))
            }
            Ok(resp) => {
                Ok(HealthStatus::unhealthy(format!(
                    "vLLM server returned status {}",
                    resp.status()
                ))
                .with_latency(latency_ms))
            }
            Err(e) => {
                Ok(HealthStatus::unhealthy(format!(
                    "vLLM server unreachable: {}",
                    e
                ))
                .with_latency(latency_ms))
            }
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let response = self.client
            .get(format!("{}/v1/models", self.base_url))
            .headers(self.build_headers())
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("vLLM list models error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Provider(format!(
                "vLLM list models API error: {} - {}",
                status, body
            )));
        }

        let models_response: VllmModelsResponse = response.json().await.map_err(|e| {
            Error::Provider(format!("vLLM models response parse error: {}", e))
        })?;

        let models: Vec<ModelInfo> = models_response
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
                family: None,
                parameters: None,
                quantization: None,
                context_length: None,
                size_bytes: None,
                capabilities: vec!["chat".to_string()],
            })
            .collect();

        Ok(models)
    }

    fn provider_name(&self) -> &str {
        "vllm"
    }

    fn default_model(&self) -> Option<&str> {
        if self.default_model.is_empty() {
            None
        } else {
            Some(&self.default_model)
        }
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

/// Create a vLLM provider from configuration
pub fn create_vllm_provider(
    base_url: &str,
    api_key: Option<&str>,
    default_model: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<VllmProvider> {
    VllmProvider::with_options(base_url, api_key, default_model, timeout_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = VllmProvider::new("http://localhost:8000");
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_with_options() {
        let provider = VllmProvider::with_options(
            "http://localhost:8000",
            Some("test-key"),
            Some("llama-3"),
            Some(60),
        );
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.default_model(), Some("llama-3"));
    }
}