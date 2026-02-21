//! Ollama provider adapter
//!
//! Implements the `ModelProvider` trait for Ollama using the existing `OllamaClient`.

use async_trait::async_trait;
use futures::StreamExt;
use std::time::Instant;
use tracing::debug;

use models_core::providers::{
    ChunkStream, EmbedRequest, EmbedResponse, FinishReason, GenerateRequest, GenerateResponse,
    HealthStatus, ModelInfo, ModelProvider, StreamChunk, TokenUsage,
};
use models_core::error::{Error, Result};
use models_ollama::{ChatRequest, ChatResponse, EmbeddingRequest, OllamaClient};

/// Ollama provider that wraps the OllamaClient
pub struct OllamaProvider {
    /// The underlying Ollama client
    client: OllamaClient,
    /// Default model to use if not specified
    default_model: Option<String>,
}

impl OllamaProvider {
    /// Create a new Ollama provider
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let client = OllamaClient::new(base_url).map_err(|e| {
            Error::Provider(format!("Failed to create Ollama client: {}", e))
        })?;
        Ok(Self {
            client,
            default_model: None,
        })
    }

    /// Create a provider with a default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Create from an existing OllamaClient
    pub fn from_client(client: OllamaClient) -> Self {
        Self {
            client,
            default_model: None,
        }
    }

    /// Get the underlying client
    pub fn client(&self) -> &OllamaClient {
        &self.client
    }

    /// Convert GenerateRequest to Ollama ChatRequest
    fn to_chat_request(&self, request: GenerateRequest, model: &str) -> ChatRequest {
        let mut chat_request = ChatRequest::new(model);

        // Add system prompt if provided
        if let Some(ref system) = request.system_prompt {
            chat_request = chat_request.with_system_message(system);
        }

        // Add the user prompt
        chat_request = chat_request.with_user_message(&request.prompt);

        // Set temperature if provided
        if let Some(temp) = request.temperature {
            chat_request = chat_request.with_temperature(temp as f32);
        }

        chat_request
    }

    /// Convert Ollama ChatResponse to GenerateResponse
    fn from_chat_response(&self, response: ChatResponse, latency_ms: u64) -> GenerateResponse {
        let tokens_used = match (response.prompt_eval_count, response.eval_count) {
            (Some(prompt), Some(completion)) => Some(TokenUsage::new(
                prompt as u64,
                completion as u64,
            )),
            _ => None,
        };

        let finish_reason = if response.done {
            FinishReason::Stop
        } else {
            FinishReason::Unknown
        };

        GenerateResponse::new(response.message.content, response.model)
            .with_tokens_opt(tokens_used)
            .with_latency(latency_ms)
            .with_finish_reason(finish_reason)
    }

    /// Get the model to use (from request or default)
    fn get_model(&self, request: &GenerateRequest) -> Result<String> {
        // Check extra_params for model
        if let Some(model) = request.extra_params.get("model") {
            if let Some(model_str) = model.as_str() {
                return Ok(model_str.to_string());
            }
        }

        // Use default model
        self.default_model
            .clone()
            .ok_or_else(|| Error::InvalidRequest("No model specified and no default model set".to_string()))
    }
}

/// Helper trait to add optional tokens
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
impl ModelProvider for OllamaProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.get_model(&request)?;
        let chat_request = self.to_chat_request(request, &model);

        debug!("OllamaProvider: generating with model {}", model);
        let start = Instant::now();

        let response = self.client.chat(chat_request).await.map_err(|e| {
            Error::Provider(format!("Ollama chat error: {}", e))
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        Ok(self.from_chat_response(response, latency_ms))
    }

    async fn generate_stream(&self, request: GenerateRequest) -> Result<ChunkStream> {
        // For now, return a non-streaming response wrapped as a stream
        // TODO: Implement proper streaming using Ollama's streaming API
        let response = self.generate(request).await?;

        let chunk = StreamChunk {
            text: response.text,
            done: true,
            tokens_used: response.tokens_used,
            finish_reason: Some(response.finish_reason),
        };

        // Create a stream that yields one chunk
        let stream = futures::stream::once(async move { Ok(chunk) });
        Ok(Box::pin(stream))
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse> {
        let model = request.model.unwrap_or_else(|| {
            self.default_model.clone().unwrap_or_else(|| "nomic-embed-text".to_string())
        });

        debug!("OllamaProvider: embedding with model {}", model);

        let embed_request = EmbeddingRequest::new(&model, &request.text);
        let response = self.client.embed(embed_request).await.map_err(|e| {
            Error::Provider(format!("Ollama embed error: {}", e))
        })?;

        // Convert f32 to f64
        let embedding: Vec<f64> = response.embedding.into_iter().map(|f| f as f64).collect();

        Ok(EmbedResponse::new(embedding, model))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let ollama_info = self.client.get_model_info(model_id).await.map_err(|e| {
            Error::Provider(format!("Ollama get_model_info error: {}", e))
        })?;

        let info = ollama_info.ok_or_else(|| {
            Error::ModelNotFound(model_id.to_string())
        })?;

        // Convert Ollama ModelInfo to our ModelInfo
        let mut capabilities = Vec::new();
        capabilities.push("chat".to_string());
        capabilities.push("embed".to_string());

        let (parameters, quantization) = if let Some(ref details) = info.details {
            (Some(details.parameter_size.clone()), Some(details.quantization_level.clone()))
        } else {
            (None, None)
        };

        Ok(ModelInfo {
            id: info.name.clone(),
            name: info.name,
            family: info.details.as_ref().map(|d| d.family.clone()),
            parameters,
            quantization,
            context_length: None,
            size_bytes: Some(info.size),
            capabilities,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();
        let healthy = self.client.health_check().await.map_err(|e| {
            Error::Provider(format!("Ollama health_check error: {}", e))
        })?;

        let latency_ms = start.elapsed().as_millis() as u64;

        if healthy {
            Ok(HealthStatus::healthy("Ollama is reachable")
                .with_latency(latency_ms))
        } else {
            Ok(HealthStatus::unhealthy("Ollama is not reachable")
                .with_latency(latency_ms))
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.client.list_models().await.map_err(|e| {
            Error::Provider(format!("Ollama list_models error: {}", e))
        })?;

        let result: Vec<ModelInfo> = models
            .into_iter()
            .map(|m| {
                let mut capabilities = vec!["chat".to_string(), "embed".to_string()];
                let (parameters, quantization) = if let Some(ref details) = m.details {
                    (Some(details.parameter_size.clone()), Some(details.quantization_level.clone()))
                } else {
                    (None, None)
                };

                ModelInfo {
                    id: m.name.clone(),
                    name: m.name,
                    family: m.details.as_ref().map(|d| d.family.clone()),
                    parameters,
                    quantization,
                    context_length: None,
                    size_bytes: Some(m.size),
                    capabilities,
                }
            })
            .collect();

        Ok(result)
    }

    fn provider_name(&self) -> &str {
        "ollama"
    }

    fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }
}

/// Create an Ollama provider from a ProviderConfig
pub fn create_ollama_provider(
    base_url: Option<&str>,
    default_model: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<OllamaProvider> {
    let url = base_url.unwrap_or("http://localhost:11434");
    let mut provider = OllamaProvider::new(url)?;

    if let Some(model) = default_model {
        provider = provider.with_default_model(model);
    }

    // Note: timeout handling would require modifying the OllamaClient
    // For now, we use the default timeout

    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434");
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_with_default_model() {
        let provider = OllamaProvider::new("http://localhost:11434")
            .unwrap()
            .with_default_model("llama3.2");

        assert_eq!(provider.default_model(), Some("llama3.2"));
        assert_eq!(provider.provider_name(), "ollama");
    }
}
