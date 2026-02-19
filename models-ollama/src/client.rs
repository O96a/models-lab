//! Ollama API client

use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::models::*;
use crate::streaming::ChatStream;

const DEFAULT_TIMEOUT: u64 = 120;

/// Ollama API client
#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    client: Client,
    timeout: Duration,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT))
            .build()
            .map_err(|e| Error::Connection {
                url: base_url.clone(),
                message: e.to_string(),
            })?;

        Ok(Self {
            base_url,
            client,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT),
        })
    }

    /// Create client with custom timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout = Duration::from_secs(timeout_secs);
        self
    }

    /// Check if Ollama is reachable
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        debug!("Health check: {}", url);

        let response = self.client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| Error::Connection {
                url: self.base_url.clone(),
                message: e.to_string(),
            })?;

        Ok(response.status().is_success())
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        info!("Listing models from {}", url);

        let response = self.client
            .get(&url)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| Error::Api(e.to_string()))?;

        let data: ModelListResponse = response.json().await?;
        Ok(data.models)
    }

    /// Get info about a specific model
    pub async fn get_model_info(&self, name: &str) -> Result<Option<ModelInfo>> {
        let models = self.list_models().await?;
        Ok(models.into_iter().find(|m| m.name == name || m.name.starts_with(&format!("{}:", name))))
    }

    /// Send a chat completion request
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let mut request = request;
        request.stream = Some(false);

        debug!("Sending chat request to {} with model {}", url, request.model);

        let response = self.client
            .post(&url)
            .json(&request)
            .timeout(self.timeout)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    Error::ModelNotFound(request.model.clone())
                } else {
                    Error::Api(e.to_string())
                }
            })?;

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    /// Generate embeddings
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = format!("{}/api/embeddings", self.base_url);
        debug!("Generating embeddings for model {}", request.model);

        let response = self.client
            .post(&url)
            .json(&request)
            .timeout(self.timeout)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| Error::Api(e.to_string()))?;

        let embedding: EmbeddingResponse = response.json().await?;
        Ok(embedding)
    }

    /// Pull a model
    pub async fn pull_model(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        info!("Pulling model: {}", name);

        let request = PullRequest::new(name);

        let response = self.client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(3600)) // 1 hour for large models
            .send()
            .await?
            .error_for_status()
            .map_err(|e| Error::Api(e.to_string()))?;

        // Consume the stream
        let _ = response.bytes().await?;
        info!("Model {} pulled successfully", name);
        Ok(())
    }

    /// Delete a model
    pub async fn delete_model(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/delete", self.base_url);
        info!("Deleting model: {}", name);

        let response = self.client
            .delete(&url)
            .json(&serde_json::json!({ "name": name }))
            .send()
            .await?
            .error_for_status()
            .map_err(|e| Error::Api(e.to_string()))?;

        let _ = response.bytes().await?;
        info!("Model {} deleted successfully", name);
        Ok(())
    }

    /// Test a model with a simple prompt
    pub async fn test_model(&self, model: &str) -> Result<String> {
        let request = ChatRequest::new(model)
            .with_user_message("Say 'Hello, I am working!' and nothing else.");

        let response = self.chat(request).await?;
        Ok(response.message.content)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OllamaClient::new("http://localhost:11434");
        assert!(client.is_ok());
    }
}