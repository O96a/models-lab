//! Universal provider trait and types
//!
//! This module defines the `ModelProvider` trait that all LLM providers must implement,
//! along with the request/response types for generation operations.

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::error::Result;

pub mod config;

pub use config::{ProviderConfig, ProviderFactory, ProviderType, ApiFormat};

// ============================================================================
// Generation Types
// ============================================================================

/// Request for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt to generate from
    pub prompt: String,
    /// System prompt to prepend (for chat-based models)
    pub system_prompt: Option<String>,
    /// Sampling temperature (0.0 to 2.0, typically)
    pub temperature: Option<f64>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Stop sequences that halt generation
    pub stop_sequences: Option<Vec<String>>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f64>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// Presence penalty (encourages new topics)
    pub presence_penalty: Option<f64>,
    /// Frequency penalty (discourages repetition)
    pub frequency_penalty: Option<f64>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to stream the response
    pub stream: bool,
    /// Additional provider-specific parameters
    #[serde(flatten)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl GenerateRequest {
    /// Create a new generate request with the given prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            stop_sequences: None,
            top_p: None,
            top_k: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            stream: false,
            extra_params: HashMap::new(),
        }
    }

    /// Set the system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the stop sequences
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Enable or disable streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add an extra parameter
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(key.into(), value);
        self
    }
}

/// Response from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// The generated text
    pub text: String,
    /// Token usage statistics
    pub tokens_used: Option<TokenUsage>,
    /// Total latency in milliseconds
    pub latency_ms: u64,
    /// Time to first token in milliseconds (for streaming)
    pub time_to_first_token_ms: Option<u64>,
    /// Why generation stopped
    pub finish_reason: FinishReason,
    /// Model name used (may differ from requested)
    pub model_name: String,
    /// Additional metadata from the provider
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GenerateResponse {
    /// Create a new generate response
    pub fn new(text: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens_used: None,
            latency_ms: 0,
            time_to_first_token_ms: None,
            finish_reason: FinishReason::Stop,
            model_name: model_name.into(),
            metadata: HashMap::new(),
        }
    }

    /// Set token usage
    pub fn with_tokens(mut self, tokens: TokenUsage) -> Self {
        self.tokens_used = Some(tokens);
        self
    }

    /// Set latency
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = reason;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u64,
    /// Number of tokens in the completion
    pub completion_tokens: u64,
    /// Total tokens (prompt + completion)
    pub total_tokens: u64,
}

impl TokenUsage {
    pub fn new(prompt: u64, completion: u64) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// Reason why generation finished
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop (end of text or stop sequence)
    Stop,
    /// Maximum tokens reached
    Length,
    /// Content filtered by provider
    ContentFilter,
    /// Error occurred
    Error,
    /// Unknown reason
    Unknown,
}

impl Default for FinishReason {
    fn default() -> Self {
        Self::Unknown
    }
}

// ============================================================================
// Model Information
// ============================================================================

/// Information about a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model family (e.g., "llama", "gpt", "claude")
    pub family: Option<String>,
    /// Parameter count (e.g., "7B", "70B")
    pub parameters: Option<String>,
    /// Quantization level (e.g., "Q4_0", "FP16")
    pub quantization: Option<String>,
    /// Context window size
    pub context_length: Option<u64>,
    /// Model size in bytes
    pub size_bytes: Option<u64>,
    /// Supported capabilities
    pub capabilities: Vec<String>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            family: None,
            parameters: None,
            quantization: None,
            context_length: None,
            size_bytes: None,
            capabilities: Vec::new(),
        }
    }
}

// ============================================================================
// Health Status
// ============================================================================

/// Health status of a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether the provider is healthy
    pub healthy: bool,
    /// Status message
    pub message: String,
    /// Latency to the provider in milliseconds
    pub latency_ms: Option<u64>,
    /// Additional details
    #[serde(flatten)]
    pub details: HashMap<String, serde_json::Value>,
}

impl HealthStatus {
    pub fn healthy(message: impl Into<String>) -> Self {
        Self {
            healthy: true,
            message: message.into(),
            latency_ms: None,
            details: HashMap::new(),
        }
    }

    pub fn unhealthy(message: impl Into<String>) -> Self {
        Self {
            healthy: false,
            message: message.into(),
            latency_ms: None,
            details: HashMap::new(),
        }
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = Some(ms);
        self
    }
}

// ============================================================================
// Embedding Types
// ============================================================================

/// Request for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// Text to embed
    pub text: String,
    /// Model to use for embedding
    pub model: Option<String>,
}

impl EmbedRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Response from embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    /// The embedding vector
    pub embedding: Vec<f64>,
    /// Model used
    pub model: String,
    /// Token count for the input
    pub token_count: Option<u64>,
}

impl EmbedResponse {
    pub fn new(embedding: Vec<f64>, model: impl Into<String>) -> Self {
        Self {
            embedding,
            model: model.into(),
            token_count: None,
        }
    }
}

// ============================================================================
// Streaming Types
// ============================================================================

/// A chunk of a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The text chunk
    pub text: String,
    /// Whether this is the final chunk
    pub done: bool,
    /// Token usage (only on final chunk)
    pub tokens_used: Option<TokenUsage>,
    /// Finish reason (only on final chunk)
    pub finish_reason: Option<FinishReason>,
}

/// Type alias for a boxed stream of chunks
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>;

// ============================================================================
// ModelProvider Trait
// ============================================================================

/// Trait that all LLM providers must implement
///
/// This trait defines the common interface for text generation, embedding,
/// and model management across different LLM backends.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Generate text from a prompt
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Generate text with streaming
    ///
    /// Returns a stream of chunks that can be processed as they arrive.
    async fn generate_stream(&self, request: GenerateRequest) -> Result<ChunkStream>;

    /// Generate embeddings for text
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse>;

    /// Get information about a specific model
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo>;

    /// Check if the provider is healthy and reachable
    async fn health_check(&self) -> Result<HealthStatus>;

    /// List available models
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Get the provider name/type
    fn provider_name(&self) -> &str;

    /// Get the default model for this provider
    fn default_model(&self) -> Option<&str> {
        None
    }
}
