//! LLM Ollama - Ollama API client
//!
//! A complete client for the Ollama API, providing chat completions,
//! embeddings, and model management.

pub mod client;
pub mod error;
pub mod models;
pub mod streaming;

pub use client::OllamaClient;
pub use error::{Error, Result};
pub use models::{ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ModelInfo};
pub use streaming::ChatStream;