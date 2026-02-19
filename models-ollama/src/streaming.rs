//! Streaming support for chat completions

use futures::Stream;
use serde::Deserialize;
use std::pin::Pin;

use crate::models::ChatMessage;

/// Streaming chat response chunk
#[derive(Debug, Clone, Deserialize)]
pub struct ChatStreamChunk {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
}

/// Chat stream type
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamChunk, crate::Error>> + Send>>;