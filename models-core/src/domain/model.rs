//! Model domain types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Entity;

/// Model provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelProvider {
    Ollama,
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Local,
}

impl Default for ModelProvider {
    fn default() -> Self {
        Self::Ollama
    }
}

/// Model status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Available,
    Downloading,
    NotDownloaded,
    Error,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: Uuid,
    pub name: String,
    pub provider: ModelProvider,
    pub status: ModelStatus,
    pub size_bytes: Option<u64>,
    pub parameters: Option<String>,
    pub quantization: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Model {
    pub fn new(name: impl Into<String>, provider: ModelProvider) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            provider,
            status: ModelStatus::NotDownloaded,
            size_bytes: None,
            parameters: None,
            quantization: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_status(mut self, status: ModelStatus) -> Self {
        self.status = status;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_size(mut self, size: u64) -> Self {
        self.size_bytes = Some(size);
        self
    }
}