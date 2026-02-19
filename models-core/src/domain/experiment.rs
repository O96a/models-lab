//! Experiment domain types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Entity;

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl Default for ExperimentStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub model: String,
    pub provider: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub top_p: f32,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2".to_string(),
            provider: "ollama".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
            top_p: 0.9,
        }
    }
}

/// Experiment entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub config: ExperimentConfig,
    pub status: ExperimentStatus,
    pub metrics: Option<ExperimentMetrics>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl Experiment {
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            config: ExperimentConfig::default(),
            status: ExperimentStatus::Pending,
            metrics: None,
            created_at: now,
            updated_at: now,
            completed_at: None,
        }
    }

    pub fn with_config(mut self, config: ExperimentConfig) -> Self {
        self.config = config;
        self
    }

    pub fn start(mut self) -> Self {
        self.status = ExperimentStatus::Running;
        self.updated_at = Utc::now();
        self
    }

    pub fn complete(mut self, metrics: ExperimentMetrics) -> Self {
        let now = Utc::now();
        self.status = ExperimentStatus::Completed;
        self.metrics = Some(metrics);
        self.completed_at = Some(now);
        self.updated_at = now;
        self
    }
}

/// Experiment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_duration_ms: u64,
    pub average_latency_ms: f64,
}