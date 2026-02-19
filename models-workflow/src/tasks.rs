//! Task definitions

use async_trait::async_trait;
use models_core::{InferenceRequest, InferenceResponse, Result};
use models_ollama::{ChatRequest, OllamaClient};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Task execution context
#[derive(Debug, Clone)]
pub struct TaskContext {
    pub variables: HashMap<String, String>,
    pub config: TaskConfig,
}

impl Default for TaskContext {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            config: TaskConfig::default(),
        }
    }
}

/// Task configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Task trait for workflow execution
#[async_trait]
pub trait Task: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    async fn execute(&self, context: TaskContext) -> TaskResult;
}

/// Inference task - executes LLM inference
pub struct InferenceTask {
    id: String,
    name: String,
    client: OllamaClient,
}

impl InferenceTask {
    pub fn new(client: OllamaClient) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: "inference".to_string(),
            client,
        }
    }
}

#[async_trait]
impl Task for InferenceTask {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, context: TaskContext) -> TaskResult {
        use std::time::Instant;

        let start = Instant::now();

        // Get prompt from context
        let prompt = context.variables.get("prompt")
            .cloned()
            .unwrap_or_else(|| "Hello, how can I help you?".to_string());

        // Create chat request
        let request = ChatRequest::new(&context.config.model)
            .with_user_message(&prompt)
            .with_temperature(context.config.temperature);

        // Execute inference
        match self.client.chat(request).await {
            Ok(response) => TaskResult {
                task_id: self.id.clone(),
                success: true,
                output: Some(response.message.content),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
            },
            Err(e) => TaskResult {
                task_id: self.id.clone(),
                success: false,
                output: None,
                error: Some(e.to_string()),
                duration_ms: start.elapsed().as_millis() as u64,
            },
        }
    }
}