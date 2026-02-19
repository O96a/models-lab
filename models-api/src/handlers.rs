//! API handlers

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use models_core::Config;
use models_ollama::{ChatRequest, EmbeddingRequest, OllamaClient};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Application state
#[derive(Debug)]
pub struct AppState {
    pub config: Config,
    pub ollama: OllamaClient,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let ollama = OllamaClient::new(&config.ollama.base_url)
            .expect("Failed to create Ollama client");
        Self { config, ollama }
    }
}

/// Health handler
pub struct HealthHandler;

impl HealthHandler {
    pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
        let ollama_healthy = state.ollama.health_check().await.unwrap_or(false);

        let status = if ollama_healthy { "healthy" } else { "degraded" };

        Json(serde_json::json!({
            "status": status,
            "ollama": ollama_healthy,
            "version": env!("CARGO_PKG_VERSION")
        }))
    }
}

/// Models handler
pub struct ModelsHandler;

impl ModelsHandler {
    pub async fn list(State(state): State<Arc<AppState>>) -> impl IntoResponse {
        match state.ollama.list_models().await {
            Ok(models) => Json(models).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }

    pub async fn get(
        State(state): State<Arc<AppState>>,
        Path(name): Path<String>,
    ) -> impl IntoResponse {
        match state.ollama.get_model_info(&name).await {
            Ok(Some(model)) => Json(model).into_response(),
            Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Model not found" }))).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }

    pub async fn pull(
        State(state): State<Arc<AppState>>,
        Path(name): Path<String>,
    ) -> impl IntoResponse {
        match state.ollama.pull_model(&name).await {
            Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "message": format!("Model {} pulled successfully", name) }))).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }

    pub async fn delete(
        State(state): State<Arc<AppState>>,
        Path(name): Path<String>,
    ) -> impl IntoResponse {
        match state.ollama.delete_model(&name).await {
            Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "message": format!("Model {} deleted", name) }))).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }
}

/// Inference handler
pub struct InferenceHandler;

#[derive(Debug, Deserialize)]
pub struct ChatRequestDto {
    pub model: Option<String>,
    pub messages: Vec<MessageDto>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct MessageDto {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseDto {
    pub model: String,
    pub content: String,
    pub done: bool,
}

impl InferenceHandler {
    pub async fn chat(
        State(state): State<Arc<AppState>>,
        Json(request): Json<ChatRequestDto>,
    ) -> impl IntoResponse {
        let model = request.model.unwrap_or_else(|| state.config.ollama.default_model.clone());

        let mut chat_request = ChatRequest::new(&model);

        for msg in request.messages {
            match msg.role.as_str() {
                "user" => chat_request = chat_request.with_user_message(&msg.content),
                "system" => chat_request = chat_request.with_system_message(&msg.content),
                _ => {}
            }
        }

        if let Some(temp) = request.temperature {
            chat_request = chat_request.with_temperature(temp);
        }

        match state.ollama.chat(chat_request).await {
            Ok(response) => Json(ChatResponseDto {
                model: response.model,
                content: response.message.content,
                done: response.done,
            }).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }

    pub async fn embed(
        State(state): State<Arc<AppState>>,
        Json(request): Json<EmbeddingRequestDto>,
    ) -> impl IntoResponse {
        let model = request.model.unwrap_or_else(|| state.config.ollama.default_embedding_model.clone());

        let embedding_request = EmbeddingRequest::new(&model, &request.prompt);

        match state.ollama.embed(embedding_request).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequestDto {
    pub model: Option<String>,
    pub prompt: String,
}