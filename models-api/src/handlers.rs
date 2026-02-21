//! API handlers

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use models_core::{
    Config, EvaluationStorage,
    InMemoryStorage, ModelProvider,
};
use models_ollama::{ChatRequest, EmbeddingRequest, OllamaClient};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Application state
pub struct AppState {
    pub config: Config,
    pub ollama: OllamaClient,
    pub benchmark_registry: models_benchmark::BenchmarkRegistry,
    pub storage: Arc<InMemoryStorage>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let ollama = OllamaClient::new(&config.ollama.base_url)
            .expect("Failed to create Ollama client");
        Self {
            config,
            ollama,
            benchmark_registry: models_benchmark::BenchmarkRegistry::with_builtin(),
            storage: Arc::new(InMemoryStorage::new()),
        }
    }
}

// ============================================================================
// Health Handler
// ============================================================================

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

// ============================================================================
// Models Handler
// ============================================================================

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

// ============================================================================
// Inference Handler
// ============================================================================

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

// ============================================================================
// Evaluation Handler
// ============================================================================

/// Evaluation handler
pub struct EvaluationHandler;

#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    /// Provider to use (ollama, openai)
    pub provider: Option<String>,
    /// Model to evaluate
    pub model: Option<String>,
    /// Benchmarks to run
    pub benchmarks: Option<Vec<String>>,
    /// Maximum samples per benchmark
    pub max_samples: Option<usize>,
    /// Temperature for generation
    pub temperature: Option<f64>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Number of few-shot examples
    pub num_few_shot: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub overall_score: f64,
    pub total_samples: usize,
    pub total_correct: usize,
    pub benchmark_results: Vec<BenchmarkResultDto>,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkResultDto {
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub accuracy: f64,
    pub total_samples: usize,
    pub correct_count: usize,
    pub mean_latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkInfoDto {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
}

impl EvaluationHandler {
    /// Run an evaluation
    pub async fn evaluate(
        State(state): State<Arc<AppState>>,
        Json(request): Json<EvaluateRequest>,
    ) -> impl IntoResponse {
        let provider_name = request.provider.unwrap_or_else(|| "ollama".to_string());
        let benchmarks = request.benchmarks.unwrap_or_else(|| vec!["mmlu".to_string()]);

        // Create provider
        let provider: Box<dyn ModelProvider> = match provider_name.as_str() {
            "ollama" => {
                match models_providers::create_ollama_provider(
                    Some(&state.config.ollama.base_url),
                    request.model.as_deref(),
                    None,
                ) {
                    Ok(p) => Box::new(p),
                    Err(e) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response();
                    }
                }
            }
            _ => {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": format!("Unknown provider: {}", provider_name) }))).into_response();
            }
        };

        // Run benchmarks
        let mut results = Vec::new();
        let mut total_samples = 0usize;
        let mut total_correct = 0usize;

        for benchmark_id in &benchmarks {
            if let Some(benchmark) = state.benchmark_registry.get(benchmark_id) {
                let config = models_benchmark::BenchmarkConfig {
                    max_samples: request.max_samples,
                    temperature: request.temperature.unwrap_or(0.0),
                    max_tokens: request.max_tokens.unwrap_or(1024),
                    num_few_shot: request.num_few_shot.unwrap_or(5),
                    ..Default::default()
                };

                match benchmark.run(provider.as_ref(), config).await {
                    Ok(result) => {
                        total_samples += result.statistics.total_samples;
                        total_correct += result.statistics.correct_count;

                        results.push(BenchmarkResultDto {
                            benchmark_id: result.benchmark_id.clone(),
                            benchmark_name: result.benchmark_name.clone(),
                            accuracy: result.statistics.accuracy,
                            total_samples: result.statistics.total_samples,
                            correct_count: result.statistics.correct_count,
                            mean_latency_ms: result.statistics.mean_latency_ms,
                        });
                    }
                    Err(e) => {
                        tracing::error!("Benchmark {} failed: {}", benchmark_id, e);
                    }
                }
            }
        }

        let overall_score = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };

        Json(EvaluateResponse {
            overall_score,
            total_samples,
            total_correct,
            benchmark_results: results,
        }).into_response()
    }

    /// List available benchmarks
    pub async fn list_benchmarks(
        State(state): State<Arc<AppState>>,
    ) -> impl IntoResponse {
        let benchmarks: Vec<BenchmarkInfoDto> = state.benchmark_registry
            .list()
            .into_iter()
            .filter_map(|id| {
                state.benchmark_registry.get(id).map(|b| BenchmarkInfoDto {
                    id: id.to_string(),
                    name: b.name().to_string(),
                    category: format!("{:?}", b.category()).to_lowercase(),
                    description: b.description().to_string(),
                })
            })
            .collect();

        Json(benchmarks)
    }

    /// List evaluation reports
    pub async fn list_reports(
        State(state): State<Arc<AppState>>,
    ) -> impl IntoResponse {
        let query = models_core::QueryParams::new().with_limit(10);
        match state.storage.list_evaluation_reports(query).await {
            Ok(reports) => Json(reports).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }

    /// List benchmark results
    pub async fn list_results(
        State(state): State<Arc<AppState>>,
    ) -> impl IntoResponse {
        let query = models_core::QueryParams::new().with_limit(100);
        match state.storage.list_benchmark_results(query).await {
            Ok(results) => Json(results).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
        }
    }
}