//! API routes

use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::handlers::{
    AppState, ModelsHandler, InferenceHandler, HealthHandler, EvaluationHandler, ComparisonHandler,
};

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health
        .route("/health", get(HealthHandler::health))
        // Models
        .route("/api/v1/models", get(ModelsHandler::list))
        .route("/api/v1/models/:name", get(ModelsHandler::get))
        .route("/api/v1/models/:name/pull", post(ModelsHandler::pull))
        .route("/api/v1/models/:name", delete(ModelsHandler::delete))
        // Inference
        .route("/api/v1/inference", post(InferenceHandler::chat))
        .route("/api/v1/embeddings", post(InferenceHandler::embed))
        // Evaluation
        .route("/api/v1/evaluate", post(EvaluationHandler::evaluate))
        .route("/api/v1/benchmarks", get(EvaluationHandler::list_benchmarks))
        .route("/api/v1/reports", get(EvaluationHandler::list_reports))
        .route("/api/v1/results", get(EvaluationHandler::list_results))
        // Comparison
        .route("/api/v1/compare", post(ComparisonHandler::compare))
        .layer(CorsLayer::permissive())
        .with_state(state)
}