//! API routes

use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::handlers::{AppState, ModelsHandler, InferenceHandler, HealthHandler};

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(HealthHandler::health))
        .route("/api/v1/models", get(ModelsHandler::list))
        .route("/api/v1/models/:name", get(ModelsHandler::get))
        .route("/api/v1/models/:name/pull", post(ModelsHandler::pull))
        .route("/api/v1/models/:name", delete(ModelsHandler::delete))
        .route("/api/v1/inference", post(InferenceHandler::chat))
        .route("/api/v1/embeddings", post(InferenceHandler::embed))
        .layer(CorsLayer::permissive())
        .with_state(state)
}