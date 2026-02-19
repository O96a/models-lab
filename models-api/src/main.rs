//! LLM API Server

use axum::routing::get;
use models_api::create_router;
use models_core::Config;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::default();

    // Create app state - import from handlers module
    let state = Arc::new(models_api::handlers::AppState::new(config));

    // Create router
    let app = create_router(state)
        .layer(TraceLayer::new_for_http());

    // Get port
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .unwrap_or(8080);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Server starting on {}", addr);

    // Start server
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}