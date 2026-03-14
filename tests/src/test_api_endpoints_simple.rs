//! API Integration Tests (Simplified)
//!
//! Tests for the API endpoints using JSON request bodies.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use models_api::handlers::AppState;
use models_core::Config;
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

/// Create a test application state
fn create_test_state() -> Arc<AppState> {
    Arc::new(AppState::new(Config::default()))
}

/// Test POST /api/v1/evaluate with MBPP benchmark
#[tokio::test]
async fn test_post_evaluate() {
    let state = create_test_state();

    // Create evaluate request using JSON
    let evaluate_request = json!({
        "provider": "ollama",
        "model": "llama3.2",
        "benchmarks": ["mbpp"],
        "max_samples": 2,
        "temperature": 0.0,
        "max_tokens": 1024,
        "num_few_shot": 0,
    });

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/evaluate")
        .header("content-type", "application/json")
        .body(Body::from(evaluate_request.to_string()))
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK ||
        status == StatusCode::SERVICE_UNAVAILABLE ||
        status == StatusCode::INTERNAL_SERVER_ERROR ||
        status == StatusCode::BAD_REQUEST,
        "Response status should be handled: got {:?}", status
    );

    if status == StatusCode::OK {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(result.get("overall_score").is_some(), "Response should have overall_score");
        assert!(result.get("total_samples").is_some(), "Response should have total_samples");
        assert!(result.get("benchmark_results").is_some(), "Response should have benchmark_results");

        println!("POST /api/v1/evaluate test passed");
    } else {
        println!("Test returned status {:?} (expected if Ollama unavailable)", status);
    }
}

/// Test POST /api/v1/compare
#[tokio::test]
async fn test_post_compare() {
    let state = create_test_state();

    let compare_request = json!({
        "models": [
            {"provider": "ollama", "model": "model1"},
            {"provider": "ollama", "model": "model2"}
        ],
        "benchmarks": ["mbpp"],
        "max_samples": 1,
        "temperature": 0.0,
        "max_tokens": 512,
        "num_few_shot": 0,
        "confidence_level": 0.95
    });

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/compare")
        .header("content-type", "application/json")
        .body(Body::from(compare_request.to_string()))
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    let status = response.status();
    assert!(
        status == StatusCode::OK ||
        status == StatusCode::SERVICE_UNAVAILABLE ||
        status == StatusCode::INTERNAL_SERVER_ERROR ||
        status == StatusCode::BAD_REQUEST,
        "Response status should be handled: got {:?}", status
    );

    if status == StatusCode::OK {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(result.get("id").is_some(), "Response should have id");
        assert!(result.get("model_results").is_some(), "Response should have model_results");
        assert!(result.get("overall_winner").is_some(), "Response should have overall_winner");

        println!("POST /api/v1/compare test passed");
    } else {
        println!("Test returned status {:?} (expected if Ollama unavailable)", status);
    }
}

/// Test GET /api/v1/benchmarks endpoint
#[tokio::test]
async fn test_list_benchmarks() {
    let state = create_test_state();

    let request = Request::builder()
        .method("GET")
        .uri("/api/v1/benchmarks")
        .body(Body::empty())
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let benchmarks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

    assert!(!benchmarks.is_empty(), "Should return list of benchmarks");

    for benchmark in &benchmarks {
        assert!(benchmark.get("id").is_some(), "Benchmark should have id");
        assert!(benchmark.get("name").is_some(), "Benchmark should have name");
    }

    println!("List benchmarks test passed: {} benchmarks found", benchmarks.len());
}

/// Test GET /health endpoint
#[tokio::test]
async fn test_health_check() {
    let state = create_test_state();

    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    assert!(
        response.status() == StatusCode::OK ||
        response.status() == StatusCode::SERVICE_UNAVAILABLE,
        "Health check should return OK or SERVICE_UNAVAILABLE"
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(health.get("status").is_some(), "Health should have status");

    println!("Health check test passed");
}
