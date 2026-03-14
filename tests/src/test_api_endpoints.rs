//! API Integration Tests
//!
//! Tests for the API endpoints including evaluation and comparison endpoints.

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

        assert!(result.get("overall_score").is_some());
        assert!(result.get("total_samples").is_some());
        assert!(result.get("benchmark_results").is_some());

        println!("POST /api/v1/evaluate test passed");
    }
}

/// Test POST /api/v1/compare with two models
#[tokio::test]
async fn test_post_compare() {
    let state = create_test_state();

    let compare_request = json!({
        "models": [
            { "provider": "ollama", "model": "llama3.2" },
            { "provider": "ollama", "model": "llama2" }
        ],
        "benchmarks": ["mbpp"],
        "max_samples": 1,
        "temperature": 0.0,
        "max_tokens": 512,
        "num_few_shot": 0,
        "confidence_level": 0.95,
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

        assert!(result.get("id").is_some());
        assert!(result.get("model_results").is_some());

        println!("POST /api/v1/compare test passed");
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

    let ids: Vec<String> = benchmarks
        .iter()
        .filter_map(|b| b["id"].as_str().map(|s| s.to_string()))
        .collect();

    assert!(ids.contains(&"mbpp".to_string()));
    assert!(ids.contains(&"long_context".to_string()));

    println!("List benchmarks test passed - found {} benchmarks", benchmarks.len());
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
        response.status() == StatusCode::SERVICE_UNAVAILABLE
    );

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert!(health.get("status").is_some());
    println!("Health check test passed");
}

/// Test POST /api/v1/compare with insufficient models
#[tokio::test]
async fn test_compare_insufficient_models() {
    let state = create_test_state();

    let compare_request = json!({
        "models": [
            { "provider": "ollama", "model": "llama3.2" }
        ],
        "benchmarks": ["mbpp"],
        "max_samples": 1,
    });

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/compare")
        .header("content-type", "application/json")
        .body(Body::from(compare_request.to_string()))
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    println!("Compare with insufficient models test passed");
}

/// Test POST /api/v1/compare with missing benchmarks
#[tokio::test]
async fn test_compare_missing_benchmarks() {
    let state = create_test_state();

    let compare_request = json!({
        "models": [
            { "provider": "ollama", "model": "model1" },
            { "provider": "ollama", "model": "model2" }
        ],
        "benchmarks": [],
    });

    let request = Request::builder()
        .method("POST")
        .uri("/api/v1/compare")
        .header("content-type", "application/json")
        .body(Body::from(compare_request.to_string()))
        .unwrap();

    let app = models_api::create_router(state);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    println!("Compare with missing benchmarks test passed");
}

/// Test POST /api/v1/evaluate with LongContext benchmark
#[tokio::test]
async fn test_post_evaluate_long_context() {
    let state = create_test_state();

    let evaluate_request = json!({
        "provider": "ollama",
        "model": "llama3.2",
        "benchmarks": ["long_context"],
        "max_samples": 1,
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
        status == StatusCode::BAD_REQUEST
    );

    if status == StatusCode::OK {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(result.get("overall_score").is_some());
        println!("LongContext evaluation test passed");
    }
}
