//! LLM API - REST API server
//!
//! Axum-based REST API for Models Lab.

pub mod handlers;
mod routes;

pub use routes::create_router;