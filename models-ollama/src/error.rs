//! Error types for Ollama client

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Connection failed to {url}: {message}")]
    Connection { url: String, message: String },

    #[error("Timeout after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Streaming error: {0}")]
    Streaming(String),
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Parse(err.to_string())
    }
}