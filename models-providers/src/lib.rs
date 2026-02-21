//! Provider adapters for Models Lab
//!
//! This crate provides implementations of the `ModelProvider` trait for various
//! LLM backends, including Ollama, OpenAI, vLLM, and more.

pub mod ollama;
pub mod openai;

pub use ollama::{OllamaProvider, create_ollama_provider};
pub use openai::{OpenAIProvider, create_openai_provider};

use models_core::providers::{ModelProvider, ProviderConfig, ProviderType};
use models_core::error::{Error, Result};

/// Create a provider from a configuration
pub fn create_provider(config: &ProviderConfig) -> Result<Box<dyn ModelProvider>> {
    match &config.provider_type {
        ProviderType::Ollama { base_url, default_model, timeout_seconds } => {
            let provider = create_ollama_provider(
                base_url.as_deref(),
                default_model.as_deref(),
                *timeout_seconds,
            )?;
            Ok(Box::new(provider))
        }
        ProviderType::OpenAI { api_key, organization, default_model, base_url, timeout_seconds } => {
            let provider = create_openai_provider(
                api_key,
                organization.as_deref(),
                default_model.as_deref(),
                base_url.as_deref(),
                *timeout_seconds,
            )?;
            Ok(Box::new(provider))
        }
        _ => Err(Error::Config(format!(
            "Provider type '{}' not yet implemented",
            config.name
        ))),
    }
}

/// Create a provider by name from a factory
pub fn create_provider_from_factory(
    name: &str,
    factory: &models_core::providers::ProviderFactory,
) -> Result<Box<dyn ModelProvider>> {
    let config = factory
        .get_config(name)
        .ok_or_else(|| Error::Config(format!("Provider '{}' not found in factory", name)))?;
    create_provider(config)
}
