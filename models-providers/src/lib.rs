//! Provider adapters for Models Lab
//!
//! This crate provides implementations of the `ModelProvider` trait for various
//! LLM backends, including Ollama, OpenAI, vLLM, Anthropic, and LM Studio.

pub mod anthropic;
pub mod lmstudio;
pub mod ollama;
pub mod openai;
pub mod vllm;

pub use anthropic::{AnthropicProvider, create_anthropic_provider};
pub use lmstudio::{LmStudioProvider, create_lmstudio_provider};
pub use ollama::{OllamaProvider, create_ollama_provider};
pub use openai::{OpenAIProvider, create_openai_provider};
pub use vllm::{VllmProvider, create_vllm_provider};

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
        ProviderType::Vllm { base_url, api_key, default_model, timeout_seconds } => {
            let provider = create_vllm_provider(
                base_url,
                api_key.as_deref(),
                default_model.as_deref(),
                *timeout_seconds,
            )?;
            Ok(Box::new(provider))
        }
        ProviderType::Anthropic { api_key, default_model, base_url, timeout_seconds } => {
            let provider = create_anthropic_provider(
                api_key,
                default_model.as_deref(),
                base_url.as_deref(),
                *timeout_seconds,
            )?;
            Ok(Box::new(provider))
        }
        ProviderType::LmStudio { base_url, default_model, timeout_seconds } => {
            let provider = create_lmstudio_provider(
                base_url.as_deref(),
                default_model.as_deref(),
                *timeout_seconds,
            )?;
            Ok(Box::new(provider))
        }
        ProviderType::Custom { name, base_url, api_key, default_model, api_format, timeout_seconds } => {
            // Custom providers use OpenAI-compatible format by default
            match api_format {
                models_core::providers::ApiFormat::OpenAI => {
                    let provider = create_vllm_provider(
                        base_url,
                        api_key.as_deref(),
                        default_model.as_deref(),
                        *timeout_seconds,
                    )?;
                    Ok(Box::new(provider))
                }
                models_core::providers::ApiFormat::Ollama => {
                    let provider = create_ollama_provider(
                        Some(base_url),
                        default_model.as_deref(),
                        *timeout_seconds,
                    )?;
                    Ok(Box::new(provider))
                }
                models_core::providers::ApiFormat::Anthropic => {
                    let api_key = api_key.as_ref()
                        .ok_or_else(|| Error::Config(format!(
                            "Custom provider '{}' using Anthropic format requires an API key",
                            name
                        )))?;
                    let provider = create_anthropic_provider(
                        api_key,
                        default_model.as_deref(),
                        Some(base_url),
                        *timeout_seconds,
                    )?;
                    Ok(Box::new(provider))
                }
            }
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
