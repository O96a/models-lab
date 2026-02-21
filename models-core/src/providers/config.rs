//! Provider configuration and factory
//!
//! This module defines the configuration types for providers and a factory
//! for creating provider instances from configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::domain::ProviderKind;
use crate::error::{Error, Result};

// ============================================================================
// Provider Type
// ============================================================================

/// Type of provider (tagged enum for configuration)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProviderType {
    /// Local Ollama instance
    Ollama {
        /// Base URL for the Ollama API
        base_url: Option<String>,
        /// Default model to use
        default_model: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// vLLM server (OpenAI-compatible API)
    Vllm {
        /// Base URL for the vLLM API
        base_url: String,
        /// API key (if required)
        api_key: Option<String>,
        /// Default model to use
        default_model: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// LM Studio (OpenAI-compatible API)
    LmStudio {
        /// Base URL for the LM Studio API
        base_url: Option<String>,
        /// Default model to use
        default_model: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// OpenAI API
    OpenAI {
        /// API key
        api_key: String,
        /// Organization ID (optional)
        organization: Option<String>,
        /// Default model to use
        default_model: Option<String>,
        /// Base URL override (for proxies)
        base_url: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// Anthropic API
    Anthropic {
        /// API key
        api_key: String,
        /// Default model to use
        default_model: Option<String>,
        /// Base URL override
        base_url: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// Google AI (Gemini)
    Google {
        /// API key
        api_key: String,
        /// Default model to use
        default_model: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// Cohere API
    Cohere {
        /// API key
        api_key: String,
        /// Default model to use
        default_model: Option<String>,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
    /// Custom provider (generic OpenAI-compatible)
    Custom {
        /// Provider name for display
        name: String,
        /// Base URL for the API
        base_url: String,
        /// API key (optional)
        api_key: Option<String>,
        /// Default model to use
        default_model: Option<String>,
        /// API format to use
        api_format: ApiFormat,
        /// Request timeout in seconds
        timeout_seconds: Option<u64>,
    },
}

impl ProviderType {
    /// Get the provider kind from the type
    pub fn kind(&self) -> ProviderKind {
        match self {
            ProviderType::Ollama { .. } => ProviderKind::Ollama,
            ProviderType::Vllm { .. } => ProviderKind::Local,
            ProviderType::LmStudio { .. } => ProviderKind::Local,
            ProviderType::OpenAI { .. } => ProviderKind::OpenAI,
            ProviderType::Anthropic { .. } => ProviderKind::Anthropic,
            ProviderType::Google { .. } => ProviderKind::Google,
            ProviderType::Cohere { .. } => ProviderKind::Cohere,
            ProviderType::Custom { .. } => ProviderKind::Local,
        }
    }

    /// Get the default model for this provider type
    pub fn default_model(&self) -> Option<&str> {
        match self {
            ProviderType::Ollama { default_model, .. } => default_model.as_deref(),
            ProviderType::Vllm { default_model, .. } => default_model.as_deref(),
            ProviderType::LmStudio { default_model, .. } => default_model.as_deref(),
            ProviderType::OpenAI { default_model, .. } => default_model.as_deref(),
            ProviderType::Anthropic { default_model, .. } => default_model.as_deref(),
            ProviderType::Google { default_model, .. } => default_model.as_deref(),
            ProviderType::Cohere { default_model, .. } => default_model.as_deref(),
            ProviderType::Custom { default_model, .. } => default_model.as_deref(),
        }
    }

    /// Get the timeout in seconds
    pub fn timeout_seconds(&self) -> u64 {
        let timeout = match self {
            ProviderType::Ollama { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::Vllm { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::LmStudio { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::OpenAI { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::Anthropic { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::Google { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::Cohere { timeout_seconds, .. } => *timeout_seconds,
            ProviderType::Custom { timeout_seconds, .. } => *timeout_seconds,
        };
        timeout.unwrap_or(120) // Default 2 minute timeout
    }
}

// ============================================================================
// API Format
// ============================================================================

/// API format for custom providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiFormat {
    /// OpenAI-compatible chat completions API
    OpenAI,
    /// Ollama native API
    Ollama,
    /// Anthropic messages API
    Anthropic,
}

impl Default for ApiFormat {
    fn default() -> Self {
        Self::OpenAI
    }
}

// ============================================================================
// Provider Config
// ============================================================================

/// Configuration for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Unique name for this provider configuration
    pub name: String,
    /// The provider type with its specific settings
    #[serde(flatten)]
    pub provider_type: ProviderType,
    /// Additional headers to send with requests
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Enable debug logging for this provider
    #[serde(default)]
    pub debug: bool,
}

impl ProviderConfig {
    /// Create a new provider config
    pub fn new(name: impl Into<String>, provider_type: ProviderType) -> Self {
        Self {
            name: name.into(),
            provider_type,
            extra_headers: HashMap::new(),
            debug: false,
        }
    }

    /// Create an Ollama provider config
    pub fn ollama(base_url: Option<&str>) -> Self {
        Self::new(
            "ollama",
            ProviderType::Ollama {
                base_url: base_url.map(|s| s.to_string()),
                default_model: None,
                timeout_seconds: None,
            },
        )
    }

    /// Create an OpenAI provider config
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(
            "openai",
            ProviderType::OpenAI {
                api_key: api_key.into(),
                organization: None,
                default_model: Some("gpt-4o-mini".to_string()),
                base_url: None,
                timeout_seconds: None,
            },
        )
    }

    /// Add an extra header
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(key.into(), value.into());
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
}

// ============================================================================
// Provider Factory
// ============================================================================

/// Factory for creating provider instances
pub struct ProviderFactory {
    /// Registered provider configurations
    configs: HashMap<String, ProviderConfig>,
}

impl ProviderFactory {
    /// Create a new empty factory
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Create a factory with default configurations
    pub fn with_defaults() -> Self {
        let mut factory = Self::new();
        // Register default Ollama config
        factory.register(ProviderConfig::ollama(None));
        factory
    }

    /// Register a provider configuration
    pub fn register(&mut self, config: ProviderConfig) {
        self.configs.insert(config.name.clone(), config);
    }

    /// Get a provider configuration by name
    pub fn get_config(&self, name: &str) -> Option<&ProviderConfig> {
        self.configs.get(name)
    }

    /// List all registered provider names
    pub fn list_providers(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }

    /// Create a provider instance from a configuration
    ///
    /// Note: This returns a boxed trait object. The actual provider
    /// implementation is in the `models-providers` crate.
    pub fn create(&self, name: &str) -> Result<Box<dyn super::ModelProvider>> {
        let config = self
            .configs
            .get(name)
            .ok_or_else(|| Error::Config(format!("Provider '{}' not found", name)))?;

        // The actual provider creation is handled in models-providers
        // This is a placeholder that returns an error suggesting to use
        // the models-providers crate
        Err(Error::Config(format!(
            "Provider creation for '{}' requires the models-providers crate. \
             Use models_providers::create_provider(config) instead.",
            config.name
        )))
    }
}

impl Default for ProviderFactory {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_kind() {
        let ollama = ProviderType::Ollama {
            base_url: None,
            default_model: None,
            timeout_seconds: None,
        };
        assert_eq!(ollama.kind(), ProviderKind::Ollama);

        let openai = ProviderType::OpenAI {
            api_key: "test".to_string(),
            organization: None,
            default_model: None,
            base_url: None,
            timeout_seconds: None,
        };
        assert_eq!(openai.kind(), ProviderKind::OpenAI);
    }

    #[test]
    fn test_provider_config_ollama() {
        let config = ProviderConfig::ollama(Some("http://localhost:11434"));
        assert_eq!(config.name, "ollama");
        assert_eq!(config.provider_type.kind(), ProviderKind::Ollama);
    }

    #[test]
    fn test_provider_config_openai() {
        let config = ProviderConfig::openai("sk-test");
        assert_eq!(config.name, "openai");
        assert_eq!(config.provider_type.kind(), ProviderKind::OpenAI);
        assert_eq!(
            config.provider_type.default_model(),
            Some("gpt-4o-mini")
        );
    }

    #[test]
    fn test_provider_factory() {
        let factory = ProviderFactory::with_defaults();
        assert!(factory.get_config("ollama").is_some());
        assert!(factory.get_config("nonexistent").is_none());
    }

    #[test]
    fn test_provider_config_serialization() {
        let config = ProviderConfig::ollama(Some("http://localhost:11434"));
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"name\":\"ollama\""));
        assert!(json.contains("\"type\":\"ollama\""));
    }
}
