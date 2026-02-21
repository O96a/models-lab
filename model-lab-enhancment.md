# Product Requirements Document: Models Lab
## Universal LLM Evaluation Platform

**Version:** 1.0  
**Date:** February 2026  
**Status:** Active Development

---

## 1. Executive Summary

### 1.1 Product Vision
Models Lab is a comprehensive LLM evaluation platform designed to provide 100% objective assessment of any language model from any provider (Ollama, vLLM, LM Studio, OpenAI, Anthropic, HuggingFace, etc.). The platform measures models across all critical dimensions: speed, reasoning power, token efficiency, accuracy, hallucination rates, drift, context handling, and more.

### 1.2 Core Value Proposition
- **Universal Compatibility:** Evaluate any LLM from any provider through unified adapter layer
- **Comprehensive Metrics:** 15+ evaluation dimensions covering all aspects of model performance
- **Reproducible Benchmarks:** Standardized test suites with deterministic scoring
- **Production Ready:** Built-in monitoring, distributed evaluation, and CI/CD integration

### 1.3 Target Users
- ML Engineers evaluating models for production deployment
- Researchers comparing model architectures
- Organizations selecting LLMs for specific use cases
- Open source community benchmarking local models

---

## 2. Technical Architecture

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Models Lab Core Engine                        │
│                     (Rust - Performance Critical)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Provider Layer  │      │ Benchmark Engine │      │  Metrics Engine  │
│  (Abstraction)   │      │  (Orchestration) │      │  (Calculation)   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   • Ollama       │      │  • Reasoning     │      │  • Speed         │
│   • vLLM         │      │  • Accuracy      │      │  • Hallucination │
│   • LM Studio    │      │  • Context       │      │  • Drift         │
│   • OpenAI API   │      │  • Safety        │      │  • Cost          │
│   • Anthropic    │      │  • Coding        │      │  • Quality       │
│   • HuggingFace  │      │  • Math          │      │  • Calibration   │
│   • Custom       │      │  • RAG           │      │  • Consistency   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Storage & Persistence      │
                    │  • PostgreSQL (Results)       │
                    │  • Redis (Cache)              │
                    │  • S3 (Datasets)              │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐             ┌───────────────┐
            │   REST API    │             │   CLI Tool    │
            │  (Axum 0.7)   │             │  (Clap 4.5)   │
            └───────────────┘             └───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │   Web Dashboard       │
                        │  (Svelte/React +      │
                        │   TailwindCSS)        │
                        └───────────────────────┘
```

### 2.2 Technology Stack

**Backend:**
- **Language:** Rust 1.75+
- **Web Framework:** Axum 0.7
- **Async Runtime:** Tokio 1.35
- **Database:** PostgreSQL 16 with pgvector extension
- **Cache:** Redis 7.2
- **Message Queue:** Optional (RabbitMQ for distributed mode)

**Frontend:**
- **Framework:** Svelte 4 or React 18
- **Styling:** TailwindCSS 3.4
- **Charts:** Chart.js or Recharts
- **State Management:** Svelte stores or Zustand

**Infrastructure:**
- **Containerization:** Docker + Docker Compose
- **Orchestration:** Kubernetes (optional for scale)
- **Monitoring:** Prometheus + Grafana
- **CI/CD:** GitHub Actions

---

## 3. Core Components Specification

### 3.1 Universal Provider Adapter

**File:** `models-core/src/providers/mod.rs`

```rust
/// Universal trait that all LLM providers must implement
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Generate text completion
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;
    
    /// Generate embeddings for text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Get model metadata and capabilities
    async fn get_model_info(&self) -> Result<ModelInfo>;
    
    /// Health check for provider availability
    async fn health_check(&self) -> Result<HealthStatus>;
    
    /// Stream generation (optional)
    async fn generate_stream(&self, request: GenerateRequest) 
        -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
}

/// Standardized generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt/messages to send to the model
    pub prompt: String,
    
    /// Optional system prompt
    pub system_prompt: Option<String>,
    
    /// Sampling temperature (0.0 to 2.0)
    pub temperature: f64,
    
    /// Maximum tokens to generate
    pub max_tokens: usize,
    
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    
    /// Top-p sampling parameter
    pub top_p: Option<f64>,
    
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    
    /// Presence penalty
    pub presence_penalty: Option<f64>,
    
    /// Frequency penalty
    pub frequency_penalty: Option<f64>,
    
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    
    /// Enable streaming
    pub stream: bool,
    
    /// Provider-specific options
    pub extra_params: HashMap<String, serde_json::Value>,
}

/// Standardized generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated text
    pub text: String,
    
    /// Token usage statistics
    pub tokens_used: TokenUsage,
    
    /// Latency in milliseconds
    pub latency_ms: u64,
    
    /// Time to first token (TTFT) in milliseconds
    pub time_to_first_token_ms: Option<u64>,
    
    /// Why generation stopped
    pub finish_reason: FinishReason,
    
    /// Model that generated the response
    pub model_name: String,
    
    /// Provider-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,           // Natural completion
    Length,         // Max tokens reached
    ContentFilter,  // Content policy violation
    Error(String),  // Error occurred
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub supports_streaming: bool,
    pub supports_function_calling: bool,
    pub cost_per_1k_input_tokens: Option<f64>,
    pub cost_per_1k_output_tokens: Option<f64>,
    pub parameter_count: Option<String>,  // e.g., "3B", "7B"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub latency_ms: u64,
    pub error_message: Option<String>,
}
```

**Provider Implementations:**

**File:** `models-providers/src/ollama.rs`

```rust
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    default_model: String,
}

impl OllamaProvider {
    pub fn new(base_url: String, default_model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            default_model,
        }
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        let mut ttft = None;
        
        let ollama_request = json!({
            "model": self.default_model,
            "prompt": request.prompt,
            "stream": false,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "seed": request.seed,
                "top_p": request.top_p,
                "top_k": request.top_k,
            }
        });
        
        let response = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&ollama_request)
            .send()
            .await?;
        
        let ollama_response: OllamaGenerateResponse = response.json().await?;
        
        Ok(GenerateResponse {
            text: ollama_response.response,
            tokens_used: TokenUsage {
                prompt_tokens: ollama_response.prompt_eval_count.unwrap_or(0),
                completion_tokens: ollama_response.eval_count.unwrap_or(0),
                total_tokens: ollama_response.prompt_eval_count.unwrap_or(0) 
                    + ollama_response.eval_count.unwrap_or(0),
            },
            latency_ms: start.elapsed().as_millis() as u64,
            time_to_first_token_ms: ttft,
            finish_reason: if ollama_response.done {
                FinishReason::Stop
            } else {
                FinishReason::Length
            },
            model_name: self.default_model.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = json!({
            "model": self.default_model,
            "prompt": text,
        });
        
        let response = self.client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&request)
            .send()
            .await?;
        
        let ollama_response: OllamaEmbeddingResponse = response.json().await?;
        Ok(ollama_response.embedding)
    }
    
    async fn get_model_info(&self) -> Result<ModelInfo> {
        let response = self.client
            .post(format!("{}/api/show", self.base_url))
            .json(&json!({"name": self.default_model}))
            .send()
            .await?;
        
        let ollama_info: OllamaModelInfo = response.json().await?;
        
        Ok(ModelInfo {
            name: self.default_model.clone(),
            version: "latest".to_string(),
            context_window: ollama_info.modelinfo.get("context_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize,
            max_output_tokens: 4096,
            supports_streaming: true,
            supports_function_calling: false,
            cost_per_1k_input_tokens: Some(0.0),  // Local models are free
            cost_per_1k_output_tokens: Some(0.0),
            parameter_count: ollama_info.details.parameter_size.clone(),
        })
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();
        
        match self.client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(_) => Ok(HealthStatus {
                is_healthy: true,
                latency_ms: start.elapsed().as_millis() as u64,
                error_message: None,
            }),
            Err(e) => Ok(HealthStatus {
                is_healthy: false,
                latency_ms: start.elapsed().as_millis() as u64,
                error_message: Some(e.to_string()),
            }),
        }
    }
}
```

**File:** `models-providers/src/vllm.rs`

```rust
pub struct VLLMProvider {
    client: reqwest::Client,
    base_url: String,
    model_name: String,
    api_key: Option<String>,
}

#[async_trait]
impl ModelProvider for VLLMProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // vLLM uses OpenAI-compatible API
        let start = Instant::now();
        
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(key) = &self.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", key).parse().unwrap(),
            );
        }
        
        let vllm_request = json!({
            "model": self.model_name,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop_sequences,
        });
        
        let response = self.client
            .post(format!("{}/v1/completions", self.base_url))
            .headers(headers)
            .json(&vllm_request)
            .send()
            .await?;
        
        let vllm_response: VLLMCompletionResponse = response.json().await?;
        let choice = &vllm_response.choices[0];
        
        Ok(GenerateResponse {
            text: choice.text.clone(),
            tokens_used: TokenUsage {
                prompt_tokens: vllm_response.usage.prompt_tokens,
                completion_tokens: vllm_response.usage.completion_tokens,
                total_tokens: vllm_response.usage.total_tokens,
            },
            latency_ms: start.elapsed().as_millis() as u64,
            time_to_first_token_ms: None,
            finish_reason: match choice.finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                other => FinishReason::Error(other.to_string()),
            },
            model_name: self.model_name.clone(),
            metadata: HashMap::new(),
        })
    }
    
    // ... implement other trait methods
}
```

**File:** `models-providers/src/openai.rs`

```rust
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model_name: String,
    organization: Option<String>,
}

#[async_trait]
impl ModelProvider for OpenAIProvider {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let start = Instant::now();
        
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        if let Some(org) = &self.organization {
            headers.insert("OpenAI-Organization", org.parse().unwrap());
        }
        
        let messages = if let Some(system) = request.system_prompt {
            vec![
                json!({"role": "system", "content": system}),
                json!({"role": "user", "content": request.prompt}),
            ]
        } else {
            vec![json!({"role": "user", "content": request.prompt})]
        };
        
        let openai_request = json!({
            "model": self.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stop": request.stop_sequences,
            "seed": request.seed,
        });
        
        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .headers(headers)
            .json(&openai_request)
            .send()
            .await?;
        
        let openai_response: OpenAIChatCompletionResponse = response.json().await?;
        let choice = &openai_response.choices[0];
        
        Ok(GenerateResponse {
            text: choice.message.content.clone(),
            tokens_used: TokenUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
            },
            latency_ms: start.elapsed().as_millis() as u64,
            time_to_first_token_ms: None,
            finish_reason: match choice.finish_reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                "content_filter" => FinishReason::ContentFilter,
                other => FinishReason::Error(other.to_string()),
            },
            model_name: self.model_name.clone(),
            metadata: HashMap::new(),
        })
    }
    
    // ... implement other trait methods
}
```

**Provider Configuration:**

**File:** `models-core/src/providers/config.rs`

```rust
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub model_name: String,
    pub endpoint: Option<String>,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: usize,
    pub retry_delay_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProviderType {
    Ollama { base_url: String },
    VLLM { base_url: String, api_key: Option<String> },
    LMStudio { base_url: String },
    OpenAI { organization: Option<String> },
    Anthropic,
    HuggingFace { api_key: String },
    AzureOpenAI { 
        endpoint: String,
        deployment_name: String,
        api_version: String,
    },
    Cohere { api_key: String },
    Custom { 
        base_url: String,
        api_format: ApiFormat,
        headers: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ApiFormat {
    OpenAI,      // /v1/chat/completions or /v1/completions
    Anthropic,   // /v1/messages
}

/// Factory to create providers from config
pub struct ProviderFactory;

impl ProviderFactory {
    pub fn create(config: ProviderConfig) -> Result<Box<dyn ModelProvider>> {
        match config.provider_type {
            ProviderType::Ollama { base_url } => {
                Ok(Box::new(OllamaProvider::new(base_url, config.model_name)))
            }
            ProviderType::VLLM { base_url, api_key } => {
                Ok(Box::new(VLLMProvider::new(base_url, config.model_name, api_key)))
            }
            ProviderType::OpenAI { organization } => {
                let api_key = config.api_key
                    .ok_or_else(|| anyhow!("OpenAI requires API key"))?;
                Ok(Box::new(OpenAIProvider::new(api_key, config.model_name, organization)))
            }
            ProviderType::Anthropic => {
                let api_key = config.api_key
                    .ok_or_else(|| anyhow!("Anthropic requires API key"))?;
                Ok(Box::new(AnthropicProvider::new(api_key, config.model_name)))
            }
            // ... other providers
            _ => Err(anyhow!("Provider not yet implemented")),
        }
    }
}
```

### 3.2 Benchmark Engine

**File:** `models-benchmark/src/lib.rs`

```rust
/// Core trait that all benchmarks must implement
#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Unique identifier for the benchmark
    fn id(&self) -> &str;
    
    /// Human-readable name
    fn name(&self) -> &str;
    
    /// Benchmark category
    fn category(&self) -> BenchmarkCategory;
    
    /// Description of what this benchmark tests
    fn description(&self) -> &str;
    
    /// Load the dataset for this benchmark
    async fn load_dataset(&self) -> Result<Dataset>;
    
    /// Run the benchmark against a provider
    async fn run(&self, provider: &dyn ModelProvider) -> Result<BenchmarkResult>;
    
    /// Evaluate a single response
    fn evaluate_response(&self, response: &str, expected: &str, metadata: &HashMap<String, Value>) -> MetricScores;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BenchmarkCategory {
    Reasoning,        // Logical reasoning, problem-solving
    Knowledge,        // Factual knowledge
    Math,             // Mathematical reasoning
    Coding,           // Code generation and understanding
    Hallucination,    // Truthfulness and factuality
    ContextHandling,  // Long context understanding
    Safety,           // Harmful content refusal
    Instruction,      // Instruction following
    Multilingual,     // Multi-language capabilities
    Creativity,       // Creative writing
    RAG,              // Retrieval-augmented generation
    MultiTurn,        // Conversation handling
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub samples: Vec<DataSample>,
    pub metadata: DatasetMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    pub id: String,
    pub input: String,
    pub expected_output: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub difficulty: Option<Difficulty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub version: String,
    pub license: String,
    pub source: String,
    pub num_samples: usize,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub category: BenchmarkCategory,
    pub timestamp: DateTime<Utc>,
    pub model_name: String,
    pub provider: String,
    
    /// Overall score (0.0 to 1.0)
    pub overall_score: f64,
    
    /// Detailed scores per metric
    pub metric_scores: HashMap<String, f64>,
    
    /// Per-sample results
    pub sample_results: Vec<SampleResult>,
    
    /// Aggregate statistics
    pub statistics: BenchmarkStatistics,
    
    /// Configuration used
    pub config: BenchmarkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResult {
    pub sample_id: String,
    pub input: String,
    pub expected: Option<String>,
    pub generated: String,
    pub score: f64,
    pub latency_ms: u64,
    pub tokens_used: usize,
    pub is_correct: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub total_samples: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub avg_score: f64,
    pub median_score: f64,
    pub std_dev: f64,
    pub avg_latency_ms: f64,
    pub total_tokens: usize,
    pub total_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub temperature: f64,
    pub max_tokens: usize,
    pub num_samples: Option<usize>,  // Limit number of samples
    pub seed: Option<u64>,
    pub few_shot_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricScores {
    pub scores: HashMap<String, f64>,
}
```

**Benchmark Registry:**

**File:** `models-benchmark/src/registry.rs`

```rust
/// Central registry for all benchmarks
pub struct BenchmarkRegistry {
    benchmarks: HashMap<String, Box<dyn Benchmark>>,
}

impl BenchmarkRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            benchmarks: HashMap::new(),
        };
        
        // Register all built-in benchmarks
        registry.register_all_builtin();
        registry
    }
    
    pub fn register(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.insert(benchmark.id().to_string(), benchmark);
    }
    
    pub fn get(&self, id: &str) -> Option<&dyn Benchmark> {
        self.benchmarks.get(id).map(|b| b.as_ref())
    }
    
    pub fn list(&self) -> Vec<&dyn Benchmark> {
        self.benchmarks.values().map(|b| b.as_ref()).collect()
    }
    
    pub fn list_by_category(&self, category: BenchmarkCategory) -> Vec<&dyn Benchmark> {
        self.benchmarks
            .values()
            .filter(|b| b.category() == category)
            .map(|b| b.as_ref())
            .collect()
    }
    
    fn register_all_builtin(&mut self) {
        // Reasoning benchmarks
        self.register(Box::new(MMLUBenchmark::new()));
        self.register(Box::new(BBHBenchmark::new()));
        self.register(Box::new(ARCBenchmark::new()));
        
        // Math benchmarks
        self.register(Box::new(GSM8KBenchmark::new()));
        self.register(Box::new(MATHBenchmark::new()));
        
        // Coding benchmarks
        self.register(Box::new(HumanEvalBenchmark::new()));
        self.register(Box::new(MBPPBenchmark::new()));
        
        // Hallucination benchmarks
        self.register(Box::new(TruthfulQABenchmark::new()));
        
        // Context benchmarks
        self.register(Box::new(LongContextBenchmark::new()));
        self.register(Box::new(NeedleInHaystackBenchmark::new()));
        
        // Instruction following
        self.register(Box::new(AlpacaEvalBenchmark::new()));
        
        // Multi-turn conversation
        self.register(Box::new(MTBenchBenchmark::new()));
        
        // Safety
        self.register(Box::new(SafetyBenchmark::new()));
    }
}
```

**Example Benchmark Implementation:**

**File:** `models-benchmark/src/reasoning/mmlu.rs`

```rust
/// MMLU (Massive Multitask Language Understanding) Benchmark
pub struct MMLUBenchmark {
    subjects: Vec<String>,
    num_shots: usize,
}

impl MMLUBenchmark {
    pub fn new() -> Self {
        Self {
            subjects: vec![
                "abstract_algebra".to_string(),
                "anatomy".to_string(),
                "astronomy".to_string(),
                "business_ethics".to_string(),
                "clinical_knowledge".to_string(),
                "college_biology".to_string(),
                "college_chemistry".to_string(),
                "college_computer_science".to_string(),
                "college_mathematics".to_string(),
                "college_medicine".to_string(),
                "college_physics".to_string(),
                "computer_security".to_string(),
                "conceptual_physics".to_string(),
                "econometrics".to_string(),
                "electrical_engineering".to_string(),
                // ... 57 subjects total
            ],
            num_shots: 5,  // 5-shot evaluation
        }
    }
    
    fn format_mmlu_prompt(&self, question: &MMLUQuestion, few_shot_examples: &[MMLUQuestion]) -> String {
        let mut prompt = String::from("Answer the following multiple choice question.\n\n");
        
        // Add few-shot examples
        for example in few_shot_examples {
            prompt.push_str(&format!(
                "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer: {}\n\n",
                example.question,
                example.choices[0],
                example.choices[1],
                example.choices[2],
                example.choices[3],
                example.answer,
            ));
        }
        
        // Add actual question
        prompt.push_str(&format!(
            "Question: {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:",
            question.question,
            question.choices[0],
            question.choices[1],
            question.choices[2],
            question.choices[3],
        ));
        
        prompt
    }
    
    fn parse_answer(&self, response: &str) -> Option<String> {
        // Extract A, B, C, or D from response
        let response = response.trim().to_uppercase();
        
        // Try to find a single letter answer
        for letter in ["A", "B", "C", "D"] {
            if response.starts_with(letter) || response == letter {
                return Some(letter.to_string());
            }
        }
        
        None
    }
}

#[async_trait]
impl Benchmark for MMLUBenchmark {
    fn id(&self) -> &str {
        "mmlu"
    }
    
    fn name(&self) -> &str {
        "MMLU (Massive Multitask Language Understanding)"
    }
    
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Reasoning
    }
    
    fn description(&self) -> &str {
        "Tests knowledge across 57 subjects including STEM, humanities, and social sciences"
    }
    
    async fn load_dataset(&self) -> Result<Dataset> {
        // Load MMLU dataset from disk or download
        let mut samples = Vec::new();
        
        for subject in &self.subjects {
            let subject_file = format!("datasets/mmlu/{}.json", subject);
            let data = tokio::fs::read_to_string(&subject_file).await?;
            let questions: Vec<MMLUQuestion> = serde_json::from_str(&data)?;
            
            for question in questions {
                samples.push(DataSample {
                    id: format!("mmlu_{}_{}", subject, question.id),
                    input: question.question.clone(),
                    expected_output: Some(question.answer.clone()),
                    metadata: hashmap! {
                        "subject" => json!(subject),
                        "choices" => json!(question.choices),
                    },
                    difficulty: None,
                });
            }
        }
        
        Ok(Dataset {
            name: "MMLU".to_string(),
            samples,
            metadata: DatasetMetadata {
                version: "1.0".to_string(),
                license: "MIT".to_string(),
                source: "https://github.com/hendrycks/test".to_string(),
                num_samples: samples.len(),
                tags: vec!["reasoning".to_string(), "knowledge".to_string()],
            },
        })
    }
    
    async fn run(&self, provider: &dyn ModelProvider) -> Result<BenchmarkResult> {
        let dataset = self.load_dataset().await?;
        let start_time = Instant::now();
        
        let mut sample_results = Vec::new();
        let mut total_correct = 0;
        let mut total_latency = 0u64;
        let mut total_tokens = 0usize;
        
        for sample in &dataset.samples {
            // Get few-shot examples (from same subject)
            let subject = sample.metadata.get("subject")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let few_shot_examples = self.get_few_shot_examples(subject, self.num_shots);
            
            // Format prompt
            let question = MMLUQuestion {
                id: sample.id.clone(),
                question: sample.input.clone(),
                choices: sample.metadata.get("choices")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default(),
                answer: sample.expected_output.clone().unwrap_or_default(),
                subject: subject.to_string(),
            };
            
            let prompt = self.format_mmlu_prompt(&question, &few_shot_examples);
            
            // Generate response
            let response = provider.generate(GenerateRequest {
                prompt,
                temperature: 0.0,  // Deterministic for benchmarking
                max_tokens: 10,    // Short answer expected
                ..Default::default()
            }).await?;
            
            // Parse answer
            let predicted_answer = self.parse_answer(&response.text);
            let expected_answer = sample.expected_output.as_ref();
            
            let is_correct = predicted_answer.as_ref() == expected_answer;
            if is_correct {
                total_correct += 1;
            }
            
            total_latency += response.latency_ms;
            total_tokens += response.tokens_used.total_tokens;
            
            sample_results.push(SampleResult {
                sample_id: sample.id.clone(),
                input: sample.input.clone(),
                expected: sample.expected_output.clone(),
                generated: response.text.clone(),
                score: if is_correct { 1.0 } else { 0.0 },
                latency_ms: response.latency_ms,
                tokens_used: response.tokens_used.total_tokens,
                is_correct,
                error: if predicted_answer.is_none() {
                    Some("Failed to parse answer".to_string())
                } else {
                    None
                },
            });
        }
        
        let accuracy = total_correct as f64 / dataset.samples.len() as f64;
        
        Ok(BenchmarkResult {
            benchmark_id: self.id().to_string(),
            benchmark_name: self.name().to_string(),
            category: self.category(),
            timestamp: Utc::now(),
            model_name: provider.get_model_info().await?.name,
            provider: "unknown".to_string(),  // TODO: Track provider type
            overall_score: accuracy,
            metric_scores: hashmap! {
                "accuracy" => accuracy,
            },
            sample_results,
            statistics: BenchmarkStatistics {
                total_samples: dataset.samples.len(),
                correct_count: total_correct,
                accuracy,
                avg_score: accuracy,
                median_score: accuracy,
                std_dev: 0.0,  // TODO: Calculate
                avg_latency_ms: total_latency as f64 / dataset.samples.len() as f64,
                total_tokens,
                total_cost_usd: 0.0,  // TODO: Calculate based on provider pricing
            },
            config: BenchmarkConfig {
                temperature: 0.0,
                max_tokens: 10,
                num_samples: None,
                seed: None,
                few_shot_count: self.num_shots,
            },
        })
    }
    
    fn evaluate_response(&self, response: &str, expected: &str, _metadata: &HashMap<String, Value>) -> MetricScores {
        let predicted = self.parse_answer(response);
        let is_correct = predicted.as_deref() == Some(expected);
        
        MetricScores {
            scores: hashmap! {
                "exact_match" => if is_correct { 1.0 } else { 0.0 },
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MMLUQuestion {
    id: String,
    question: String,
    choices: Vec<String>,
    answer: String,
    subject: String,
}
```

**File:** `models-benchmark/src/hallucination/truthfulqa.rs`

```rust
/// TruthfulQA Benchmark - Tests model truthfulness and hallucination resistance
pub struct TruthfulQABenchmark {
    categories: Vec<String>,
}

impl TruthfulQABenchmark {
    pub fn new() -> Self {
        Self {
            categories: vec![
                "Health".to_string(),
                "Law".to_string(),
                "Finance".to_string(),
                "Politics".to_string(),
                "Science".to_string(),
                "Conspiracies".to_string(),
                "Misconceptions".to_string(),
            ],
        }
    }
    
    fn calculate_truthfulness_score(&self, response: &str, correct_answers: &[String], incorrect_answers: &[String]) -> f64 {
        let response_lower = response.to_lowercase();
        
        // Check if response matches correct answers
        let matches_correct = correct_answers.iter().any(|answer| {
            response_lower.contains(&answer.to_lowercase())
        });
        
        // Check if response contains incorrect information
        let contains_incorrect = incorrect_answers.iter().any(|answer| {
            response_lower.contains(&answer.to_lowercase())
        });
        
        match (matches_correct, contains_incorrect) {
            (true, false) => 1.0,   // Truthful
            (false, true) => 0.0,   // Hallucinates
            (true, true) => 0.5,    // Mixed
            (false, false) => 0.5,  // Unclear
        }
    }
}

#[async_trait]
impl Benchmark for TruthfulQABenchmark {
    fn id(&self) -> &str {
        "truthfulqa"
    }
    
    fn name(&self) -> &str {
        "TruthfulQA"
    }
    
    fn category(&self) -> BenchmarkCategory {
        BenchmarkCategory::Hallucination
    }
    
    fn description(&self) -> &str {
        "Tests whether models generate truthful answers to questions that humans might answer falsely due to misconceptions"
    }
    
    async fn load_dataset(&self) -> Result<Dataset> {
        let data = tokio::fs::read_to_string("datasets/truthfulqa/questions.json").await?;
        let questions: Vec<TruthfulQAQuestion> = serde_json::from_str(&data)?;
        
        let samples = questions.into_iter().map(|q| DataSample {
            id: format!("truthfulqa_{}", q.id),
            input: q.question,
            expected_output: Some(q.correct_answers.join("; ")),
            metadata: hashmap! {
                "category" => json!(q.category),
                "correct_answers" => json!(q.correct_answers),
                "incorrect_answers" => json!(q.incorrect_answers),
            },
            difficulty: None,
        }).collect();
        
        Ok(Dataset {
            name: "TruthfulQA".to_string(),
            samples,
            metadata: DatasetMetadata {
                version: "1.0".to_string(),
                license: "Apache 2.0".to_string(),
                source: "https://github.com/sylinrl/TruthfulQA".to_string(),
                num_samples: samples.len(),
                tags: vec!["hallucination".to_string(), "truthfulness".to_string()],
            },
        })
    }
    
    async fn run(&self, provider: &dyn ModelProvider) -> Result<BenchmarkResult> {
        let dataset = self.load_dataset().await?;
        let mut sample_results = Vec::new();
        let mut total_score = 0.0;
        
        for sample in &dataset.samples {
            let response = provider.generate(GenerateRequest {
                prompt: sample.input.clone(),
                temperature: 0.0,
                max_tokens: 200,
                ..Default::default()
            }).await?;
            
            let correct_answers: Vec<String> = sample.metadata
                .get("correct_answers")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            
            let incorrect_answers: Vec<String> = sample.metadata
                .get("incorrect_answers")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            
            let score = self.calculate_truthfulness_score(
                &response.text,
                &correct_answers,
                &incorrect_answers,
            );
            
            total_score += score;
            
            sample_results.push(SampleResult {
                sample_id: sample.id.clone(),
                input: sample.input.clone(),
                expected: sample.expected_output.clone(),
                generated: response.text,
                score,
                latency_ms: response.latency_ms,
                tokens_used: response.tokens_used.total_tokens,
                is_correct: score >= 0.8,
                error: None,
            });
        }
        
        let avg_score = total_score / dataset.samples.len() as f64;
        
        Ok(BenchmarkResult {
            benchmark_id: self.id().to_string(),
            benchmark_name: self.name().to_string(),
            category: self.category(),
            timestamp: Utc::now(),
            model_name: provider.get_model_info().await?.name,
            provider: "unknown".to_string(),
            overall_score: avg_score,
            metric_scores: hashmap! {
                "truthfulness" => avg_score,
                "hallucination_rate" => 1.0 - avg_score,
            },
            sample_results,
            statistics: BenchmarkStatistics {
                total_samples: dataset.samples.len(),
                correct_count: sample_results.iter().filter(|r| r.is_correct).count(),
                accuracy: avg_score,
                avg_score,
                median_score: avg_score,
                std_dev: 0.0,
                avg_latency_ms: sample_results.iter().map(|r| r.latency_ms).sum::<u64>() as f64 / sample_results.len() as f64,
                total_tokens: sample_results.iter().map(|r| r.tokens_used).sum(),
                total_cost_usd: 0.0,
            },
            config: BenchmarkConfig {
                temperature: 0.0,
                max_tokens: 200,
                num_samples: None,
                seed: None,
                few_shot_count: 0,
            },
        })
    }
    
    fn evaluate_response(&self, response: &str, _expected: &str, metadata: &HashMap<String, Value>) -> MetricScores {
        let correct_answers: Vec<String> = metadata
            .get("correct_answers")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        
        let incorrect_answers: Vec<String> = metadata
            .get("incorrect_answers")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        
        let score = self.calculate_truthfulness_score(response, &correct_answers, &incorrect_answers);
        
        MetricScores {
            scores: hashmap! {
                "truthfulness" => score,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TruthfulQAQuestion {
    id: String,
    question: String,
    category: String,
    correct_answers: Vec<String>,
    incorrect_answers: Vec<String>,
}
```

### 3.3 Metrics Engine

**File:** `models-metrics/src/lib.rs`

```rust
/// Core trait for all metrics
pub trait Metric: Send + Sync {
    /// Unique identifier
    fn name(&self) -> &str;
    
    /// Calculate the metric value
    fn calculate(&self, data: &MetricData) -> f64;
    
    /// Unit of measurement
    fn unit(&self) -> MetricUnit;
    
    /// Higher is better or lower is better?
    fn direction(&self) -> MetricDirection;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricUnit {
    Percentage,           // 0-100%
    Milliseconds,         // Time
    TokensPerSecond,      // Throughput
    Count,                // Raw count
    Score,                // Normalized 0-1
    CostUSD,              // US Dollars
    Ratio,                // Generic ratio
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// Data container for metric calculation
#[derive(Debug, Clone)]
pub struct MetricData {
    pub responses: Vec<GenerateResponse>,
    pub expected_outputs: Vec<Option<String>>,
    pub benchmark_results: Option<BenchmarkResult>,
    pub extra: HashMap<String, Value>,
}

/// Collection of all metrics
pub struct MetricsEngine {
    metrics: Vec<Box<dyn Metric>>,
}

impl MetricsEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            metrics: Vec::new(),
        };
        
        // Register all built-in metrics
        engine.register_all_builtin();
        engine
    }
    
    pub fn register(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }
    
    pub fn calculate_all(&self, data: &MetricData) -> HashMap<String, f64> {
        self.metrics
            .iter()
            .map(|m| (m.name().to_string(), m.calculate(data)))
            .collect()
    }
    
    fn register_all_builtin(&mut self) {
        // Speed metrics
        self.register(Box::new(TokensPerSecondMetric));
        self.register(Box::new(FirstTokenLatencyMetric));
        self.register(Box::new(P95LatencyMetric));
        
        // Accuracy metrics
        self.register(Box::new(ExactMatchMetric));
        self.register(Box::new(F1ScoreMetric));
        self.register(Box::new(BLEUScoreMetric));
        self.register(Box::new(ROUGEScoreMetric));
        
        // Hallucination metrics
        self.register(Box::new(HallucinationRateMetric));
        self.register(Box::new(FactualConsistencyMetric));
        
        // Efficiency metrics
        self.register(Box::new(CostPerTokenMetric));
        self.register(Box::new(QualityPerTokenMetric));
        
        // Consistency metrics
        self.register(Box::new(OutputDriftMetric));
        self.register(Box::new(TemperatureVarianceMetric));
    }
}
```

**Speed Metrics:**

**File:** `models-metrics/src/speed.rs`

```rust
pub struct TokensPerSecondMetric;

impl Metric for TokensPerSecondMetric {
    fn name(&self) -> &str {
        "tokens_per_second"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        let total_tokens: usize = data.responses
            .iter()
            .map(|r| r.tokens_used.completion_tokens)
            .sum();
        
        let total_time_seconds: f64 = data.responses
            .iter()
            .map(|r| r.latency_ms as f64 / 1000.0)
            .sum();
        
        if total_time_seconds > 0.0 {
            total_tokens as f64 / total_time_seconds
        } else {
            0.0
        }
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::TokensPerSecond
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }
}

pub struct FirstTokenLatencyMetric;

impl Metric for FirstTokenLatencyMetric {
    fn name(&self) -> &str {
        "first_token_latency"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        let ttfts: Vec<u64> = data.responses
            .iter()
            .filter_map(|r| r.time_to_first_token_ms)
            .collect();
        
        if ttfts.is_empty() {
            return 0.0;
        }
        
        ttfts.iter().sum::<u64>() as f64 / ttfts.len() as f64
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Milliseconds
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }
}

pub struct P95LatencyMetric;

impl Metric for P95LatencyMetric {
    fn name(&self) -> &str {
        "p95_latency"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        let mut latencies: Vec<u64> = data.responses
            .iter()
            .map(|r| r.latency_ms)
            .collect();
        
        if latencies.is_empty() {
            return 0.0;
        }
        
        latencies.sort();
        let idx = (latencies.len() as f64 * 0.95) as usize;
        latencies[idx.min(latencies.len() - 1)] as f64
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Milliseconds
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }
}
```

**Accuracy Metrics:**

**File:** `models-metrics/src/accuracy.rs`

```rust
pub struct ExactMatchMetric;

impl Metric for ExactMatchMetric {
    fn name(&self) -> &str {
        "exact_match"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        let mut matches = 0;
        let mut total = 0;
        
        for (response, expected) in data.responses.iter().zip(data.expected_outputs.iter()) {
            if let Some(expected) = expected {
                total += 1;
                let generated = response.text.trim().to_lowercase();
                let expected = expected.trim().to_lowercase();
                
                if generated == expected {
                    matches += 1;
                }
            }
        }
        
        if total > 0 {
            matches as f64 / total as f64
        } else {
            0.0
        }
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Percentage
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }
}

pub struct F1ScoreMetric;

impl Metric for F1ScoreMetric {
    fn name(&self) -> &str {
        "f1_score"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        let mut total_f1 = 0.0;
        let mut count = 0;
        
        for (response, expected) in data.responses.iter().zip(data.expected_outputs.iter()) {
            if let Some(expected) = expected {
                let f1 = self.calculate_f1(&response.text, expected);
                total_f1 += f1;
                count += 1;
            }
        }
        
        if count > 0 {
            total_f1 / count as f64
        } else {
            0.0
        }
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }
}

impl F1ScoreMetric {
    fn calculate_f1(&self, prediction: &str, reference: &str) -> f64 {
        let pred_tokens = self.tokenize(prediction);
        let ref_tokens = self.tokenize(reference);
        
        let common: HashSet<_> = pred_tokens.intersection(&ref_tokens).collect();
        
        if pred_tokens.is_empty() || ref_tokens.is_empty() {
            return 0.0;
        }
        
        let precision = common.len() as f64 / pred_tokens.len() as f64;
        let recall = common.len() as f64 / ref_tokens.len() as f64;
        
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }
    
    fn tokenize(&self, text: &str) -> HashSet<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

pub struct BLEUScoreMetric;

impl Metric for BLEUScoreMetric {
    fn name(&self) -> &str {
        "bleu_score"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        // Implement BLEU score calculation
        // This is a simplified version
        let mut total_bleu = 0.0;
        let mut count = 0;
        
        for (response, expected) in data.responses.iter().zip(data.expected_outputs.iter()) {
            if let Some(expected) = expected {
                let bleu = self.calculate_bleu(&response.text, expected);
                total_bleu += bleu;
                count += 1;
            }
        }
        
        if count > 0 {
            total_bleu / count as f64
        } else {
            0.0
        }
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }
}

impl BLEUScoreMetric {
    fn calculate_bleu(&self, candidate: &str, reference: &str) -> f64 {
        // Simplified BLEU calculation (1-gram to 4-gram)
        let candidate_tokens: Vec<&str> = candidate.split_whitespace().collect();
        let reference_tokens: Vec<&str> = reference.split_whitespace().collect();
        
        let mut precisions = Vec::new();
        
        for n in 1..=4 {
            let precision = self.ngram_precision(
                &candidate_tokens,
                &reference_tokens,
                n,
            );
            precisions.push(precision);
        }
        
        // Calculate geometric mean of precisions
        let geometric_mean = precisions.iter().product::<f64>().powf(1.0 / 4.0);
        
        // Apply brevity penalty
        let brevity_penalty = if candidate_tokens.len() < reference_tokens.len() {
            (1.0 - (reference_tokens.len() as f64 / candidate_tokens.len() as f64)).exp()
        } else {
            1.0
        };
        
        brevity_penalty * geometric_mean
    }
    
    fn ngram_precision(&self, candidate: &[&str], reference: &[&str], n: usize) -> f64 {
        if candidate.len() < n {
            return 0.0;
        }
        
        let candidate_ngrams = self.get_ngrams(candidate, n);
        let reference_ngrams = self.get_ngrams(reference, n);
        
        let matches = candidate_ngrams
            .iter()
            .filter(|ngram| reference_ngrams.contains(ngram))
            .count();
        
        matches as f64 / candidate_ngrams.len() as f64
    }
    
    fn get_ngrams(&self, tokens: &[&str], n: usize) -> Vec<Vec<&str>> {
        tokens
            .windows(n)
            .map(|window| window.to_vec())
            .collect()
    }
}
```

**Hallucination Metrics:**

**File:** `models-metrics/src/hallucination.rs`

```rust
use serde_json::json;

pub struct HallucinationRateMetric;

impl Metric for HallucinationRateMetric {
    fn name(&self) -> &str {
        "hallucination_rate"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        // This requires checking if the response contains factual errors
        // In practice, this would use an NLI model or fact-checking system
        
        if let Some(benchmark_result) = &data.benchmark_results {
            // If benchmark has hallucination scores, use those
            if let Some(score) = benchmark_result.metric_scores.get("hallucination_rate") {
                return *score;
            }
        }
        
        // Fallback: estimate based on confidence and coherence
        0.0  // Placeholder
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Percentage
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }
}

pub struct FactualConsistencyMetric;

impl Metric for FactualConsistencyMetric {
    fn name(&self) -> &str {
        "factual_consistency"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        // Use NLI model to check if response is entailed by context
        // This is a placeholder implementation
        
        // In production, you would:
        // 1. Extract claims from the response
        // 2. Check each claim against provided context
        // 3. Use an NLI model (e.g., DeBERTa) to verify entailment
        
        1.0 - self.estimate_hallucination_rate(data)
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::HigherIsBetter
    }
}

impl FactualConsistencyMetric {
    fn estimate_hallucination_rate(&self, data: &MetricData) -> f64 {
        // Simplified heuristic-based estimation
        let mut suspicious_count = 0;
        let total = data.responses.len();
        
        for response in &data.responses {
            if self.contains_suspicious_patterns(&response.text) {
                suspicious_count += 1;
            }
        }
        
        suspicious_count as f64 / total as f64
    }
    
    fn contains_suspicious_patterns(&self, text: &str) -> bool {
        // Check for hedging language that might indicate uncertainty
        let hedging_phrases = [
            "i think", "i believe", "maybe", "possibly",
            "might be", "could be", "i'm not sure",
        ];
        
        let text_lower = text.to_lowercase();
        hedging_phrases.iter().any(|phrase| text_lower.contains(phrase))
    }
}
```

**Drift Detection Metrics:**

**File:** `models-metrics/src/drift.rs`

```rust
pub struct OutputDriftMetric;

impl Metric for OutputDriftMetric {
    fn name(&self) -> &str {
        "output_drift"
    }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        // Measure variance in outputs for the same prompt
        // Requires grouping responses by prompt
        
        let grouped_responses = self.group_by_prompt(data);
        
        if grouped_responses.is_empty() {
            return 0.0;
        }
        
        let mut total_variance = 0.0;
        
        for (_, responses) in grouped_responses {
            if responses.len() < 2 {
                continue;
            }
            
            let variance = self.calculate_response_variance(&responses);
            total_variance += variance;
        }
        
        total_variance / grouped_responses.len() as f64
    }
    
    fn unit(&self) -> MetricUnit {
        MetricUnit::Score
    }
    
    fn direction(&self) -> MetricDirection {
        MetricDirection::LowerIsBetter
    }
}

impl OutputDriftMetric {
    fn group_by_prompt(&self, data: &MetricData) -> HashMap<String, Vec<String>> {
        // Group responses by their prompts
        // This assumes prompts are stored in metadata
        let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
        
        for response in &data.responses {
            if let Some(prompt) = response.metadata.get("prompt") {
                if let Some(prompt_str) = prompt.as_str() {
                    grouped
                        .entry(prompt_str.to_string())
                        .or_insert_with(Vec::new)
                        .push(response.text.clone());
                }
            }
        }
        
        grouped
    }
    
    fn calculate_response_variance(&self, responses: &[String]) -> f64 {
        // Calculate variance using edit distance
        if responses.len() < 2 {
            return 0.0;
        }
        
        let mut distances = Vec::new();
        
        for i in 0..responses.len() {
            for j in (i + 1)..responses.len() {
                let distance = self.levenshtein_distance(&responses[i], &responses[j]);
                let max_len = responses[i].len().max(responses[j].len()) as f64;
                let normalized_distance = distance as f64 / max_len;
                distances.push(normalized_distance);
            }
        }
        
        if distances.is_empty() {
            return 0.0;
        }
        
        // Calculate mean distance as drift measure
        distances.iter().sum::<f64>() / distances.len() as f64
    }
    
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }
        
        matrix[len1][len2]
    }
}
```

### 3.4 Evaluation Orchestrator

**File:** `models-core/src/evaluation/orchestrator.rs`

```rust
/// Main orchestrator for running evaluations
pub struct EvaluationOrchestrator {
    benchmark_registry: BenchmarkRegistry,
    metrics_engine: MetricsEngine,
    storage: Box<dyn EvaluationStorage>,
}

impl EvaluationOrchestrator {
    pub fn new(storage: Box<dyn EvaluationStorage>) -> Self {
        Self {
            benchmark_registry: BenchmarkRegistry::new(),
            metrics_engine: MetricsEngine::new(),
            storage,
        }
    }
    
    /// Run full evaluation suite
    pub async fn run_full_evaluation(
        &self,
        provider: Box<dyn ModelProvider>,
        config: EvaluationConfig,
    ) -> Result<EvaluationReport> {
        let start_time = Instant::now();
        let mut benchmark_results = Vec::new();
        
        // Get benchmarks to run
        let benchmarks = if let Some(categories) = &config.benchmark_categories {
            categories
                .iter()
                .flat_map(|cat| self.benchmark_registry.list_by_category(*cat))
                .collect()
        } else {
            self.benchmark_registry.list()
        };
        
        // Run each benchmark
        for benchmark in benchmarks {
            println!("Running benchmark: {}", benchmark.name());
            
            let result = benchmark.run(provider.as_ref()).await?;
            benchmark_results.push(result.clone());
            
            // Save intermediate result
            self.storage.save_benchmark_result(&result).await?;
        }
        
        // Calculate aggregate metrics
        let overall_score = self.calculate_overall_score(&benchmark_results);
        let category_scores = self.calculate_category_scores(&benchmark_results);
        let detailed_metrics = self.calculate_detailed_metrics(&benchmark_results);
        let cost_analysis = self.calculate_cost_analysis(&benchmark_results, provider.as_ref()).await?;
        
        let report = EvaluationReport {
            id: Uuid::new_v4(),
            model_name: provider.get_model_info().await?.name,
            provider_type: config.provider_type.clone(),
            timestamp: Utc::now(),
            duration_seconds: start_time.elapsed().as_secs(),
            overall_score,
            category_scores,
            detailed_metrics,
            cost_analysis,
            benchmark_results,
            config,
            recommendations: self.generate_recommendations(&category_scores),
        };
        
        // Save final report
        self.storage.save_evaluation_report(&report).await?;
        
        Ok(report)
    }
    
    /// Run specific benchmarks
    pub async fn run_benchmarks(
        &self,
        provider: Box<dyn ModelProvider>,
        benchmark_ids: Vec<String>,
    ) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for benchmark_id in benchmark_ids {
            let benchmark = self.benchmark_registry
                .get(&benchmark_id)
                .ok_or_else(|| anyhow!("Benchmark not found: {}", benchmark_id))?;
            
            let result = benchmark.run(provider.as_ref()).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Compare multiple models
    pub async fn compare_models(
        &self,
        providers: Vec<(String, Box<dyn ModelProvider>)>,
        benchmark_ids: Vec<String>,
    ) -> Result<ComparisonReport> {
        let mut model_results = HashMap::new();
        
        for (model_name, provider) in providers {
            let results = self.run_benchmarks(provider, benchmark_ids.clone()).await?;
            model_results.insert(model_name, results);
        }
        
        Ok(ComparisonReport {
            timestamp: Utc::now(),
            models: model_results.keys().cloned().collect(),
            benchmark_ids,
            results: model_results,
            winner_by_category: self.determine_winners(&model_results),
        })
    }
    
    fn calculate_overall_score(&self, results: &[BenchmarkResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        // Weighted average based on benchmark importance
        let weights = self.get_benchmark_weights();
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;
        
        for result in results {
            let weight = weights.get(&result.category).copied().unwrap_or(1.0);
            weighted_sum += result.overall_score * weight;
            total_weight += weight;
        }
        
        weighted_sum / total_weight
    }
    
    fn get_benchmark_weights(&self) -> HashMap<BenchmarkCategory, f64> {
        hashmap! {
            BenchmarkCategory::Reasoning => 2.0,
            BenchmarkCategory::Hallucination => 2.0,
            BenchmarkCategory::Safety => 1.5,
            BenchmarkCategory::ContextHandling => 1.5,
            BenchmarkCategory::Coding => 1.0,
            BenchmarkCategory::Math => 1.0,
        }
    }
    
    fn calculate_category_scores(&self, results: &[BenchmarkResult]) -> HashMap<BenchmarkCategory, CategoryScore> {
        let mut category_results: HashMap<BenchmarkCategory, Vec<BenchmarkResult>> = HashMap::new();
        
        for result in results {
            category_results
                .entry(result.category)
                .or_insert_with(Vec::new)
                .push(result.clone());
        }
        
        category_results
            .into_iter()
            .map(|(category, results)| {
                let avg_score = results.iter().map(|r| r.overall_score).sum::<f64>() / results.len() as f64;
                (category, CategoryScore {
                    category,
                    score: avg_score,
                    benchmark_results: results,
                })
            })
            .collect()
    }
    
    fn calculate_detailed_metrics(&self, results: &[BenchmarkResult]) -> DetailedMetrics {
        let all_samples: Vec<&SampleResult> = results
            .iter()
            .flat_map(|r| &r.sample_results)
            .collect();
        
        let total_latency: u64 = all_samples.iter().map(|s| s.latency_ms).sum();
        let latencies: Vec<u64> = all_samples.iter().map(|s| s.latency_ms).collect();
        
        DetailedMetrics {
            // Speed metrics
            avg_tokens_per_second: self.calculate_avg_tokens_per_second(results),
            first_token_latency_ms: 0.0,  // TODO: Calculate from responses
            p50_latency_ms: self.calculate_percentile(&latencies, 0.5),
            p95_latency_ms: self.calculate_percentile(&latencies, 0.95),
            p99_latency_ms: self.calculate_percentile(&latencies, 0.99),
            
            // Reasoning metrics
            reasoning_accuracy: self.get_category_score(results, BenchmarkCategory::Reasoning),
            logical_consistency: 0.0,  // TODO
            chain_of_thought_quality: 0.0,  // TODO
            
            // Quality metrics
            exact_match_accuracy: self.calculate_exact_match_accuracy(results),
            f1_score: 0.0,  // TODO
            bleu_score: 0.0,  // TODO
            rouge_score: RougeScores::default(),
            semantic_similarity: 0.0,  // TODO
            
            // Hallucination metrics
            hallucination_rate: 1.0 - self.get_category_score(results, BenchmarkCategory::Hallucination),
            factual_consistency: self.get_category_score(results, BenchmarkCategory::Hallucination),
            groundedness_score: 0.0,  // TODO
            
            // Stability metrics
            output_drift: 0.0,  // TODO
            temperature_variance: 0.0,  // TODO
            seed_reproducibility: 0.0,  // TODO
            
            // Context metrics
            max_context_utilized: 0,  // TODO
            context_recall_rate: self.get_category_score(results, BenchmarkCategory::ContextHandling),
            needle_in_haystack_score: 0.0,  // TODO
            
            // Efficiency metrics
            avg_input_tokens: all_samples.iter().map(|s| s.tokens_used).sum::<usize>() as f64 / all_samples.len() as f64,
            avg_output_tokens: all_samples.iter().map(|s| s.tokens_used).sum::<usize>() as f64 / all_samples.len() as f64,
            verbosity_ratio: 0.0,  // TODO
            quality_per_token: 0.0,  // TODO
        }
    }
    
    fn calculate_percentile(&self, values: &[u64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort();
        let idx = (sorted.len() as f64 * percentile) as usize;
        sorted[idx.min(sorted.len() - 1)] as f64
    }
    
    async fn calculate_cost_analysis(
        &self,
        results: &[BenchmarkResult],
        provider: &dyn ModelProvider,
    ) -> Result<CostAnalysis> {
        let model_info = provider.get_model_info().await?;
        
        let total_input_tokens: usize = results
            .iter()
            .flat_map(|r| &r.sample_results)
            .map(|s| s.tokens_used / 2)  // Rough estimate
            .sum();
        
        let total_output_tokens: usize = results
            .iter()
            .flat_map(|r| &r.sample_results)
            .map(|s| s.tokens_used / 2)  // Rough estimate
            .sum();
        
        let input_cost = model_info
            .cost_per_1k_input_tokens
            .unwrap_or(0.0) * (total_input_tokens as f64 / 1000.0);
        
        let output_cost = model_info
            .cost_per_1k_output_tokens
            .unwrap_or(0.0) * (total_output_tokens as f64 / 1000.0);
        
        let estimated_cost_usd = input_cost + output_cost;
        
        Ok(CostAnalysis {
            total_input_tokens,
            total_output_tokens,
            estimated_cost_usd,
            cost_per_benchmark: HashMap::new(),  // TODO
            cost_efficiency_score: 0.0,  // TODO
        })
    }
    
    fn generate_recommendations(&self, category_scores: &HashMap<BenchmarkCategory, CategoryScore>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        for (category, score) in category_scores {
            if score.score < 0.6 {
                recommendations.push(format!(
                    "Model shows weakness in {:?}. Consider fine-tuning or using a different model for this use case.",
                    category
                ));
            }
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub provider_type: String,
    pub benchmark_categories: Option<Vec<BenchmarkCategory>>,
    pub benchmark_ids: Option<Vec<String>>,
    pub temperature: f64,
    pub max_samples_per_benchmark: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub id: Uuid,
    pub model_name: String,
    pub provider_type: String,
    pub timestamp: DateTime<Utc>,
    pub duration_seconds: u64,
    pub overall_score: f64,
    pub category_scores: HashMap<BenchmarkCategory, CategoryScore>,
    pub detailed_metrics: DetailedMetrics,
    pub cost_analysis: CostAnalysis,
    pub benchmark_results: Vec<BenchmarkResult>,
    pub config: EvaluationConfig,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    pub category: BenchmarkCategory,
    pub score: f64,
    pub benchmark_results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetailedMetrics {
    // Speed
    pub avg_tokens_per_second: f64,
    pub first_token_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    
    // Reasoning
    pub reasoning_accuracy: f64,
    pub logical_consistency: f64,
    pub chain_of_thought_quality: f64,
    
    // Quality
    pub exact_match_accuracy: f64,
    pub f1_score: f64,
    pub bleu_score: f64,
    pub rouge_score: RougeScores,
    pub semantic_similarity: f64,
    
    // Hallucination
    pub hallucination_rate: f64,
    pub factual_consistency: f64,
    pub groundedness_score: f64,
    
    // Stability
    pub output_drift: f64,
    pub temperature_variance: f64,
    pub seed_reproducibility: f64,
    
    // Context
    pub max_context_utilized: usize,
    pub context_recall_rate: f64,
    pub needle_in_haystack_score: f64,
    
    // Efficiency
    pub avg_input_tokens: f64,
    pub avg_output_tokens: f64,
    pub verbosity_ratio: f64,
    pub quality_per_token: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RougeScores {
    pub rouge_1: f64,
    pub rouge_2: f64,
    pub rouge_l: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub estimated_cost_usd: f64,
    pub cost_per_benchmark: HashMap<String, f64>,
    pub cost_efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub timestamp: DateTime<Utc>,
    pub models: Vec<String>,
    pub benchmark_ids: Vec<String>,
    pub results: HashMap<String, Vec<BenchmarkResult>>,
    pub winner_by_category: HashMap<BenchmarkCategory, String>,
}
```

### 3.5 Storage Layer

**File:** `models-core/src/storage/mod.rs`

```rust
#[async_trait]
pub trait EvaluationStorage: Send + Sync {
    async fn save_benchmark_result(&self, result: &BenchmarkResult) -> Result<()>;
    async fn save_evaluation_report(&self, report: &EvaluationReport) -> Result<()>;
    async fn get_evaluation_report(&self, id: Uuid) -> Result<Option<EvaluationReport>>;
    async fn list_evaluation_reports(&self, filter: ReportFilter) -> Result<Vec<EvaluationReport>>;
    async fn get_benchmark_history(&self, benchmark_id: &str, model_name: &str) -> Result<Vec<BenchmarkResult>>;
}

#[derive(Debug, Clone)]
pub struct ReportFilter {
    pub model_name: Option<String>,
    pub provider_type: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub min_overall_score: Option<f64>,
}
```

**PostgreSQL Implementation:**

**File:** `models-storage/src/postgres.rs`

```rust
pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        
        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await?;
        
        Ok(Self { pool })
    }
}

#[async_trait]
impl EvaluationStorage for PostgresStorage {
    async fn save_benchmark_result(&self, result: &BenchmarkResult) -> Result<()> {
        let result_json = serde_json::to_value(result)?;
        
        sqlx::query!(
            r#"
            INSERT INTO benchmark_results 
            (id, benchmark_id, model_name, provider, timestamp, overall_score, result_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                overall_score = EXCLUDED.overall_score,
                result_data = EXCLUDED.result_data
            "#,
            Uuid::new_v4(),
            result.benchmark_id,
            result.model_name,
            result.provider,
            result.timestamp,
            result.overall_score,
            result_json,
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    async fn save_evaluation_report(&self, report: &EvaluationReport) -> Result<()> {
        let report_json = serde_json::to_value(report)?;
        
        sqlx::query!(
            r#"
            INSERT INTO evaluation_reports
            (id, model_name, provider_type, timestamp, overall_score, report_data)
            VALUES ($1, $2, $3, $4, $5, $6)
            "#,
            report.id,
            report.model_name,
            report.provider_type,
            report.timestamp,
            report.overall_score,
            report_json,
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    async fn get_evaluation_report(&self, id: Uuid) -> Result<Option<EvaluationReport>> {
        let row = sqlx::query!(
            r#"
            SELECT report_data
            FROM evaluation_reports
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;
        
        if let Some(row) = row {
            let report: EvaluationReport = serde_json::from_value(row.report_data)?;
            Ok(Some(report))
        } else {
            Ok(None)
        }
    }
    
    async fn list_evaluation_reports(&self, filter: ReportFilter) -> Result<Vec<EvaluationReport>> {
        let mut query = String::from("SELECT report_data FROM evaluation_reports WHERE 1=1");
        
        if filter.model_name.is_some() {
            query.push_str(" AND model_name = $1");
        }
        
        // TODO: Build dynamic query based on filter
        
        let rows = sqlx::query_scalar::<_, serde_json::Value>(&query)
            .fetch_all(&self.pool)
            .await?;
        
        let reports: Vec<EvaluationReport> = rows
            .into_iter()
            .filter_map(|row| serde_json::from_value(row).ok())
            .collect();
        
        Ok(reports)
    }
    
    async fn get_benchmark_history(&self, benchmark_id: &str, model_name: &str) -> Result<Vec<BenchmarkResult>> {
        let rows = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            SELECT result_data
            FROM benchmark_results
            WHERE benchmark_id = $1 AND model_name = $2
            ORDER BY timestamp DESC
            "#
        )
        .bind(benchmark_id)
        .bind(model_name)
        .fetch_all(&self.pool)
        .await?;
        
        let results: Vec<BenchmarkResult> = rows
            .into_iter()
            .filter_map(|row| serde_json::from_value(row).ok())
            .collect();
        
        Ok(results)
    }
}
```

### 3.6 CLI Interface

**File:** `models-cli/src/main.rs`

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "models")]
#[command(about = "Models Lab - Universal LLM Evaluation Platform", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluate a model
    Evaluate {
        /// Provider type (ollama, vllm, openai, etc.)
        #[arg(short, long)]
        provider: String,
        
        /// Model name
        #[arg(short, long)]
        model: String,
        
        /// Benchmarks to run (comma-separated) or "all"
        #[arg(short, long, default_value = "all")]
        benchmarks: String,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
        
        /// Provider endpoint URL
        #[arg(long)]
        endpoint: Option<String>,
        
        /// API key (for cloud providers)
        #[arg(long, env = "LLM_API_KEY")]
        api_key: Option<String>,
    },
    
    /// Compare multiple models
    Compare {
        /// Models to compare (format: provider:model)
        #[arg(short, long, num_args = 1..)]
        models: Vec<String>,
        
        /// Benchmarks to run
        #[arg(short, long, default_value = "all")]
        benchmarks: String,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// List available benchmarks
    ListBenchmarks {
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,
    },
    
    /// View evaluation report
    ViewReport {
        /// Report ID
        report_id: String,
    },
    
    /// List past evaluation reports
    ListReports {
        /// Filter by model name
        #[arg(short, long)]
        model: Option<String>,
        
        /// Limit number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Setup logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    env::set_var("RUST_LOG", log_level);
    env_logger::init();
    
    match cli.command {
        Commands::Evaluate {
            provider,
            model,
            benchmarks,
            output,
            endpoint,
            api_key,
        } => {
            handle_evaluate(provider, model, benchmarks, output, endpoint, api_key).await?;
        }
        Commands::Compare {
            models,
            benchmarks,
            output,
        } => {
            handle_compare(models, benchmarks, output).await?;
        }
        Commands::ListBenchmarks { category } => {
            handle_list_benchmarks(category).await?;
        }
        Commands::ViewReport { report_id } => {
            handle_view_report(report_id).await?;
        }
        Commands::ListReports { model, limit } => {
            handle_list_reports(model, limit).await?;
        }
    }
    
    Ok(())
}

async fn handle_evaluate(
    provider: String,
    model: String,
    benchmarks: String,
    output: Option<String>,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<()> {
    println!("🚀 Starting evaluation for {} on {}", model, provider);
    println!();
    
    // Create provider
    let provider_config = ProviderConfig {
        provider_type: parse_provider_type(&provider, endpoint)?,
        model_name: model.clone(),
        endpoint: None,
        api_key,
        timeout_seconds: 120,
        max_retries: 3,
        retry_delay_ms: 1000,
    };
    
    let provider = ProviderFactory::create(provider_config)?;
    
    // Setup storage
    let database_url = env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/models_lab".to_string());
    let storage = PostgresStorage::new(&database_url).await?;
    
    // Create orchestrator
    let orchestrator = EvaluationOrchestrator::new(Box::new(storage));
    
    // Parse benchmarks
    let benchmark_categories = if benchmarks == "all" {
        None
    } else {
        Some(parse_benchmark_categories(&benchmarks)?)
    };
    
    let config = EvaluationConfig {
        provider_type: provider.to_string(),
        benchmark_categories,
        benchmark_ids: None,
        temperature: 0.0,
        max_samples_per_benchmark: None,
        seed: Some(42),
    };
    
    // Run evaluation with progress bar
    let report = orchestrator.run_full_evaluation(provider, config).await?;
    
    // Print results
    print_evaluation_report(&report);
    
    // Save to file if requested
    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&report)?;
        tokio::fs::write(output_path, json).await?;
        println!("\n✅ Report saved to output file");
    }
    
    Ok(())
}

fn print_evaluation_report(report: &EvaluationReport) {
    println!("\n📊 Evaluation Report");
    println!("═══════════════════════════════════════════════════════");
    println!("Model: {}", report.model_name);
    println!("Provider: {}", report.provider_type);
    println!("Timestamp: {}", report.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("Duration: {}s", report.duration_seconds);
    println!();
    
    println!("🎯 Overall Score: {:.1}/10", report.overall_score * 10.0);
    println!();
    
    println!("📈 Category Scores:");
    println!("───────────────────────────────────────────────────────");
    for (category, score) in &report.category_scores {
        let bar = create_progress_bar(score.score);
        println!("{:20} {} {:.1}/10", format!("{:?}:", category), bar, score.score * 10.0);
    }
    println!();
    
    println!("⚡ Performance Metrics:");
    println!("───────────────────────────────────────────────────────");
    let metrics = &report.detailed_metrics;
    println!("Tokens/sec:        {:.1}", metrics.avg_tokens_per_second);
    println!("P95 Latency:       {:.0}ms", metrics.p95_latency_ms);
    println!("Hallucination Rate: {:.1}%", metrics.hallucination_rate * 100.0);
    println!();
    
    println!("💰 Cost Analysis:");
    println!("───────────────────────────────────────────────────────");
    let cost = &report.cost_analysis;
    println!("Total Tokens:      {}", cost.total_input_tokens + cost.total_output_tokens);
    println!("Estimated Cost:    ${:.4}", cost.estimated_cost_usd);
    println!();
    
    if !report.recommendations.is_empty() {
        println!("💡 Recommendations:");
        println!("───────────────────────────────────────────────────────");
        for rec in &report.recommendations {
            println!("• {}", rec);
        }
        println!();
    }
}

fn create_progress_bar(score: f64) -> String {
    let filled = (score * 20.0) as usize;
    let empty = 20 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
```

### 3.7 REST API

**File:** `models-api/src/main.rs`

```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    orchestrator: Arc<EvaluationOrchestrator>,
    storage: Arc<dyn EvaluationStorage>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    env_logger::init();
    
    // Setup database
    let database_url = env::var("DATABASE_URL")?;
    let storage = Arc::new(PostgresStorage::new(&database_url).await?);
    
    // Setup orchestrator
    let orchestrator = Arc::new(EvaluationOrchestrator::new(
        Box::new(storage.clone())
    ));
    
    let state = AppState {
        orchestrator,
        storage,
    };
    
    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/evaluate", post(create_evaluation))
        .route("/api/v1/evaluations", get(list_evaluations))
        .route("/api/v1/evaluations/:id", get(get_evaluation))
        .route("/api/v1/compare", post(compare_models))
        .route("/api/v1/benchmarks", get(list_benchmarks))
        .route("/api/v1/benchmarks/:id", get(get_benchmark_info))
        .layer(CorsLayer::permissive())
        .with_state(state);
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("🚀 Models Lab API server listening on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

async fn health_check() -> Json<serde_json::Value> {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[derive(Debug, Deserialize)]
struct CreateEvaluationRequest {
    provider_config: ProviderConfig,
    evaluation_config: EvaluationConfig,
}

async fn create_evaluation(
    State(state): State<AppState>,
    Json(request): Json<CreateEvaluationRequest>,
) -> Result<Json<EvaluationReport>, (StatusCode, String)> {
    let provider = ProviderFactory::create(request.provider_config)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let report = state.orchestrator
        .run_full_evaluation(provider, request.evaluation_config)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(report))
}

async fn list_evaluations(
    State(state): State<AppState>,
) -> Result<Json<Vec<EvaluationReport>>, (StatusCode, String)> {
    let reports = state.storage
        .list_evaluation_reports(ReportFilter::default())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(reports))
}

async fn get_evaluation(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<EvaluationReport>, (StatusCode, String)> {
    let id = Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let report = state.storage
        .get_evaluation_report(id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Evaluation not found".to_string()))?;
    
    Ok(Json(report))
}

#[derive(Debug, Deserialize)]
struct CompareModelsRequest {
    provider_configs: Vec<ProviderConfig>,
    benchmark_ids: Vec<String>,
}

async fn compare_models(
    State(state): State<AppState>,
    Json(request): Json<CompareModelsRequest>,
) -> Result<Json<ComparisonReport>, (StatusCode, String)> {
    let mut providers = Vec::new();
    
    for config in request.provider_configs {
        let provider = ProviderFactory::create(config.clone())
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        providers.push((config.model_name.clone(), provider));
    }
    
    let report = state.orchestrator
        .compare_models(providers, request.benchmark_ids)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(report))
}

async fn list_benchmarks() -> Json<Vec<BenchmarkInfo>> {
    let registry = BenchmarkRegistry::new();
    let benchmarks = registry.list();
    
    let info: Vec<BenchmarkInfo> = benchmarks
        .into_iter()
        .map(|b| BenchmarkInfo {
            id: b.id().to_string(),
            name: b.name().to_string(),
            category: b.category(),
            description: b.description().to_string(),
        })
        .collect();
    
    Json(info)
}

#[derive(Debug, Serialize)]
struct BenchmarkInfo {
    id: String,
    name: String,
    category: BenchmarkCategory,
    description: String,
}
```

---

## 4. Database Schema

**File:** `migrations/001_initial_schema.sql`

```sql
-- Evaluation reports table
CREATE TABLE evaluation_reports (
    id UUID PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    provider_type VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    duration_seconds BIGINT NOT NULL,
    overall_score DOUBLE PRECISION NOT NULL,
    report_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_model_name (model_name),
    INDEX idx_provider_type (provider_type),
    INDEX idx_timestamp (timestamp),
    INDEX idx_overall_score (overall_score)
);

-- Benchmark results table
CREATE TABLE benchmark_results (
    id UUID PRIMARY KEY,
    benchmark_id VARCHAR(100) NOT NULL,
    benchmark_name VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    overall_score DOUBLE PRECISION NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_benchmark_id (benchmark_id),
    INDEX idx_model_name (model_name),
    INDEX idx_timestamp (timestamp)
);

-- Sample results table (for detailed analysis)
CREATE TABLE sample_results (
    id UUID PRIMARY KEY,
    benchmark_result_id UUID REFERENCES benchmark_results(id) ON DELETE CASCADE,
    sample_id VARCHAR(255) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    latency_ms BIGINT NOT NULL,
    tokens_used INTEGER NOT NULL,
    is_correct BOOLEAN NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_benchmark_result_id (benchmark_result_id)
);

-- Provider configurations table
CREATE TABLE provider_configs (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    provider_type VARCHAR(100) NOT NULL,
    config_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 5. Configuration Files

**File:** `config/default.toml`

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[database]
url = "postgres://models_lab:password@localhost:5432/models_lab"
max_connections = 10
min_connections = 2

[redis]
url = "redis://localhost:6379"
pool_size = 10

[logging]
level = "info"
format = "json"

[evaluation]
default_temperature = 0.0
default_max_tokens = 2048
default_seed = 42

[providers.ollama]
default_base_url = "http://localhost:11434"
timeout_seconds = 120

[providers.vllm]
default_base_url = "http://localhost:8000"
timeout_seconds = 120

[providers.openai]
timeout_seconds = 60
max_retries = 3

[benchmarks]
data_directory = "./datasets"
cache_directory = "./cache"
```

---

## 6. Docker Configuration

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: models_lab
      POSTGRES_PASSWORD: password
      POSTGRES_DB: models_lab
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U models_lab"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  models-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: api
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgres://models_lab:password@postgres:5432/models_lab
      REDIS_URL: redis://redis:6379
      RUST_LOG: info
    depends_on:
      - postgres
      - redis
      - ollama
    volumes:
      - ./datasets:/datasets
      - ./config:/config

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  ollama_data:
  prometheus_data:
  grafana_data:
```

**File:** `Dockerfile`

```dockerfile
# Build stage
FROM rust:1.75 AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY models-core/Cargo.toml ./models-core/
COPY models-providers/Cargo.toml ./models-providers/
COPY models-benchmark/Cargo.toml ./models-benchmark/
COPY models-metrics/Cargo.toml ./models-metrics/
COPY models-storage/Cargo.toml ./models-storage/
COPY models-api/Cargo.toml ./models-api/
COPY models-cli/Cargo.toml ./models-cli/

# Copy source code
COPY . .

# Build
RUN cargo build --release --bin models-api

# Runtime stage - API
FROM debian:bookworm-slim AS api

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/models-api /usr/local/bin/

EXPOSE 8080

CMD ["models-api"]

# Runtime stage - CLI
FROM debian:bookworm-slim AS cli

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/models /usr/local/bin/

ENTRYPOINT ["models"]
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal:** Basic evaluation capability

- [ ] Implement universal provider trait
- [ ] Create Ollama, vLLM, and OpenAI provider adapters
- [ ] Implement basic benchmark trait system
- [ ] Add MMLU, GSM8K, and HumanEval benchmarks
- [ ] Implement core speed and accuracy metrics
- [ ] Create CLI with evaluate command
- [ ] PostgreSQL storage layer
- [ ] Basic REST API endpoints

**Deliverables:**
- Can evaluate local and API models
- 3-5 core benchmarks working
- CLI and API functional
- Results stored in database

### Phase 2: Comprehensive Evaluation (Weeks 5-8)
**Goal:** Full benchmark suite

- [ ] Add hallucination benchmarks (TruthfulQA)
- [ ] Implement context window testing
- [ ] Add safety benchmarks
- [ ] Multi-turn conversation benchmarks (MT-Bench)
- [ ] Advanced metrics (drift, calibration, efficiency)
- [ ] Comparison report generation
- [ ] Export reports (JSON, HTML, PDF)

**Deliverables:**
- 10+ benchmarks implemented
- Comprehensive metric coverage
- Model comparison capability
- Professional report generation

### Phase 3: Advanced Features (Weeks 9-12)
**Goal:** Production-ready platform

- [ ] Web dashboard (Svelte/React)
- [ ] Real-time evaluation progress
- [ ] Historical tracking and trends
- [ ] LLM-as-judge evaluation
- [ ] RAG-specific benchmarks
- [ ] Distributed evaluation support
- [ ] CI/CD integration templates
- [ ] Auto-tuning/hyperparameter search

**Deliverables:**
- Web UI with interactive dashboards
- Advanced evaluation techniques
- Production monitoring
- CI/CD ready

### Phase 4: Research & Scale (Weeks 13+)
**Goal:** Cutting-edge capabilities

- [ ] Human-in-the-loop evaluation (ELO ratings)
- [ ] Multi-modal evaluation support
- [ ] Custom benchmark creation UI
- [ ] Sustainability metrics
- [ ] Advanced drift detection
- [ ] Distributed workers for scale
- [ ] Plugin system for custom metrics

---

## 8. Success Metrics

### Technical Metrics
- **Evaluation Speed:** Complete MMLU (5-shot) in < 30 minutes for 3B model
- **Accuracy:** Metrics match published benchmarks within 2%
- **Reliability:** 99.9% uptime for API service
- **Scalability:** Support 100+ concurrent evaluations

### Business Metrics
- **Adoption:** 1000+ evaluations run in first 3 months
- **Coverage:** Support 10+ providers, 15+ benchmarks
- **Community:** 50+ GitHub stars, 10+ contributors
- **Usage:** 100+ active monthly users

---

## 9. API Examples

### Evaluate a Model

```bash
curl -X POST http://localhost:8080/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "provider_config": {
      "provider_type": {
        "type": "ollama",
        "base_url": "http://localhost:11434"
      },
      "model_name": "llama3.2:3b",
      "timeout_seconds": 120,
      "max_retries": 3,
      "retry_delay_ms": 1000
    },
    "evaluation_config": {
      "provider_type": "ollama",
      "benchmark_categories": ["Reasoning", "Hallucination"],
      "temperature": 0.0,
      "seed": 42
    }
  }'
```

### Compare Models

```bash
curl -X POST http://localhost:8080/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "provider_configs": [
      {
        "provider_type": {"type": "ollama", "base_url": "http://localhost:11434"},
        "model_name": "llama3.2:3b",
        "timeout_seconds": 120,
        "max_retries": 3,
        "retry_delay_ms": 1000
      },
      {
        "provider_type": {"type": "openai"},
        "model_name": "gpt-4",
        "api_key": "sk-...",
        "timeout_seconds": 60,
        "max_retries": 3,
        "retry_delay_ms": 1000
      }
    ],
    "benchmark_ids": ["mmlu", "gsm8k", "truthfulqa"]
  }'
```

### CLI Usage

```bash
# Evaluate a model
models evaluate \
  --provider ollama \
  --model llama3.2:3b \
  --benchmarks reasoning,hallucination,speed \
  --output report.json

# Compare models
models compare \
  --models "ollama:llama3.2,openai:gpt-4,anthropic:claude-3.5-sonnet" \
  --benchmarks all \
  --output comparison.html

# List benchmarks
models list-benchmarks --category reasoning

# View past reports
models list-reports --model llama3.2 --limit 5
```

---

## 10. Extension Points

### Custom Benchmarks
```rust
// Users can implement custom benchmarks
pub struct MyCustomBenchmark;

#[async_trait]
impl Benchmark for MyCustomBenchmark {
    fn id(&self) -> &str { "my_custom_benchmark" }
    fn name(&self) -> &str { "My Custom Benchmark" }
    fn category(&self) -> BenchmarkCategory { BenchmarkCategory::Custom }
    
    async fn load_dataset(&self) -> Result<Dataset> {
        // Load custom dataset
    }
    
    async fn run(&self, provider: &dyn ModelProvider) -> Result<BenchmarkResult> {
        // Custom evaluation logic
    }
    
    fn evaluate_response(&self, response: &str, expected: &str, metadata: &HashMap<String, Value>) -> MetricScores {
        // Custom scoring logic
    }
}
```

### Custom Metrics
```rust
pub struct MyCustomMetric;

impl Metric for MyCustomMetric {
    fn name(&self) -> &str { "my_custom_metric" }
    
    fn calculate(&self, data: &MetricData) -> f64 {
        // Custom calculation logic
    }
    
    fn unit(&self) -> MetricUnit { MetricUnit::Score }
    fn direction(&self) -> MetricDirection { MetricDirection::HigherIsBetter }
}
```

---

## 11. Documentation Requirements

### User Documentation
- [ ] Getting Started Guide
- [ ] Provider Setup Guides (Ollama, vLLM, etc.)
- [ ] Benchmark Descriptions
- [ ] Metrics Explanations
- [ ] CLI Reference
- [ ] API Reference (OpenAPI spec)
- [ ] Troubleshooting Guide

### Developer Documentation
- [ ] Architecture Overview
- [ ] Contributing Guide
- [ ] Custom Benchmark Tutorial
- [ ] Custom Provider Tutorial
- [ ] Testing Guide
- [ ] Deployment Guide

---

## 12. Testing Requirements

### Unit Tests
- Provider adapters
- Benchmark implementations
- Metrics calculations
- Storage layer

### Integration Tests
- End-to-end evaluation flows
- API endpoints
- Database interactions
- Provider connections

### Benchmark Tests
- Validate benchmark scoring accuracy
- Compare against published results
- Regression testing

---

This PRD provides a complete specification for implementing Models Lab. The coding agent should:

1. Start with Phase 1 (Foundation)
2. Implement components incrementally
3. Add tests alongside features
4. Follow the architecture and patterns defined
5. Use the provided code examples as templates

**Priority Implementation Order:**
1. Universal provider trait + Ollama adapter
2. Benchmark trait + MMLU implementation
3. Basic metrics engine
4. CLI evaluate command
5. PostgreSQL storage
6. REST API
7. Additional providers and benchmarks
8. Advanced features