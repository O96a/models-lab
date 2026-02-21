//! Models Lab CLI - Command line interface for LLM evaluation and management

use clap::{Parser, Subcommand};
use models_core::{
    BenchmarkRegistry, EvaluationConfig, EvaluationOrchestrator,
    InMemoryStorage, ModelProvider,
};
use models_ollama::OllamaClient;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "models")]
#[command(about = "Models Lab CLI - LLM evaluation and management", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama base URL
    #[arg(short, long, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Ollama model management
    Ollama {
        #[command(subcommand)]
        command: OllamaCommands,
    },

    /// Run model evaluations
    Evaluate {
        /// Provider to use (ollama, openai)
        #[arg(short, long, default_value = "ollama")]
        provider: String,

        /// Model to evaluate
        #[arg(short, long)]
        model: Option<String>,

        /// Benchmarks to run (comma-separated)
        #[arg(short, long, default_value = "mmlu")]
        benchmarks: String,

        /// Number of samples per benchmark (0 = all)
        #[arg(short, long, default_value = "10")]
        samples: usize,

        /// Temperature for generation
        #[arg(short, long, default_value = "0.0")]
        temperature: f64,

        /// Maximum tokens to generate
        #[arg(short, long, default_value = "1024")]
        max_tokens: u32,

        /// Number of few-shot examples
        #[arg(short, long, default_value = "5")]
        few_shot: usize,

        /// Output file for report (JSON)
        #[arg(short, long)]
        output: Option<String>,

        /// OpenAI API key (if using openai provider)
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_api_key: Option<String>,
    },

    /// List available benchmarks
    ListBenchmarks {
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,
    },

    /// View evaluation reports
    Reports {
        /// Limit number of reports
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Filter by model
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Compare models
    Compare {
        /// Models to compare (comma-separated)
        #[arg(short, long)]
        models: String,

        /// Benchmarks to run (comma-separated)
        #[arg(short, long, default_value = "mmlu")]
        benchmarks: String,

        /// Number of samples per benchmark
        #[arg(short, long, default_value = "10")]
        samples: usize,
    },

    /// Run an experiment (quick test)
    Experiment {
        /// Model to use
        #[arg(short, long, default_value = "llama3.2")]
        model: String,
        /// Prompt to send
        #[arg(short, long)]
        prompt: String,
        /// Temperature
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Subcommand)]
enum OllamaCommands {
    /// List available models
    List,
    /// Pull a model
    Pull {
        /// Model name to pull
        name: String,
    },
    /// Delete a model
    Rm {
        /// Model name to delete
        name: String,
    },
    /// Test connection to Ollama
    Test,
    /// Get model info
    Info {
        /// Model name
        name: String,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set {
        key: String,
        value: String,
    },
}

// ============================================================================
// Benchmark Registry Wrapper
// ============================================================================

struct _CliBenchmarkRegistry {
    registry: models_benchmark::BenchmarkRegistry,
}

impl _CliBenchmarkRegistry {
    fn new() -> Self {
        Self {
            registry: models_benchmark::BenchmarkRegistry::with_builtin(),
        }
    }
}

impl BenchmarkRegistry for _CliBenchmarkRegistry {
    fn get(&self, _id: &str) -> Option<&dyn models_core::Benchmark> {
        // Placeholder - actual implementation uses benchmark crate directly
        None
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    match cli.command {
        Commands::Ollama { command } => {
            let client = match OllamaClient::new(&cli.ollama_url) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error creating Ollama client: {}", e);
                    std::process::exit(1);
                }
            };
            handle_ollama(client, command).await
        }
        Commands::Evaluate {
            provider,
            model,
            benchmarks,
            samples,
            temperature,
            max_tokens,
            few_shot,
            output,
            openai_api_key,
        } => {
            handle_evaluate(
                &cli.ollama_url,
                provider,
                model,
                benchmarks,
                samples,
                temperature,
                max_tokens,
                few_shot,
                output,
                openai_api_key,
            )
            .await
        }
        Commands::ListBenchmarks { category } => handle_list_benchmarks(category),
        Commands::Reports { limit, model } => handle_reports(limit, model).await,
        Commands::Compare { models, benchmarks, samples } => {
            handle_compare(models, benchmarks, samples).await
        }
        Commands::Experiment { model, prompt, temperature } => {
            let client = match OllamaClient::new(&cli.ollama_url) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error creating Ollama client: {}", e);
                    std::process::exit(1);
                }
            };
            handle_experiment(client, model, prompt, temperature).await
        }
        Commands::Config { command } => handle_config(command),
    }
}

// ============================================================================
// Evaluate Command
// ============================================================================

async fn handle_evaluate(
    ollama_url: &str,
    provider_name: String,
    model: Option<String>,
    benchmarks: String,
    samples: usize,
    temperature: f64,
    max_tokens: u32,
    few_shot: usize,
    output: Option<String>,
    openai_api_key: Option<String>,
) {
    println!("🚀 Starting evaluation...");
    println!("   Provider: {}", provider_name);
    println!("   Benchmarks: {}", benchmarks);

    // Create provider
    let provider: Box<dyn ModelProvider> = match provider_name.as_str() {
        "ollama" => {
            let ollama = models_providers::create_ollama_provider(
                Some(ollama_url),
                model.as_deref(),
                None,
            ).unwrap_or_else(|e| {
                eprintln!("Failed to create Ollama provider: {}", e);
                std::process::exit(1);
            });
            Box::new(ollama)
        }
        "openai" => {
            let api_key = openai_api_key.unwrap_or_else(|| {
                eprintln!("OpenAI API key required. Set OPENAI_API_KEY environment variable.");
                std::process::exit(1);
            });
            let openai = models_providers::create_openai_provider(
                &api_key,
                None,
                model.as_deref(),
                None,
                None,
            ).unwrap_or_else(|e| {
                eprintln!("Failed to create OpenAI provider: {}", e);
                std::process::exit(1);
            });
            Box::new(openai)
        }
        _ => {
            eprintln!("Unknown provider: {}", provider_name);
            std::process::exit(1);
        }
    };

    // Parse benchmarks
    let benchmark_list: Vec<String> = benchmarks.split(',').map(|s| s.trim().to_string()).collect();

    // Create evaluation config
    let config = EvaluationConfig {
        temperature,
        max_tokens,
        num_few_shot: few_shot,
        max_samples_per_benchmark: if samples > 0 { Some(samples) } else { None },
        benchmarks: benchmark_list,
        save_results: true,
        detailed_report: true,
    };

    // Create storage and orchestrator
    let _storage = Arc::new(InMemoryStorage::new());
    let _orchestrator = EvaluationOrchestrator::new(_storage);

    // For now, run each benchmark directly since we have a simpler setup
    let registry = models_benchmark::BenchmarkRegistry::with_builtin();

    let mut overall_results = Vec::new();
    let mut total_samples = 0usize;
    let mut total_correct = 0usize;

    for benchmark_id in &config.benchmarks {
        if let Some(benchmark) = registry.get(benchmark_id) {
            println!("\n📊 Running benchmark: {}", benchmark_id);

            let bench_config = models_benchmark::BenchmarkConfig {
                max_samples: config.max_samples_per_benchmark,
                temperature: config.temperature,
                max_tokens: config.max_tokens,
                num_few_shot: config.num_few_shot,
                ..Default::default()
            };

            match benchmark.run(provider.as_ref(), bench_config).await {
                Ok(result) => {
                    println!("   Accuracy: {:.2}%", result.statistics.accuracy * 100.0);
                    println!("   Samples: {}", result.statistics.total_samples);
                    println!("   Mean Latency: {:.2}ms", result.statistics.mean_latency_ms);

                    total_samples += result.statistics.total_samples;
                    total_correct += result.statistics.correct_count;
                    overall_results.push(result);
                }
                Err(e) => {
                    eprintln!("   ❌ Benchmark failed: {}", e);
                }
            }
        } else {
            eprintln!("⚠️  Benchmark not found: {}", benchmark_id);
        }
    }

    // Calculate overall score
    let overall_score = if total_samples > 0 {
        total_correct as f64 / total_samples as f64
    } else {
        0.0
    };

    println!("\n{}", "=".repeat(50));
    println!("📈 Evaluation Summary");
    println!("{}", "=".repeat(50));
    println!("   Overall Score: {:.2}%", overall_score * 100.0);
    println!("   Total Samples: {}", total_samples);
    println!("   Total Correct: {}", total_correct);

    // Save to file if requested
    if let Some(ref output_path) = output {
        let report = serde_json::json!({
            "overall_score": overall_score,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "results": overall_results,
            "config": config,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        match std::fs::write(output_path, serde_json::to_string_pretty(&report).unwrap()) {
            Ok(()) => println!("\n💾 Report saved to: {}", output_path),
            Err(e) => eprintln!("Failed to save report: {}", e),
        }
    }
}

// ============================================================================
// List Benchmarks Command
// ============================================================================

fn handle_list_benchmarks(category: Option<String>) {
    let registry = models_benchmark::BenchmarkRegistry::with_builtin();
    let benchmarks = registry.list();

    println!("📋 Available Benchmarks");
    println!("{}", "=".repeat(50));

    for id in benchmarks {
        if let Some(benchmark) = registry.get(id) {
            let bench_category = format!("{:?}", benchmark.category()).to_lowercase();

            // Filter by category if specified
            if let Some(ref cat) = category {
                if !bench_category.contains(&cat.to_lowercase()) {
                    continue;
                }
            }

            println!("\n  📦 {}", id);
            println!("     Name: {}", benchmark.name());
            println!("     Category: {}", bench_category);
            println!("     {}", benchmark.description());
        }
    }
}

// ============================================================================
// Reports Command
// ============================================================================

async fn handle_reports(limit: usize, model: Option<String>) {
    // Since we're using in-memory storage, this would be empty
    // In a real implementation with PostgreSQL, this would fetch from DB
    println!("📋 Evaluation Reports");
    println!("{}", "=".repeat(50));
    println!("\n⚠️  No reports found.");
    println!("   Run 'models evaluate' to generate reports.");
    println!("   (Note: Reports are stored in memory and not persisted between runs)");

    let _ = (limit, model); // Suppress unused variable warnings
}

// ============================================================================
// Compare Command
// ============================================================================

async fn handle_compare(models: String, benchmarks: String, samples: usize) {
    let model_list: Vec<String> = models.split(',').map(|s| s.trim().to_string()).collect();
    let benchmark_list: Vec<String> = benchmarks.split(',').map(|s| s.trim().to_string()).collect();

    println!("🔄 Model Comparison");
    println!("{}", "=".repeat(50));
    println!("   Models: {}", models);
    println!("   Benchmarks: {}", benchmarks);
    println!("   Samples per benchmark: {}", samples);

    println!("\n⚠️  Model comparison requires running evaluations for each model.");
    println!("   Run 'models evaluate --model <name>' for each model first.");

    let _ = (model_list, benchmark_list); // Suppress unused variable warnings
}

// ============================================================================
// Ollama Commands
// ============================================================================

async fn handle_ollama(client: OllamaClient, command: OllamaCommands) {
    match command {
        OllamaCommands::List => {
            match client.list_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No models found.");
                    } else {
                        println!("Available models:");
                        for model in models {
                            println!("  {} ({:.2} GB)", model.name, model.size as f64 / 1e9);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error listing models: {}", e);
                    std::process::exit(1);
                }
            }
        }
        OllamaCommands::Pull { name } => {
            println!("Pulling model: {}...", name);
            match client.pull_model(&name).await {
                Ok(()) => println!("Successfully pulled {}", name),
                Err(e) => {
                    eprintln!("Error pulling model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        OllamaCommands::Rm { name } => {
            match client.delete_model(&name).await {
                Ok(()) => println!("Deleted model: {}", name),
                Err(e) => {
                    eprintln!("Error deleting model: {}", e);
                    std::process::exit(1);
                }
            }
        }
        OllamaCommands::Test => {
            match client.health_check().await {
                Ok(true) => println!("Ollama is healthy at {}", client.base_url()),
                Ok(false) => {
                    println!("Ollama is not responding properly");
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Connection failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        OllamaCommands::Info { name } => {
            match client.get_model_info(&name).await {
                Ok(Some(model)) => {
                    println!("Model: {}", model.name);
                    println!("Size: {:.2} GB", model.size as f64 / 1e9);
                    println!("Modified: {}", model.modified_at);
                    if let Some(details) = model.details {
                        println!("Family: {}", details.family);
                        println!("Parameters: {}", details.parameter_size);
                        println!("Quantization: {}", details.quantization_level);
                    }
                }
                Ok(None) => {
                    println!("Model not found: {}", name);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

// ============================================================================
// Experiment Command
// ============================================================================

async fn handle_experiment(client: OllamaClient, model: String, prompt: String, temperature: f32) {
    use models_ollama::ChatRequest;

    println!("Running experiment with model: {}", model);
    println!("Prompt: {}", prompt);

    let request = ChatRequest::new(&model)
        .with_user_message(&prompt)
        .with_temperature(temperature);

    let start = std::time::Instant::now();

    match client.chat(request).await {
        Ok(response) => {
            println!("\n--- Response ---");
            println!("{}", response.message.content);
            println!("\n--- Stats ---");
            if let Some(eval_count) = response.eval_count {
                println!("Output tokens: {}", eval_count);
            }
            if let Some(prompt_count) = response.prompt_eval_count {
                println!("Prompt tokens: {}", prompt_count);
            }
            println!("Duration: {:?}", start.elapsed());
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

// ============================================================================
// Config Commands
// ============================================================================

fn handle_config(command: ConfigCommands) {
    match command {
        ConfigCommands::Show => {
            let config = models_core::Config::default();
            println!("{}", serde_json::to_string_pretty(&config).unwrap());
        }
        ConfigCommands::Set { key, value } => {
            println!("Setting {} = {}", key, value);
            println!("(Configuration persistence not implemented yet)");
        }
    }
}
