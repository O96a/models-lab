//! Models Lab CLI - Command line interface for LLM evaluation and management

use clap::{Parser, Subcommand};
use models_core::{
    generate_html_report, generate_comparison_html_report,
    save_html_report, save_comparison_html_report,
    generate_html_from_stored,
    HtmlReportConfig,
    BenchmarkRegistry, EvaluationConfig, EvaluationOrchestrator,
    InMemoryStorage, ModelProvider, StoredEvaluationReport,
    QueryParams, Error, EvaluationStorage,
    ComparisonConfig, ComparisonReport, ComparisonReportBuilder,
    ModelBenchmarkResult, ModelCategoryScore, ModelComparisonResult, ModelConfig,
    BenchmarkWinner, StatisticalTest, WinnerInfo,
};
use models_ollama::OllamaClient;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

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

        /// Output format (text, json, html)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// View a specific report in browser
    ViewReport {
        /// Report ID or file path
        report_id: String,

        /// Open in browser
        #[arg(short, long)]
        open: bool,

        /// Output to file instead of browser
        #[arg(short, long)]
        output: Option<String>,

        /// Output format (html, json)
        #[arg(long, default_value = "html")]
        format: String,
    },

    /// List all saved reports
    ListReports {
        /// Limit number of reports
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Filter by model
        #[arg(short, long)]
        model: Option<String>,

        /// Show detailed info
        #[arg(short, long)]
        detailed: bool,
    },

    /// Compare models
    Compare {
        /// Models to compare (comma-separated, format: provider:model)
        #[arg(short, long)]
        models: String,

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
        #[arg(long, default_value = "1024")]
        max_tokens: u32,

        /// Number of few-shot examples
        #[arg(long, default_value = "5")]
        few_shot: usize,

        /// Confidence level for statistical tests (e.g., 0.95)
        #[arg(long, default_value = "0.95")]
        confidence_level: f64,

        /// Ollama base URL
        #[arg(long, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
        ollama_url: String,

        /// Output file for report (JSON)
        #[arg(short, long)]
        output: Option<String>,

        /// OpenAI API key (if comparing OpenAI models)
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_api_key: Option<String>,
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
        Commands::Reports { limit, model, format } => handle_reports(limit, model, format).await,
        Commands::ViewReport { report_id, open, output, format } => {
            handle_view_report(report_id, open, output, format).await
        }
        Commands::ListReports { limit, model, detailed } => {
            handle_list_reports(limit, model, detailed).await
        }
        Commands::Compare {
            models,
            benchmarks,
            samples,
            temperature,
            max_tokens,
            few_shot,
            confidence_level,
            ollama_url,
            output,
            openai_api_key,
        } => {
            handle_compare(
                models,
                benchmarks,
                samples,
                temperature,
                max_tokens,
                few_shot,
                confidence_level,
                &ollama_url,
                output,
                openai_api_key,
            )
            .await
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

async fn handle_reports(limit: usize, model: Option<String>, format: String) {
    let storage = Arc::new(InMemoryStorage::new());

    // Build query params
    let mut query = QueryParams::new().with_limit(limit);
    if let Some(ref model_name) = model {
        query = query.with_model(model_name);
    }

    match storage.list_evaluation_reports(query).await {
        Ok(reports) => {
            display_reports_list(&reports, &format, false);
        }
        Err(e) => {
            eprintln!("❌ Error fetching reports: {}", e);
            println!("\n⚠️  No reports found.");
        }
    }
}

/// Display a list of evaluation reports in the specified format
fn display_reports_list(reports: &[StoredEvaluationReport], format: &str, detailed: bool) {
    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&reports).unwrap_or_default());
        }
        "html" => {
            for report in reports {
                if let Ok(html) = generate_html_from_stored(report) {
                    println!("{}", html);
                } else {
                    eprintln!("Failed to generate HTML for report {}", report.id);
                }
            }
        }
        _ => {
            // Text format (default)
            if reports.is_empty() {
                println!("📋 Evaluation Reports");
                println!("{}", "=".repeat(50));
                println!("\n⚠️  No reports found.");
                return;
            }

            println!("📋 Evaluation Reports ({} total)", reports.len());
            println!("{}", "=".repeat(50));

            for report in reports {
                println!();
                println!("Report ID: {}", report.id);
                println!("  Model: {} ({})", report.model_name, report.provider_name);
                println!("  Overall Score: {:.2}%", report.overall_score * 100.0);
                println!("  Benchmarks: {}", report.benchmark_count);
                println!("  Created: {}", report.created_at.format("%Y-%m-%d %H:%M:%S"));
                if detailed {
                    println!("  Category Scores: {}",
                        serde_json::to_string(&report.category_scores).unwrap_or_default());
                }
            }
        }
    }
}

// ============================================================================
// Compare Command
// ============================================================================

async fn handle_compare(
    models: String,
    benchmarks: String,
    samples: usize,
    temperature: f64,
    max_tokens: u32,
    few_shot: usize,
    confidence_level: f64,
    ollama_url: &str,
    output: Option<String>,
    _openai_api_key: Option<String>,
) {
    // Parse model configs
    let model_configs: Vec<ModelConfig> = models
        .split(',')
        .map(|s| {
            let parts: Vec<&str> = s.trim().split(':').collect();
            if parts.len() == 2 {
                ModelConfig {
                    provider: parts[0].to_string(),
                    model: parts[1].to_string(),
                    base_url: None,
                    api_key: None,
                }
            } else {
                // Default to ollama provider if not specified
                ModelConfig {
                    provider: "ollama".to_string(),
                    model: s.trim().to_string(),
                    base_url: None,
                    api_key: None,
                }
            }
        })
        .collect();

    if model_configs.len() < 2 {
        eprintln!("Error: At least 2 models are required for comparison");
        eprintln!("Usage: models compare --models provider:model1,provider:model2");
        std::process::exit(1);
    }

    let benchmark_list: Vec<String> = benchmarks.split(',').map(|s| s.trim().to_string()).collect();

    println!("🔄 Model Comparison");
    println!("{}", "=".repeat(80));
    println!("   Models: {}", model_configs.iter().map(|m| format!("{}:{}", m.provider, m.model)).collect::<Vec<_>>().join(", "));
    println!("   Benchmarks: {}", benchmarks);
    println!("   Samples per benchmark: {}", if samples == 0 { "all".to_string() } else { samples.to_string() });
    println!("   Temperature: {}", temperature);
    println!("   Max tokens: {}", max_tokens);
    println!("   Few-shot examples: {}", few_shot);
    println!("   Confidence level: {:.0}%", confidence_level * 100.0);
    println!();

    // Create comparison config
    let config = ComparisonConfig {
        models: model_configs.clone(),
        benchmarks: benchmark_list.clone(),
        max_samples: if samples > 0 { Some(samples) } else { None },
        temperature,
        max_tokens,
        num_few_shot: few_shot,
        confidence_level,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);
    let registry = models_benchmark::BenchmarkRegistry::with_builtin();

    // Run evaluation for each model
    for model_config in &model_configs {
        println!("📊 Evaluating {}:{}", model_config.provider, model_config.model);

        // Create provider
        let provider_result = match model_config.provider.as_str() {
            "ollama" => {
                models_providers::create_ollama_provider(
                    Some(ollama_url),
                    Some(&model_config.model),
                    None,
                )
            }
            _ => {
                eprintln!("Unknown provider: {}", model_config.provider);
                std::process::exit(1);
            }
        };

        let provider = match provider_result {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to create provider: {}", e);
                std::process::exit(1);
            }
        };

        // Run benchmarks
        let mut benchmark_results = Vec::new();
        let mut category_scores_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        let mut total_samples = 0usize;
        let mut total_latency = 0.0;
        let mut benchmarks_run = 0usize;

        for benchmark_id in &benchmark_list {
            print!("   Running {}... ", benchmark_id);

            if let Some(benchmark) = registry.get(benchmark_id) {
                let bench_config = models_benchmark::BenchmarkConfig {
                    max_samples: if samples > 0 { Some(samples) } else { None },
                    temperature,
                    max_tokens,
                    num_few_shot: few_shot,
                    ..Default::default()
                };

                match benchmark.run(&provider, bench_config).await {
                    Ok(result) => {
                        let category = format!("{:?}", benchmark.category()).to_lowercase();

                        benchmark_results.push(ModelBenchmarkResult {
                            benchmark_id: result.benchmark_id.clone(),
                            benchmark_name: result.benchmark_name.clone(),
                            category: category.clone(),
                            accuracy: result.statistics.accuracy,
                            sample_count: result.statistics.total_samples,
                            mean_latency_ms: result.statistics.mean_latency_ms,
                            std_dev: None,
                        });

                        // Track category scores
                        category_scores_map
                            .entry(category)
                            .or_default()
                            .push((result.benchmark_id.clone(), result.statistics.accuracy));

                        total_samples += result.statistics.total_samples;
                        total_latency += result.statistics.mean_latency_ms;
                        benchmarks_run += 1;

                        println!("{:.2}% ({}/{} samples, {:.0}ms avg)",
                            result.statistics.accuracy * 100.0,
                            result.statistics.correct_count,
                            result.statistics.total_samples,
                            result.statistics.mean_latency_ms
                        );
                    }
                    Err(e) => {
                        println!("❌ FAILED");
                        eprintln!("      Error: {}", e);
                    }
                }
            } else {
                println!("⚠️  NOT FOUND");
                eprintln!("      Benchmark '{}' not found in registry", benchmark_id);
            }
        }

        // Calculate category scores
        let category_scores: Vec<ModelCategoryScore> = category_scores_map
            .iter()
            .map(|(category, scores)| {
                let avg_score = scores.iter().map(|(_, s)| s).sum::<f64>() / scores.len() as f64;
                let benchmark_scores: HashMap<String, f64> = scores.iter().cloned().collect();

                ModelCategoryScore {
                    category: category.clone(),
                    score: avg_score,
                    benchmark_count: scores.len(),
                    benchmark_scores,
                }
            })
            .collect();

        // Calculate overall score
        let overall_score = if benchmarks_run > 0 {
            benchmark_results.iter().map(|r| r.accuracy).sum::<f64>() / benchmarks_run as f64
        } else {
            0.0
        };

        let avg_latency = if benchmarks_run > 0 {
            total_latency / benchmarks_run as f64
        } else {
            0.0
        };

        println!("   Overall Score: {:.2}%", overall_score * 100.0);
        println!();

        // Create model comparison result
        let model_result = ModelComparisonResult {
            model_name: model_config.model.clone(),
            provider_name: model_config.provider.clone(),
            overall_score,
            category_scores,
            benchmark_results,
            rank: 0,
            total_samples,
            avg_latency_ms: avg_latency,
        };

        report_builder = report_builder.add_result(model_result);
    }

    // Build final report
    let report = report_builder.build();

    // Display the comparison report
    println!("{}", report.generate_summary_table());

    // Save to file if requested
    if let Some(output_path) = output {
        match std::fs::write(
            &output_path,
            serde_json::to_string_pretty(&report).unwrap(),
        ) {
            Ok(()) => println!("💾 Comparison report saved to: {}", output_path),
            Err(e) => eprintln!("❌ Failed to save report: {}", e),
        }
    }
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

// ============================================================================
// View Report Command
// ============================================================================

async fn handle_view_report(
    report_id: String,
    should_open: bool,
    output: Option<String>,
    format: String,
) {
    println!("📄 Viewing Report: {}", report_id);

    // Check if report_id looks like a file path (contains / or .json at start)
    let is_file_path = report_id.contains('/') || report_id.starts_with("reports/");

    let report = if is_file_path {
        // Read from file
        let report_path = report_id.clone();
        let report_content = match std::fs::read_to_string(&report_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("❌ Failed to read report file '{}': {}", report_path, e);
                eprintln!("   Make sure the report exists or provide a full path.");
                std::process::exit(1);
            }
        };

        // Parse the report
        match serde_json::from_str::<StoredEvaluationReport>(&report_content) {
            Ok(r) => r,
            Err(_) => {
                // Try to parse as a raw EvaluationReport
                match serde_json::from_str::<models_core::EvaluationReport>(&report_content) {
                    Ok(er) => {
                        // Convert to StoredEvaluationReport format
                        StoredEvaluationReport {
                            id: er.id,
                            model_name: er.model_name.clone(),
                            provider_name: er.provider_name.clone(),
                            overall_score: er.overall_score,
                            benchmark_count: er.benchmark_results.len(),
                            category_scores: serde_json::to_value(&er.category_scores).unwrap_or_default(),
                            raw_report: serde_json::to_value(&er).unwrap_or_default(),
                            created_at: er.timestamp,
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to parse report JSON: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
    } else {
        // Try to fetch from storage by ID
        let storage = Arc::new(InMemoryStorage::new());
        let uuid = match report_id.parse::<uuid::Uuid>() {
            Ok(id) => id,
            Err(_) => {
                eprintln!("❌ Invalid report ID: '{}'", report_id);
                eprintln!("   Provide a valid UUID or a file path.");
                std::process::exit(1);
            }
        };

        match storage.get_evaluation_report(uuid).await {
            Ok(Some(report)) => report,
            Ok(None) => {
                eprintln!("❌ Report not found: '{}'", report_id);
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("❌ Error fetching report: {}", e);
                std::process::exit(1);
            }
        }
    };

    // Generate output based on format
    let output_content = match format.as_str() {
        "html" => {
            match generate_html_from_stored(&report) {
                Ok(html) => html,
                Err(e) => {
                    eprintln!("❌ Failed to generate HTML report: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "json" => {
            serde_json::to_string_pretty(&report).unwrap_or_else(|_| serde_json::to_string(&report).unwrap_or_default())
        }
        _ => {
            eprintln!("❌ Unknown format: {}. Use 'html' or 'json'.", format);
            std::process::exit(1);
        }
    };

    // Handle output
    if let Some(ref output_path) = output {
        match std::fs::write(output_path, &output_content) {
            Ok(()) => println!("💾 Report saved to: {}", output_path),
            Err(e) => {
                eprintln!("❌ Failed to save report: {}", e);
                std::process::exit(1);
            }
        }
        // Don't print to stdout when writing to file
        return;
    }

    // Open in browser if requested (only for HTML)
    if should_open && format == "html" {
        // Create a temporary file
        let temp = std::env::temp_dir().join(format!("report_{}.html", report.id));
        match std::fs::write(&temp, &output_content) {
            Ok(()) => {
                println!("Opening report in browser...");
                match open::that(temp) {
                    Ok(()) => {
                        println!("   Report opened successfully!");
                    }
                    Err(e) => {
                        eprintln!("⚠️  Failed to open browser: {}", e);
                        eprintln!("   Report saved to: {}", temp.display());
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to create temporary file: {}", e);
                eprintln!("   Report content:\n{}", output_content);
            }
        }
    } else {
        // Print to stdout
        println!("{}", output_content);
    }
}

// ============================================================================
// List Reports Command
// ============================================================================

async fn handle_list_reports(limit: usize, model: Option<String>, detailed: bool) {
    // First try to get from storage
    let storage = Arc::new(InMemoryStorage::new());

    // Build query params
    let mut query = QueryParams::new().with_limit(limit);
    if let Some(ref model_name) = model {
        query = query.with_model(model_name);
    }

    let result = storage.list_evaluation_reports(query).await;
    match result {
        Ok(reports) => {
            display_reports_list(&reports, "text", detailed);
        }
        Err(_e) => {
            eprintln!("❌ Error fetching reports from storage: {}", _e);
            // Fallback to file-based listing
            list_reports_from_files(limit, model, detailed);
        }
    }
}

/// List reports from file system (fallback)
fn list_reports_from_files(limit: usize, model: Option<String>, detailed: bool) {
    println!("📋 Saved Evaluation Reports");
    println!("{}", "=".repeat(80));

    // Look for reports in common locations
    let report_dirs = vec!["reports", "./reports", "~/.models-lab/reports"];
    let mut found_reports = Vec::new();

    for dir in &report_dirs {
        // Handle home directory expansion
        let dir_path = if dir.starts_with("~") {
            dir.replacen("~", &std::env::var("HOME").unwrap_or_else(|_| ".".to_string()), 1)
        } else {
            dir.to_string()
        };

        if let Ok(entries) = std::fs::read_dir(&dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "json" || ext == "html" {
                        if let Ok(metadata) = entry.metadata() {
                            if let Ok(modified) = metadata.modified() {
                                // Check model filter if specified
                                if let Some(ref model_name) = model {
                                    // Try to read and parse the file to check model
                                    if let Ok(content) = std::fs::read_to_string(&path) {
                                        if let Ok(report) = serde_json::from_str::<StoredEvaluationReport>(&content) {
                                            if report.model_name != *model_name {
                                                continue;
                                            }
                                        }
                                    }
                                }

                                let report_info = ReportInfo {
                                    path: path.to_string_lossy().to_string(),
                                    name: path.file_stem()
                                        .map(|s| s.to_string_lossy().to_string())
                                        .unwrap_or_else(|| "unknown".to_string()),
                                    modified,
                                    size: metadata.len(),
                                };
                                found_reports.push(report_info);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort by modification time (newest first)
    found_reports.sort_by(|a, b| b.modified.cmp(&a.modified));

    // Apply limit
    found_reports.truncate(limit);

    if found_reports.is_empty() {
        println!("\n⚠️  No reports found.");
        println!("   Run 'models evaluate --output reports/<name>.json' to generate reports.");
        return;
    }

    // Display reports
    if detailed {
        println!("\n{:<40} {:<20} {:<15} {}", "Name", "Modified", "Size", "Path");
        println!("{}", "-".repeat(80));
        for report in &found_reports {
            let size_str = format_file_size(report.size);
            let modified_str = chrono::DateTime::<chrono::Local>::from(report.modified)
                .format("%Y-%m-%d %H:%M");
            println!(
                "{:<40} {:<20} {:<15} {}",
                truncate(&report.name, 40),
                modified_str,
                size_str,
                report.path
            );
        }
    } else {
        println!();
        for (i, report) in found_reports.iter().enumerate() {
            let size_str = format_file_size(report.size);
            let modified_str = chrono::DateTime::<chrono::Local>::from(report.modified)
                .format("%Y-%m-%d %H:%M");
            println!(
                "  {}. {} ({}, {})",
                i + 1,
                report.name,
                modified_str,
                size_str
            );
        }
        println!("\n💡 Use 'models view-report <name> --open' to view a report in your browser");
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

struct ReportInfo {
    path: String,
    name: String,
    modified: std::time::SystemTime,
    size: u64,
}

fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = size as f64;
    let mut unit: &str = UNITS[0];

    for u in UNITS.iter() {
        if size < 1024.0 {
            unit = *u;
            break;
        }
        size /= 1024.0;
    }

    format!("{:.1} {}", size, unit)
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
