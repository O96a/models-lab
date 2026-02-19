//! LLM CLI - Command line interface for Models Lab

use clap::{Parser, Subcommand};
use models_core::Config;
use models_ollama::OllamaClient;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llm")]
#[command(about = "Models Lab CLI - Manage local LLM inference", long_about = None)]
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
    /// Run an experiment
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

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .init();

    // Create Ollama client
    let client = match OllamaClient::new(&cli.ollama_url) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error creating Ollama client: {}", e);
            std::process::exit(1);
        }
    };

    match cli.command {
        Commands::Ollama { command } => handle_ollama(client, command).await,
        Commands::Experiment { model, prompt, temperature } => {
            handle_experiment(client, model, prompt, temperature).await
        }
        Commands::Config { command } => handle_config(command),
    }
}

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

fn handle_config(command: ConfigCommands) {
    match command {
        ConfigCommands::Show => {
            let config = Config::default();
            println!("{}", serde_json::to_string_pretty(&config).unwrap());
        }
        ConfigCommands::Set { key, value } => {
            println!("Setting {} = {}", key, value);
            println!("(Configuration persistence not implemented yet)");
        }
    }
}