# Models Lab

A local-first LLM research platform with native Ollama integration for running and managing large language models locally.

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

## Features

### Core Capabilities
- **Native Ollama Integration** - Direct API client for local LLM inference
- **Multi-Model Support** - Works with Llama 3.2, Qwen, Gemma, Mistral, and all Ollama-compatible models
- **REST API** - Full-featured Axum-based HTTP API
- **CLI Tool** - Command-line interface for model management and experiments
- **Workflow Engine** - Task-based execution pipeline for experiments
- **Docker Ready** - Complete containerized deployment with observability stack

### Supported Operations
- Chat completions with streaming
- Text embeddings generation
- Model management (list, pull, delete)
- Health monitoring and status checks
- Batch inference workflows

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Rust 1.75+ for local development

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/O96a/models-lab.git
cd models-lab

# Start services (Ollama + optional observability stack)
docker-compose up -d

# Pull your first model
docker exec -it ollama ollama pull llama3.2

# Test the connection
curl http://localhost:11434/api/tags | jq
```

### Minimal Setup (Ollama only)

```bash
# Use minimal compose file
docker-compose -f docker-compose.minimal.yml up -d
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Models Lab                              │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  models-cli │  models-api │models-workflow│  models-ollama  │
│  (CLI Tool) │  (REST API) │  (Engine)    │  (Client)       │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                      models-core                             │
│              (Domain Types & Config)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Ollama      │
                    │  (localhost:11434)│
                    └─────────────────┘
```

---

## Project Structure

```
models-lab/
├── models-core/           # Core domain types and configuration
│   ├── src/
│   │   ├── config.rs      # Configuration management
│   │   ├── domain/        # Domain entities (Model, Experiment, Dataset)
│   │   ├── inference.rs   # Inference abstractions
│   │   └── error.rs       # Error types
│   └── Cargo.toml
│
├── models-ollama/         # Ollama API client
│   ├── src/
│   │   ├── client.rs      # HTTP client implementation
│   │   ├── models.rs      # Request/response types
│   │   ├── streaming.rs   # SSE streaming support
│   │   └── error.rs       # Ollama-specific errors
│   └── Cargo.toml
│
├── models-api/            # REST API server
│   ├── src/
│   │   ├── main.rs        # Server entry point
│   │   ├── handlers.rs    # Request handlers
│   │   └── routes.rs      # Route definitions
│   └── Cargo.toml
│
├── models-cli/            # Command-line interface
│   ├── src/
│   │   └── main.rs        # CLI implementation
│   └── Cargo.toml
│
├── models-workflow/       # Workflow execution engine
│   ├── src/
│   │   ├── engine.rs      # Workflow orchestration
│   │   └── tasks.rs       # Task definitions
│   └── Cargo.toml
│
├── docker/                # Docker configuration
│   └── prometheus.yml     # Prometheus config
│
├── config/                # Application config
│   └── default.toml       # Default settings
│
├── docker-compose.yml     # Full stack deployment
├── docker-compose.minimal.yml  # Ollama only
├── Dockerfile             # Multi-stage build
└── Cargo.toml             # Workspace definition
```

---

## CLI Reference

### Installation

```bash
# Build from source
cargo build --release -p models-cli

# Binary location
./target/release/models --help
```

### Commands

#### Model Management

```bash
# List all available models
models ollama list

# Get detailed info about a model
models ollama info llama3.2

# Pull a new model
models ollama pull llama3.2
models ollama pull mistral:7b
models ollama pull qwen2.5:0.5b

# Delete a model
models ollama rm llama3.2

# Test Ollama connection
models ollama test
```

#### Running Experiments

```bash
# Basic inference
models experiment --model llama3.2 --prompt "Explain quantum computing"

# With custom temperature
models experiment --model llama3.2 --prompt "Write a poem" --temperature 0.9

# With custom Ollama URL
models --ollama-url http://remote:11434 experiment --model llama3.2 --prompt "Hello"
```

#### Configuration

```bash
# View current config
models config show

# Set config value
models config set ollama.url http://localhost:11434
```

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ollama-url` | Ollama server URL | `http://localhost:11434` |
| `--verbose` | Enable verbose logging | `false` |

---

## REST API Reference

### Base URL
```
http://localhost:8080
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama": true,
  "version": "0.1.0"
}
```

---

#### List Models
```http
GET /api/v1/models
```

**Response:**
```json
[
  {
    "name": "llama3.2:latest",
    "modified_at": "2025-02-18T12:00:00Z",
    "size": 4661224676,
    "details": {
      "format": "gguf",
      "family": "llama",
      "parameter_size": "3.2B",
      "quantization_level": "Q4_K_M"
    }
  }
]
```

---

#### Get Model Info
```http
GET /api/v1/models/{name}
```

---

#### Pull Model
```http
POST /api/v1/models/{name}/pull
```

**Response:**
```json
{
  "message": "Model llama3.2 pulled successfully"
}
```

---

#### Delete Model
```http
DELETE /api/v1/models/{name}
```

---

#### Chat Completion
```http
POST /api/v1/inference
Content-Type: application/json

{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096
}
```

**Response:**
```json
{
  "model": "llama3.2",
  "content": "I'm doing well, thank you for asking!",
  "done": true
}
```

---

#### Generate Embeddings
```http
POST /api/v1/embeddings
Content-Type: application/json

{
  "model": "nomic-embed-text",
  "prompt": "Hello world"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `PORT` | API server port | `8080` |
| `RUST_LOG` | Log level | `info` |

### Configuration File (`config/default.toml`)

```toml
[server]
host = "0.0.0.0"
port = 8080
log_level = "info"

[ollama]
base_url = "http://localhost:11434"
timeout_seconds = 120
default_model = "llama3.2"
default_embedding_model = "nomic-embed-text"

[database]
url = "postgres://llm:llm_secret@localhost:5432/llm_lab"
max_connections = 10

[redis]
url = "redis://localhost:6379"
```

---

## Docker Deployment

### Full Stack (with Observability)

```bash
# Start all services
docker-compose up -d

# Services included:
# - models-api (API server)
# - ollama (LLM inference)
# - postgres (persistence)
# - redis (caching)
# - prometheus (metrics)
# - grafana (visualization)
```

### GPU Support

```bash
# Use GPU-enabled profile
docker-compose --profile gpu up -d
```

### Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:8080 |
| Ollama | http://localhost:11434 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## Development

### Building

```bash
# Build all crates
cargo build

# Build specific crate
cargo build -p models-ollama

# Build release
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p models-core
```

### Running the API Server

```bash
# Development mode
cargo run -p models-api

# With custom Ollama URL
OLLAMA_URL=http://remote:11434 cargo run -p models-api
```

---

## Supported Models

Models Lab works with any Ollama-compatible model:

| Model | Size | Use Case |
|-------|------|----------|
| llama3.2 | 3B | General purpose |
| llama3.2:1b | 1B | Lightweight tasks |
| mistral:7b | 7B | High quality responses |
| qwen2.5:0.5b | 0.5B | Ultra-fast inference |
| gemma3:270m | 270M | Minimal footprint |
| nomic-embed-text | 137M | Embeddings |

```bash
# Pull any model
docker exec -it ollama ollama pull <model-name>
```

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Axum](https://github.com/tokio-rs/axum) - Web framework
- [Tokio](https://tokio.rs/) - Async runtime