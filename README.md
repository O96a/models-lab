# Models Lab

A local-first LLM research platform with Ollama integration.

## Features

- **Ollama Integration**: Native support for local LLM inference
- **REST API**: Axum-based API for model management and inference
- **CLI Tool**: Command-line interface for easy management
- **Docker Ready**: Complete Docker Compose setup with observability

## Quick Start

```bash
# Start all services
docker-compose up -d

# Pull a model
docker exec -it ollama ollama pull llama3.2

# Test the API
curl http://localhost:8080/health
```

## CLI Usage

```bash
# Build CLI
cargo build -p models-cli

# List models
./target/debug/models ollama list

# Pull a model
./target/debug/models ollama pull llama3.2

# Run an experiment
./target/debug/models experiment --model llama3.2 --prompt "Hello, world!"
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/models` - List models
- `POST /api/v1/models/{name}/pull` - Pull a model
- `DELETE /api/v1/models/{name}` - Delete a model
- `POST /api/v1/inference` - Chat completion
- `POST /api/v1/embeddings` - Generate embeddings

## Project Structure

```
models-lab/
├── models-core/        # Core domain types
├── models-ollama/      # Ollama client
├── models-api/         # REST API
├── models-cli/         # CLI tool
├── models-workflow/    # Workflow engine
└── docker/             # Docker configuration
```

## License

MIT OR Apache-2.0