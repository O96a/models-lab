# Build stage
FROM rust:1.82-bookworm AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY llm-core/Cargo.toml ./llm-core/
COPY llm-ollama/Cargo.toml ./llm-ollama/
COPY models-api/Cargo.toml ./models-api/
COPY llm-cli/Cargo.toml ./llm-cli/
COPY llm-workflow/Cargo.toml ./llm-workflow/

# Create dummy main.rs files to cache dependencies
RUN mkdir -p llm-core/src && echo "fn main() {}" > llm-core/src/lib.rs
RUN mkdir -p llm-ollama/src && echo "fn main() {}" > llm-ollama/src/lib.rs
RUN mkdir -p models-api/src && echo "fn main() {}" > models-api/src/lib.rs
RUN mkdir -p llm-cli/src && echo "fn main() {}" > llm-cli/src/main.rs
RUN mkdir -p llm-workflow/src && echo "fn main() {}" > llm-workflow/src/lib.rs

# Build dependencies
RUN cargo build --release

# Copy actual source files
COPY llm-core/src ./llm-core/src
COPY llm-ollama/src ./llm-ollama/src
COPY models-api/src ./models-api/src
COPY llm-cli/src ./llm-cli/src
COPY llm-workflow/src ./llm-workflow/src

# Build the application
RUN cargo build --release -p models-api

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary
COPY --from=builder /app/target/release/models-api /app/models-api

# Expose port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info
ENV OLLAMA_URL=http://ollama:11434

# Run the binary
CMD ["./models-api"]