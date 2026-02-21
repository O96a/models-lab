# Build stage
FROM rust:1.82-bookworm AS builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY models-core/Cargo.toml ./models-core/
COPY models-ollama/Cargo.toml ./models-ollama/
COPY models-api/Cargo.toml ./models-api/
COPY models-cli/Cargo.toml ./models-cli/
COPY models-workflow/Cargo.toml ./models-workflow/
COPY models-providers/Cargo.toml ./models-providers/
COPY models-benchmark/Cargo.toml ./models-benchmark/
COPY models-metrics/Cargo.toml ./models-metrics/
COPY models-storage/Cargo.toml ./models-storage/

# Create dummy main.rs/lib.rs files to cache dependencies
RUN mkdir -p models-core/src && echo "fn main() {}" > models-core/src/lib.rs
RUN mkdir -p models-ollama/src && echo "fn main() {}" > models-ollama/src/lib.rs
RUN mkdir -p models-api/src && echo "fn main() {}" > models-api/src/main.rs
RUN mkdir -p models-cli/src && echo "fn main() {}" > models-cli/src/main.rs
RUN mkdir -p models-workflow/src && echo "fn main() {}" > models-workflow/src/lib.rs
RUN mkdir -p models-providers/src && echo "fn main() {}" > models-providers/src/lib.rs
RUN mkdir -p models-benchmark/src && echo "fn main() {}" > models-benchmark/src/lib.rs
RUN mkdir -p models-metrics/src && echo "fn main() {}" > models-metrics/src/lib.rs
RUN mkdir -p models-storage/src && echo "fn main() {}" > models-storage/src/lib.rs

# Build dependencies
RUN cargo build --release

# Copy actual source files
COPY models-core/src ./models-core/src
COPY models-ollama/src ./models-ollama/src
COPY models-api/src ./models-api/src
COPY models-cli/src ./models-cli/src
COPY models-workflow/src ./models-workflow/src
COPY models-providers/src ./models-providers/src
COPY models-benchmark/src ./models-benchmark/src
COPY models-metrics/src ./models-metrics/src
COPY models-storage/src ./models-storage/src

# Build the application
RUN cargo build --release -p models-api -p models-cli

# Runtime stage for API
FROM debian:bookworm-slim AS api

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

# Runtime stage for CLI
FROM debian:bookworm-slim AS cli

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary
COPY --from=builder /app/target/release/models /app/models

# Set environment variables
ENV RUST_LOG=info
ENV OLLAMA_URL=http://localhost:11434

ENTRYPOINT ["./models"]
