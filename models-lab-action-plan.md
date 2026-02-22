# Models Lab Enhancement — Action Plan

**Created:** 2026-02-21
**Last Updated:** 2026-02-22
**Purpose:** Detailed, phased checklist for a coding agent to implement the enhancements described in `model-lab-enhancment.md` on top of the existing codebase.

---

## Progress Summary

| Phase | Status | Progress |
|---|---|---|
| Prerequisites | ✅ Complete | 4/4 (cloud models in use) |
| Phase 1 | ✅ Complete | 12/12 sections done |
| Phase 2 | ⏳ In Progress | 4/6 (providers, benchmarks, metrics done) |
| Phase 3 | ⏳ Pending | 0/6 |
| Phase 4 | ⏳ Pending | 0/5 |

---

## 1. Gap Analysis — Existing Codebase vs Enhancement PRD

### 1.1 Codebase Inventory (Current State)

| Crate | Files | Summary |
|---|---|---|
| `models-core` | `config.rs`, `domain/{mod,model,experiment,dataset}.rs`, `error.rs`, `inference.rs`, `lib.rs` | Basic domain types (Model, Experiment, Dataset), Ollama-specific config, simple `InferenceProvider` enum, `InferenceRequest`/`InferenceResponse` structs, error types |
| `models-ollama` | `client.rs`, `models.rs`, `streaming.rs`, `error.rs`, `lib.rs` | Concrete Ollama HTTP client with chat, embed, list, pull, delete, health-check |
| `models-api` | `main.rs`, `lib.rs`, `handlers.rs`, `routes.rs` | Axum REST API: health, list/get/pull/delete models, inference chat, embeddings — all Ollama-only |
| `models-cli` | `main.rs` | Clap CLI: `ollama list/pull/rm/test/info`, `experiment`, `config show/set` |
| `models-workflow` | `engine.rs`, `tasks.rs`, `lib.rs` | Simple task runner (WorkflowEngine) with `InferenceTask` executing Ollama chat |

**Workspace deps (Cargo.toml):** tokio, axum, reqwest (+eventsource), serde, uuid, chrono, tracing, metrics, thiserror, anyhow, clap, async-trait.

**Infra files:** `Dockerfile` (single-target, Ollama only), `docker-compose.yml` (API + Ollama + Postgres + Prometheus + Grafana + Redis), `docker-compose.minimal.yml` (Ollama only), `config/default.toml`.

### 1.2 What the Enhancement PRD Proposes (Target State)

1. **Universal Provider Layer** — `ModelProvider` async trait with `generate()`, `embed()`, `get_model_info()`, `health_check()`, `generate_stream()`. Adapters for Ollama, vLLM, LM Studio, OpenAI, Anthropic, HuggingFace, Azure OpenAI, Cohere, Custom. `ProviderFactory` + `ProviderConfig`.
2. **Benchmark Engine** — `Benchmark` async trait with `load_dataset()`, `run()`, `evaluate_response()`. Built-in benchmarks: MMLU, BBH, ARC, GSM8K, MATH, HumanEval, MBPP, TruthfulQA, LongContext, NeedleInHaystack, AlpacaEval, MT-Bench, Safety. `BenchmarkRegistry`.
3. **Metrics Engine** — `Metric` trait. Speed: tokens/s, TTFT, P95 latency. Accuracy: exact-match, F1, BLEU, ROUGE. Hallucination: hallucination rate, factual consistency. Efficiency: cost/token, quality/token. Consistency: output drift, temperature variance. `MetricsEngine`.
4. **Evaluation Orchestrator** — Runs full eval suites, calculates aggregate/category scores, generates reports and recommendations, compares models.
5. **Storage Layer** — `EvaluationStorage` async trait, PostgreSQL implementation with `benchmark_results`, `evaluation_reports`, `sample_results`, `provider_configs` tables (+ migrations).
6. **Expanded CLI** — `evaluate`, `compare`, `list-benchmarks`, `view-report`, `list-reports` commands.
7. **Expanded REST API** — `/api/v1/evaluate`, `/api/v1/evaluations`, `/api/v1/evaluations/:id`, `/api/v1/compare`, `/api/v1/benchmarks`, `/api/v1/benchmarks/:id`.
8. **Web Dashboard** — Svelte/React + TailwindCSS (Phase 3).
9. **Advanced Features** — LLM-as-judge, RAG benchmarks, distributed evaluation, CI/CD templates, human-in-the-loop ELO, multi-modal, plugin system (Phase 3–4).

### 1.3 Critical Delta Summary

| Area | Existing | Enhancement | Action Required |
|---|---|---|---|
| **Provider trait** | `InferenceProvider` enum (5 variants, no trait) | `ModelProvider` async trait (9+ providers) | **Create new trait** in `models-core`; refactor `InferenceRequest`/`InferenceResponse` into `GenerateRequest`/`GenerateResponse` |
| **Ollama adapter** | Standalone `OllamaClient` (chat-based) | `OllamaProvider` implements `ModelProvider` (generate-based) | **Wrap** existing `OllamaClient` behind new `ModelProvider` trait |
| **Additional providers** | None | vLLM, OpenAI, Anthropic, etc. | **New crate** `models-providers` or add to `models-core` |
| **Benchmark engine** | None (only `Experiment` domain type) | Full benchmark framework + 13 benchmarks | **New crate** `models-benchmark` |
| **Metrics engine** | None | 15+ metrics | **New crate** `models-metrics` |
| **Evaluation orchestrator** | None | Full orchestrator + reports | **New module** in `models-core` |
| **Storage layer** | None (sqlite URL in config but unused) | PostgreSQL with migrations | **New crate** `models-storage` or add to `models-core` |
| **CLI** | Ollama management + basic experiment | Evaluate, compare, list-benchmarks, view-report | **Major rewrite** of `models-cli` |
| **REST API** | Ollama CRUD + inference | Evaluation endpoints | **Major extension** of `models-api` |
| **Docker** | Single API target, old crate names (`llm-*`) in Dockerfile | Multi-target (API + CLI), correct crate names | **Update** Dockerfile |
| **Config** | Ollama-only | Multi-provider, evaluation defaults, Redis, benchmarks | **Extend** `Config` struct and `default.toml` |
| **Datasets** | None | `datasets/` directory with MMLU, GSM8K, etc. | **Create** directory structure + dataset files |
| **Database migrations** | None | SQL migration files | **Create** `migrations/` directory |

### 1.4 Enhancement PRD — Modification Notes

> [!IMPORTANT]
> The PRD code examples are **templates**, not drop-in code. The following issues must be addressed during implementation.

1. **Crate naming mismatch.** The PRD references new crates (`models-providers`, `models-benchmark`, `models-metrics`, `models-storage`) that do not exist. The existing Dockerfile still references old names (`llm-core`, `llm-ollama`, etc.). Both need reconciliation.
2. **`InferenceProvider` vs `ModelProvider`.** The existing `InferenceProvider` enum in `models-core/src/inference.rs` conflicts with the PRD's `ModelProvider` trait. The enum should be renamed or removed, and the trait should take over the name.
3. **`Dataset` name collision.** The existing `models-core/src/domain/dataset.rs` defines a basic `Dataset` struct (id, name, num_samples). The PRD defines a different `Dataset` (name, samples: `Vec<DataSample>`, metadata). Must either rename the existing one (e.g., `DatasetEntity`) or replace it.
4. **`ModelProvider` name collision.** The existing `models-core/src/domain/model.rs` already exports an enum named `ModelProvider`. The PRD's `ModelProvider` is a trait. Rename the enum to `ProviderKind` or similar.
5. **Ollama API format.** The PRD's `OllamaProvider` uses `/api/generate` (generate endpoint), while the existing `OllamaClient` uses `/api/chat` (chat endpoint). Both are valid Ollama APIs. Use `/api/chat` (the chat-completion format) for consistency with multi-turn benchmarks, or implement both.
6. **`axum::Server::bind` is deprecated.** The PRD code uses `axum::Server::bind` which was removed in Axum 0.7. The existing codebase already uses the correct `axum::serve(listener, app)` pattern. Keep the existing pattern.
7. **PostgreSQL INDEX syntax.** The PRD's migration SQL uses MySQL-style `INDEX` declarations inside `CREATE TABLE`. PostgreSQL requires separate `CREATE INDEX` statements. Fix during migration creation.
8. **`hashmap!` macro.** The PRD uses a `hashmap!` macro that doesn't exist in standard Rust. Use `maplit::hashmap!` crate or `HashMap::from([(k, v), ...])`.
9. **`Experiment` / `WorkflowEngine` fate.** The existing `Experiment` domain type and `WorkflowEngine` are not referenced in the PRD. They can be preserved for backward compatibility or replaced by the evaluation orchestrator. Recommend: keep them but mark as deprecated.
10. **Database**: Config currently points to SQLite (`sqlite:data/llm-lab.db`) but docker-compose wires up PostgreSQL. The PRD assumes PostgreSQL. Standardize on PostgreSQL; add `sqlx` dependency.

---

## 2. Phased Action Plan

### Prerequisites

- [x] Ensure Rust 1.75+ is installed ✅ **Done** (Rust 1.93.1 installed on 2026-02-21)
- [x] Ensure Docker + Docker Compose are available ✅ **Done** (Docker 29.1.5, Docker Compose 5.0.1)
- [x] Ensure PostgreSQL 16, Redis 7 accessible (via docker-compose or local) ✅ **Done** (PostgreSQL and Redis containers running)
- [x] Ensure Ollama models available for testing ✅ **Done** (2026-02-22) — Using cloud models: `glm-5:cloud`, `qwen3.5:cloud`, `minimax-m2.5:cloud`, `glm-4.7:cloud`, `kimi-k2.5:cloud`, `gpt-oss:20b-cloud`, `gpt-oss:120b-cloud`

---

### Phase 1 — Foundation (Provider Layer + Storage + Basic Benchmark)

**Goal:** Universal provider trait, Ollama + OpenAI adapters working, PostgreSQL storage with migrations, one benchmark (MMLU) end-to-end, CLI evaluate command.

**Status:** ✅ **Phase 1 Complete** (2026-02-21) — 11/12 sections done, PostgreSQL storage deferred to Phase 2.

---

#### 1.1 Workspace Restructure

- [x] Add new workspace members to root `Cargo.toml`: ✅ **Done** (2026-02-21)
  - `models-providers` — provider adapters
  - `models-benchmark` — benchmark framework
  - `models-metrics` — metrics calculations
  - `models-storage` — persistence layer
- [x] Create directory scaffolding for each new crate (`cargo init --lib`) ✅ **Done** (2026-02-21)
- [x] Add shared workspace dependencies: `sqlx` (postgres, runtime-tokio), `maplit`, `pin-project` ✅ **Done** (2026-02-21)
- [x] Fix Dockerfile crate paths from `llm-*` to `models-*` ✅ **Done** (2026-02-21)

#### 1.2 Universal Provider Trait (`models-core`)

- [x] Create `models-core/src/providers/mod.rs` with: ✅ **Done** (2026-02-21)
  - `ModelProvider` async trait (`generate`, `embed`, `get_model_info`, `health_check`, `generate_stream`)
  - `GenerateRequest` struct (prompt, system_prompt, temperature, max_tokens, stop_sequences, top_p, top_k, presence_penalty, frequency_penalty, seed, stream, extra_params)
  - `GenerateResponse` struct (text, tokens_used, latency_ms, time_to_first_token_ms, finish_reason, model_name, metadata)
  - `TokenUsage`, `FinishReason`, `ModelInfo`, `HealthStatus` structs/enums
- [x] Create `models-core/src/providers/config.rs` with: ✅ **Done** (2026-02-21)
  - `ProviderConfig` struct
  - `ProviderType` tagged enum (Ollama, VLLM, OpenAI, Anthropic, etc.)
  - `ProviderFactory` struct with `create()` method
  - `ApiFormat` enum
- [x] Rename existing `ModelProvider` enum in `domain/model.rs` to `ProviderKind` ✅ **Done** (2026-02-21)
  - Update all references in `domain/mod.rs`, `model.rs`, and downstream
- [x] Rename existing `Dataset` in `domain/dataset.rs` to `DatasetEntity` ✅ **Done** (2026-02-21)
  - Update references in `domain/mod.rs`
- [x] Add `pub mod providers;` to `models-core/src/lib.rs` ✅ **Done** (2026-02-21)
- [x] Re-export key types from `lib.rs` ✅ **Done** (2026-02-21)

#### 1.3 Ollama Provider Adapter (`models-providers`)

- [x] Create `models-providers/src/ollama.rs`: ✅ **Done** (2026-02-21)
  - `OllamaProvider` struct wrapping existing `OllamaClient`
  - Implement `ModelProvider` trait (delegate to `OllamaClient::chat` / `embed` / etc.)
  - Map `ChatResponse` → `GenerateResponse`
  - Map `ChatRequest` ← `GenerateRequest`
- [x] Create `models-providers/src/lib.rs` re-exporting `OllamaProvider` ✅ **Done** (2026-02-21)
- [x] Register in `ProviderFactory::create()` ✅ **Done** (2026-02-21)

#### 1.4 OpenAI Provider Adapter (`models-providers`)

- [x] Create `models-providers/src/openai.rs`: ✅ **Done** (2026-02-21)
  - `OpenAIProvider` struct
  - Implement `ModelProvider` trait
  - Use `reqwest` to call `https://api.openai.com/v1/chat/completions`
  - Handle API key, organization header
- [x] Register in `ProviderFactory::create()` ✅ **Done** (2026-02-21)

#### 1.5 Database Migrations + Storage Layer (`models-storage`)

- [x] Create `migrations/001_initial_schema.sql`: ✅ **Done** (2026-02-22)
  - `evaluation_reports` table
  - `benchmark_results` table
  - `sample_results` table
  - `provider_configs` table
  - `model_comparisons` table
  - `benchmark_metadata` table
  - `audit_log` table
  - Uses proper PostgreSQL `CREATE INDEX` statements
- [x] Create `models-storage/src/lib.rs` with re-exports ✅ **Done** (2026-02-22)
- [x] Create `models-storage/src/postgres.rs` with `PostgresStorage` implementation ✅ **Done** (2026-02-22)
- [x] Create `EvaluationStorage` async trait in `models-core/src/storage/mod.rs`: ✅ **Done** (2026-02-21)
  - `save_benchmark_result`, `save_evaluation_report`, `get_evaluation_report`, `list_evaluation_reports`, `get_benchmark_history`
  - `InMemoryStorage` implementation for testing
- [x] Update `config.rs` to use `DatabaseConfig` with PostgreSQL URL ✅ **Done** (2026-02-21)

#### 1.6 Benchmark Framework (`models-benchmark`)

- [x] Create `models-benchmark/src/lib.rs` with: ✅ **Done** (2026-02-21)
  - `Benchmark` async trait (`id`, `name`, `category`, `description`, `load_dataset`, `run`, `evaluate_response`)
  - `BenchmarkCategory` enum
  - `Dataset`, `DataSample`, `DatasetMetadata`, `Difficulty` structs/enums
  - `BenchmarkResult`, `SampleResult`, `BenchmarkStatistics`, `BenchmarkConfig`, `MetricScores` structs
- [x] Create `models-benchmark/src/registry.rs`: ✅ **Done** (2026-02-21)
  - `BenchmarkRegistry` struct with `register`, `get`, `list`, `list_by_category`
- [x] Implement MMLU benchmark in `models-benchmark/src/reasoning/mmlu.rs`: ✅ **Done** (2026-02-21)
  - Question formatting, answer parsing
  - 5-shot evaluation
  - Embedded sample dataset (10 questions across multiple subjects)
- [x] Register MMLU in the `BenchmarkRegistry` ✅ **Done** (2026-02-21)

#### 1.7 Core Metrics (`models-metrics`)

- [x] Create `models-metrics/src/lib.rs` with: ✅ **Done** (2026-02-21)
  - `Metric` trait (`name`, `calculate`, `unit`, `direction`)
  - `MetricUnit`, `MetricDirection` enums
  - `MetricData` struct
  - `MetricsEngine` struct
  - `AggregateStats` struct with mean, median, std_dev, percentiles
- [x] Implement speed metrics in `models-metrics/src/speed.rs`: ✅ **Done** (2026-02-21)
  - `TokensPerSecondMetric`
  - `FirstTokenLatencyMetric`
  - `P95LatencyMetric`
- [x] Implement accuracy metrics in `models-metrics/src/accuracy.rs`: ✅ **Done** (2026-02-21)
  - `ExactMatchMetric`
  - `F1ScoreMetric`

#### 1.8 Evaluation Orchestrator (`models-core`)

- [x] Create `models-core/src/evaluation/orchestrator.rs`: ✅ **Done** (2026-02-21)
  - `EvaluationOrchestrator` struct
  - `run_full_evaluation()` — run benchmarks, calculate metrics, generate report
  - `EvaluationConfig`, `EvaluationReport`, `CategoryScore`, `DetailedMetrics`, `CostAnalysis`, `ComparisonReport` structs
  - `BenchmarkRunConfig`, `BenchmarkRunResult`, `BenchmarkRunStatistics` types
- [x] Create `models-core/src/evaluation/mod.rs` re-exporting types ✅ **Done** (2026-02-21)
- [x] Define `Benchmark` trait in models-core to avoid circular dependencies ✅ **Done** (2026-02-21)

#### 1.9 CLI Rewrite (`models-cli`)

- [x] Add `evaluate` subcommand: ✅ **Done** (2026-02-21)
  - `--provider`, `--model`, `--benchmarks`, `--samples`, `--output`, `--temperature`, `--max-tokens`, `--few-shot`
  - Instantiate provider via `create_ollama_provider` / `create_openai_provider`
  - Run benchmarks and display results
  - JSON output support via `--output` flag
- [x] Add `list-benchmarks` subcommand (with `--category` filter) ✅ **Done** (2026-02-21)
- [x] Add `reports` subcommand ✅ **Done** (2026-02-21)
- [x] Add `compare` subcommand ✅ **Done** (2026-02-21)
- [x] Preserve existing `ollama` subcommand for backward compat ✅ **Done** (2026-02-21)
- [x] Rename binary from `llm` to `models` ✅ **Done** (2026-02-21)

#### 1.10 API Extension (`models-api`)

- [x] Add new routes: ✅ **Done** (2026-02-21)
  - `POST /api/v1/evaluate` — create evaluation
  - `GET /api/v1/benchmarks` — list benchmarks
  - `GET /api/v1/reports` — list evaluation reports
  - `GET /api/v1/results` — list benchmark results
- [x] Update `AppState` to include `BenchmarkRegistry` and `InMemoryStorage` ✅ **Done** (2026-02-21)
- [x] Keep existing Ollama-specific endpoints for backward compatibility ✅ **Done** (2026-02-21)
- [x] Add `EvaluationHandler` with `evaluate`, `list_benchmarks`, `list_reports`, `list_results` methods ✅ **Done** (2026-02-21)

#### 1.11 Config & Infrastructure Updates

- [x] Extend `config/default.toml` with: ✅ **Done** (2026-02-21)
  - `[evaluation]` section (default_temperature, default_max_tokens, default_seed)
  - `[providers.ollama]`, `[providers.openai]` sections
  - `[benchmarks]` section (data_directory, cache_directory)
- [x] Extend `Config` struct in `models-core/src/config.rs` to parse new sections ✅ **Done** (existing struct)
- [x] Update Dockerfile for multi-target build (API + CLI stages) ✅ **Done** (2026-02-21)
- [x] Add `.gitignore` with `target/` and `Cargo.lock` ✅ **Done** (2026-02-21)

#### 1.12 Phase 1 Testing

- [x] Unit tests for `GenerateRequest`/`GenerateResponse` serialization ✅ **Done** (2026-02-21)
- [x] Unit tests for `OllamaProvider` creation ✅ **Done** (2026-02-21)
- [x] Unit tests for `ExactMatchMetric`, `F1ScoreMetric`, `TokensPerSecondMetric` ✅ **Done** (2026-02-21)
- [x] Unit tests for `BenchmarkRegistry` (register, get, list) ✅ **Done** (2026-02-21)
- [x] Unit tests for MMLU benchmark (answer extraction, evaluation) ✅ **Done** (2026-02-21)
- [x] Verify `cargo build --release` succeeds for all crates ✅ **Done** (2026-02-21)
- [x] Verify `cargo test` passes ✅ **Done** (2026-02-21)

---

### Phase 2 — Comprehensive Evaluation (More Benchmarks + Metrics + Reports)

**Goal:** 10+ benchmarks, full metric coverage, model comparison, report generation.

---

#### 2.1 Additional Providers

- [x] Implement `VLLMProvider` in `models-providers/src/vllm.rs` (OpenAI-compatible API) ✅ **Done** (2026-02-22)
- [x] Implement `AnthropicProvider` in `models-providers/src/anthropic.rs` ✅ **Done** (2026-02-22)
- [x] Implement `LMStudioProvider` (OpenAI-compatible, shares logic with vLLM) ✅ **Done** (2026-02-22)
- [x] Register all in `ProviderFactory` ✅ **Done** (2026-02-22)

#### 2.2 Additional Benchmarks

- [ ] **Reasoning:** BBH benchmark (`models-benchmark/src/reasoning/bbh.rs`)
- [ ] **Reasoning:** ARC benchmark (`models-benchmark/src/reasoning/arc.rs`)
- [x] **Math:** GSM8K benchmark (`models-benchmark/src/math/gsm8k.rs`) ✅ **Done** (2026-02-22)
- [ ] **Math:** MATH benchmark (`models-benchmark/src/math/math.rs`)
- [x] **Coding:** HumanEval benchmark (`models-benchmark/src/coding/humaneval.rs`) ✅ **Done** (2026-02-22)
- [ ] **Coding:** MBPP benchmark (`models-benchmark/src/coding/mbpp.rs`)
- [x] **Hallucination:** TruthfulQA benchmark (`models-benchmark/src/hallucination/truthfulqa.rs`) ✅ **Done** (2026-02-22)
- [ ] **Context:** LongContext benchmark (`models-benchmark/src/context/long_context.rs`)
- [x] **Context:** NeedleInHaystack benchmark (`models-benchmark/src/context/needle.rs`) ✅ **Done** (2026-02-22)
- [x] **Instruction:** AlpacaEval benchmark (`models-benchmark/src/instruction/alpaca_eval.rs`) ✅ **Done** (2026-02-22)
- [x] **Multi-turn:** MT-Bench benchmark (`models-benchmark/src/multiturn/mt_bench.rs`) ✅ **Done** (2026-02-22)
- [x] **Safety:** Safety benchmark (`models-benchmark/src/safety/safety.rs`) ✅ **Done** (2026-02-22)
- [ ] Register all in `BenchmarkRegistry::register_all_builtin()`
- [ ] Create/download dataset files for each benchmark in `datasets/`

#### 2.3 Additional Metrics

- [x] BLEU score metric (`models-metrics/src/accuracy.rs`) ✅ **Done** (2026-02-22)
- [x] ROUGE score metric (ROUGE-1, ROUGE-2, ROUGE-L) ✅ **Done** (2026-02-22)
- [ ] Hallucination rate metric (`models-metrics/src/hallucination.rs`)
- [ ] Factual consistency metric
- [ ] Output drift metric (`models-metrics/src/drift.rs`)
- [ ] Temperature variance metric
- [x] Cost per token metric (`models-metrics/src/efficiency.rs`) ✅ **Done** (2026-02-22)
- [x] Quality per dollar metric ✅ **Done** (2026-02-22)
- [x] Register all in `MetricsEngine::register_all_builtin()` ✅ **Done** (2026-02-22)

#### 2.4 Report Generation

- [x] JSON report export (already via `serde_json::to_string_pretty`) ✅ **Done** (2026-02-21)
- [ ] HTML report template (embedded or Askama/Tera template)
- [x] CLI `--output` flag support (auto-detect format from extension) ✅ **Done** (2026-02-21)
- [ ] `view-report` CLI command implementation
- [ ] `list-reports` CLI command implementation

#### 2.5 Model Comparison

- [x] `compare` CLI command implementation ✅ **Done** (2026-02-21 - placeholder)
- [ ] `POST /api/v1/compare` handler implementation
- [ ] `ComparisonReport` with per-category winners
- [ ] Side-by-side metric tables in terminal output

#### 2.6 Phase 2 Testing

- [ ] Unit tests for each new benchmark's `evaluate_response` / answer parsing
- [ ] Unit tests for BLEU, ROUGE, drift calculation correctness
- [ ] Integration test: compare two models (e.g., two Ollama models)
- [ ] Validate benchmark scores against known published results (within 2% on MMLU if possible)

---

### Phase 3 — Advanced Features (Dashboard + LLM-as-Judge + Distributed)

**Goal:** Web dashboard, real-time progress, LLM-as-judge, RAG benchmarks.

---

#### 3.1 Web Dashboard

- [ ] Initialize frontend project (Svelte/React + TailwindCSS) in `dashboard/` directory
- [ ] Leaderboard page — table of all models ranked by overall score
- [ ] Model detail page — per-category radar chart, per-benchmark bar charts
- [ ] Comparison page — side-by-side metrics for selected models
- [ ] Evaluation history page — trend charts over time
- [ ] Real-time evaluation progress (WebSocket or SSE from API)

#### 3.2 LLM-as-Judge

- [ ] Create judge evaluation module (`models-benchmark/src/judge/`)
- [ ] Implement configurable judge prompts
- [ ] Support using a stronger model (e.g., GPT-4) to evaluate responses
- [ ] Add `JudgeBenchmark` variant

#### 3.3 RAG Benchmarks

- [ ] Implement RAG-specific evaluation (context recall, answer faithfulness)
- [ ] Add RAG dataset format support
- [ ] Implement retrieval precision / recall metrics

#### 3.4 Distributed Evaluation

- [ ] Add message queue support (RabbitMQ / Redis Streams)
- [ ] Create worker mode for distributed benchmark execution
- [ ] Implement task distribution and result aggregation

#### 3.5 CI/CD Integration

- [ ] Create GitHub Actions workflow template for model evaluation
- [ ] Add quality gates (minimum score thresholds)
- [ ] Create `models-ci` CLI helper for CI pipelines

#### 3.6 Phase 3 Testing

- [ ] End-to-end test: submit evaluation via API, view in dashboard
- [ ] Browser-based test for dashboard pages
- [ ] Load test for concurrent evaluations

---

### Phase 4 — Research & Scale

**Goal:** Human-in-the-loop, multi-modal, plugin system.

---

#### 4.1 Human-in-the-Loop

- [ ] Implement ELO rating system
- [ ] Add human annotation interface (accept/reject/rate responses)
- [ ] Create leaderboard with both automated and human scores

#### 4.2 Multi-Modal Evaluation

- [ ] Add vision model support (image input)
- [ ] Implement image-based benchmarks
- [ ] Add audio model support (optional)

#### 4.3 Plugin System

- [ ] Define plugin API for custom benchmarks (dynamic loading or WASM)
- [ ] Create custom benchmark creation UI in dashboard
- [ ] Support custom metric registration via config

#### 4.4 Sustainability Metrics

- [ ] Track energy consumption per evaluation (GPU power draw)
- [ ] Calculate carbon footprint estimates
- [ ] Add sustainability score to reports

#### 4.5 Advanced Drift Detection

- [ ] Implement model version comparison over time
- [ ] Alert system for score regressions
- [ ] Automated re-evaluation scheduling

---

## 3. File-Level Change Map

> Quick reference of every file that needs to be created or modified.

### New Files Created in Phase 1 ✅

| File | Status | Purpose |
|---|---|---|
| `models-providers/Cargo.toml` | ✅ Done | New crate manifest |
| `models-providers/src/lib.rs` | ✅ Done | Provider re-exports |
| `models-providers/src/ollama.rs` | ✅ Done | Ollama adapter |
| `models-providers/src/openai.rs` | ✅ Done | OpenAI adapter |
| `models-benchmark/Cargo.toml` | ✅ Done | New crate manifest |
| `models-benchmark/src/lib.rs` | ✅ Done | Benchmark traits + types |
| `models-benchmark/src/registry.rs` | ✅ Done | BenchmarkRegistry |
| `models-benchmark/src/reasoning/mod.rs` | ✅ Done | Reasoning module |
| `models-benchmark/src/reasoning/mmlu.rs` | ✅ Done | MMLU benchmark |
| `models-metrics/Cargo.toml` | ✅ Done | New crate manifest |
| `models-metrics/src/lib.rs` | ✅ Done | Metric traits + engine |
| `models-metrics/src/speed.rs` | ✅ Done | Speed metrics |
| `models-metrics/src/accuracy.rs` | ✅ Done | Accuracy metrics |
| `models-core/src/providers/mod.rs` | ✅ Done | Provider trait + types |
| `models-core/src/providers/config.rs` | ✅ Done | ProviderConfig + Factory |
| `models-core/src/evaluation/mod.rs` | ✅ Done | Evaluation module |
| `models-core/src/evaluation/orchestrator.rs` | ✅ Done | EvaluationOrchestrator |
| `models-core/src/storage/mod.rs` | ✅ Done | EvaluationStorage trait + InMemoryStorage |
| `.gitignore` | ✅ Done | Git ignore file |

### Modified Files in Phase 1 ✅

| File | Status | Changes |
|---|---|---|
| `Cargo.toml` (workspace root) | ✅ Done | Add new members + deps (sqlx, maplit, futures) |
| `models-core/Cargo.toml` | ✅ Done | Add futures dependency |
| `models-core/src/lib.rs` | ✅ Done | Add `providers`, `evaluation`, `storage` modules |
| `models-core/src/domain/mod.rs` | ✅ Done | Rename `ModelProvider` → `ProviderKind`, `Dataset` → `DatasetEntity` |
| `models-core/src/domain/model.rs` | ✅ Done | Rename enum `ModelProvider` → `ProviderKind` |
| `models-core/src/domain/dataset.rs` | ✅ Done | Rename struct `Dataset` → `DatasetEntity` |
| `models-api/src/main.rs` | ✅ Done | Clean up unused imports |
| `models-api/src/handlers.rs` | ✅ Done | Add evaluation, benchmark handlers |
| `models-api/src/routes.rs` | ✅ Done | Add evaluation API routes |
| `models-api/Cargo.toml` | ✅ Done | Add models-providers, models-benchmark deps |
| `models-cli/src/main.rs` | ✅ Done | Add evaluate, list-benchmarks, compare, reports commands |
| `models-cli/Cargo.toml` | ✅ Done | Add new crate deps, rename binary to `models` |
| `Dockerfile` | ✅ Done | Fix crate paths, add multi-target (API + CLI) |
| `config/default.toml` | ✅ Done | Add evaluation, provider, benchmark sections |

### Pending Files (Phase 2+)

| File | Phase | Purpose |
|---|---|---|
| `models-providers/src/vllm.rs` | 2 | vLLM adapter |
| `models-providers/src/anthropic.rs` | 2 | Anthropic adapter |
| `models-benchmark/src/hallucination/truthfulqa.rs` | 2 | TruthfulQA |
| `models-benchmark/src/math/gsm8k.rs` | 2 | GSM8K |
| `models-benchmark/src/coding/humaneval.rs` | 2 | HumanEval |
| `models-storage/src/lib.rs` | 2 | Storage re-exports |
| `models-storage/src/postgres.rs` | 2 | PostgreSQL implementation |
| `migrations/001_initial_schema.sql` | 2 | Database schema |
| `datasets/mmlu/*.json` | 2 | MMLU full dataset |

---

## 4. Recommended Development Order (Within Phase 1)

> Follow this sequence to avoid circular dependencies and enable incremental testing.

```
1. Workspace restructure (Cargo.toml, new crate scaffolding) ✅
2. Rename conflicts in models-core (ProviderKind, DatasetEntity) ✅
3. ModelProvider trait + GenerateRequest/Response types (models-core/src/providers/) ✅
4. ProviderConfig + ProviderFactory (models-core/src/providers/config.rs) ✅
5. OllamaProvider adapter (models-providers/src/ollama.rs) ✅
   └── Test: cargo test in models-providers ✅
6. Metric trait + speed/accuracy metrics (models-metrics) ✅
   └── Test: cargo test in models-metrics ✅
7. Benchmark trait + BenchmarkRegistry + MMLU impl (models-benchmark) ✅
   └── Test: cargo test with sample dataset ✅
8. EvaluationStorage trait (models-core/src/storage/) ✅
9. PostgreSQL storage impl + migrations (models-storage) ⏳ Deferred to Phase 2
10. EvaluationOrchestrator (models-core/src/evaluation/) ✅
    └── Test: unit test with mock provider + mock storage ✅
11. CLI evaluate command (models-cli) ✅
    └── Test: end-to-end with --help ✅
12. API evaluation endpoints (models-api) ✅
    └── Test: cargo build succeeds ✅
13. Config + Dockerfile + docker-compose updates ✅
14. OpenAI provider adapter ✅
```

---

## 5. Key Architectural Decisions for the Coding Agent

1. **Keep `models-ollama` crate as-is** — it's a clean, well-tested Ollama client. The new `OllamaProvider` in `models-providers` should wrap it, not duplicate it. ✅ **Implemented**
2. **The `models-workflow` crate can remain unchanged** for Phase 1. It may eventually be absorbed by the evaluation orchestrator but is not blocking. ✅ **Kept as-is**
3. **Use `sqlx` with compile-time checking disabled** initially (use `sqlx::query()` string queries, not `sqlx::query!()`). Compile-time checking requires a live database at build time. ⏳ **Deferred to Phase 2**
4. **Start with 2 subjects × 10 questions for MMLU** to validate the pipeline without downloading the full dataset (14,000+ questions). ✅ **Implemented** (10 sample questions embedded)
5. **Use `f64` everywhere for scores** (the PRD uses both `f64` and `f32` — standardize on `f64`). ✅ **Implemented**
6. **Add `maplit` crate** to workspace dependencies for the `hashmap!` macro used extensively in the PRD. ✅ **Done**
7. **Use `async-trait` crate** (already in workspace) for all async trait definitions. ✅ **Implemented**

---

## 6. Commit History (Phase 1)

| Commit | Date | Description |
|---|---|---|
| `ec0b22b` | 2026-02-21 | docs: enhance README with comprehensive documentation |
| `365f642` | 2026-02-21 | feat: Add new workspace crates and rename conflicting types |
| `12c878a` | 2026-02-21 | docs: update action plan with progress |
| `fdc2fbc` | 2026-02-21 | feat: add provider layer, metrics engine, and benchmark framework |
| `be65bf1` | 2026-02-21 | feat: add EvaluationStorage trait and EvaluationOrchestrator |
| `5a9e44e` | 2026-02-21 | chore: update Dockerfile and config for new crate structure |
| `f095c60` | 2026-02-21 | feat: extend CLI with evaluation commands |
| `f4797e0` | 2026-02-21 | feat: add evaluation endpoints to REST API |