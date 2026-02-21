# Models Lab Enhancement — Action Plan

**Created:** 2026-02-21  
**Purpose:** Detailed, phased checklist for a coding agent to implement the enhancements described in `model-lab-enhancment.md` on top of the existing codebase.

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

- [ ] Ensure Rust 1.75+ is installed
- [ ] Ensure Docker + Docker Compose are available
- [ ] Ensure PostgreSQL 16, Redis 7 accessible (via docker-compose or local)
- [ ] Pull at least one Ollama model for testing (e.g., `llama3.2:3b`)

---

### Phase 1 — Foundation (Provider Layer + Storage + Basic Benchmark)

**Goal:** Universal provider trait, Ollama + OpenAI adapters working, PostgreSQL storage with migrations, one benchmark (MMLU) end-to-end, CLI evaluate command.

---

#### 1.1 Workspace Restructure

- [ ] Add new workspace members to root `Cargo.toml`:
  - `models-providers` — provider adapters
  - `models-benchmark` — benchmark framework
  - `models-metrics` — metrics calculations
  - `models-storage` — persistence layer
- [ ] Create directory scaffolding for each new crate (`cargo init --lib`)
- [ ] Add shared workspace dependencies: `sqlx` (postgres, runtime-tokio), `maplit`, `pin-project`
- [ ] Fix Dockerfile crate paths from `llm-*` to `models-*` (lines 8–12, 15–19, 25–29)

#### 1.2 Universal Provider Trait (`models-core`)

- [ ] Create `models-core/src/providers/mod.rs` with:
  - `ModelProvider` async trait (`generate`, `embed`, `get_model_info`, `health_check`, `generate_stream`)
  - `GenerateRequest` struct (prompt, system_prompt, temperature, max_tokens, stop_sequences, top_p, top_k, presence_penalty, frequency_penalty, seed, stream, extra_params)
  - `GenerateResponse` struct (text, tokens_used, latency_ms, time_to_first_token_ms, finish_reason, model_name, metadata)
  - `TokenUsage`, `FinishReason`, `ModelInfo`, `HealthStatus` structs/enums
- [ ] Create `models-core/src/providers/config.rs` with:
  - `ProviderConfig` struct
  - `ProviderType` tagged enum (Ollama, VLLM, OpenAI, Anthropic, etc.)
  - `ProviderFactory` struct with `create()` method
  - `ApiFormat` enum
- [ ] Rename existing `ModelProvider` enum in `domain/model.rs` to `ProviderKind`
  - Update all references in `domain/mod.rs`, `model.rs`, and downstream
- [ ] Rename existing `Dataset` in `domain/dataset.rs` to `DatasetEntity`
  - Update references in `domain/mod.rs`
- [ ] Add `pub mod providers;` to `models-core/src/lib.rs`
- [ ] Re-export key types from `lib.rs`

#### 1.3 Ollama Provider Adapter (`models-providers`)

- [ ] Create `models-providers/src/ollama.rs`:
  - `OllamaProvider` struct wrapping existing `OllamaClient`
  - Implement `ModelProvider` trait (delegate to `OllamaClient::chat` / `embed` / etc.)
  - Map `ChatResponse` → `GenerateResponse`
  - Map `ChatRequest` ← `GenerateRequest`
- [ ] Create `models-providers/src/lib.rs` re-exporting `OllamaProvider`
- [ ] Register in `ProviderFactory::create()`

#### 1.4 OpenAI Provider Adapter (`models-providers`)

- [ ] Create `models-providers/src/openai.rs`:
  - `OpenAIProvider` struct
  - Implement `ModelProvider` trait
  - Use `reqwest` to call `https://api.openai.com/v1/chat/completions`
  - Handle API key, organization header
- [ ] Register in `ProviderFactory::create()`

#### 1.5 Database Migrations + Storage Layer (`models-storage`)

- [ ] Create `migrations/001_initial_schema.sql`:
  - `evaluation_reports` table
  - `benchmark_results` table
  - `sample_results` table
  - `provider_configs` table
  - Use proper PostgreSQL `CREATE INDEX` statements (not inline INDEX)
- [ ] Create `models-storage/src/lib.rs` with `EvaluationStorage` trait (re-export from `models-core`)
- [ ] Create `models-storage/src/postgres.rs`:
  - `PostgresStorage` struct with `PgPool`
  - Implement `EvaluationStorage` trait
  - Run migrations on construction
- [ ] Create `EvaluationStorage` async trait in `models-core/src/storage/mod.rs`:
  - `save_benchmark_result`, `save_evaluation_report`, `get_evaluation_report`, `list_evaluation_reports`, `get_benchmark_history`
- [ ] Update `config.rs` to use `DatabaseConfig` with PostgreSQL URL

#### 1.6 Benchmark Framework (`models-benchmark`)

- [ ] Create `models-benchmark/src/lib.rs` with:
  - `Benchmark` async trait (`id`, `name`, `category`, `description`, `load_dataset`, `run`, `evaluate_response`)
  - `BenchmarkCategory` enum
  - `Dataset`, `DataSample`, `DatasetMetadata`, `Difficulty` structs/enums
  - `BenchmarkResult`, `SampleResult`, `BenchmarkStatistics`, `BenchmarkConfig`, `MetricScores` structs
- [ ] Create `models-benchmark/src/registry.rs`:
  - `BenchmarkRegistry` struct with `register`, `get`, `list`, `list_by_category`
- [ ] Implement MMLU benchmark in `models-benchmark/src/reasoning/mmlu.rs`:
  - Question formatting, answer parsing
  - 5-shot evaluation
  - Create placeholder dataset files in `datasets/mmlu/` (at least 2 subjects with 10 questions each for testing)
- [ ] Register MMLU in the `BenchmarkRegistry`

#### 1.7 Core Metrics (`models-metrics`)

- [ ] Create `models-metrics/src/lib.rs` with:
  - `Metric` trait (`name`, `calculate`, `unit`, `direction`)
  - `MetricUnit`, `MetricDirection` enums
  - `MetricData` struct
  - `MetricsEngine` struct
- [ ] Implement speed metrics in `models-metrics/src/speed.rs`:
  - `TokensPerSecondMetric`
  - `FirstTokenLatencyMetric`
  - `P95LatencyMetric`
- [ ] Implement accuracy metrics in `models-metrics/src/accuracy.rs`:
  - `ExactMatchMetric`
  - `F1ScoreMetric`

#### 1.8 Evaluation Orchestrator (`models-core`)

- [ ] Create `models-core/src/evaluation/orchestrator.rs`:
  - `EvaluationOrchestrator` struct
  - `run_full_evaluation()` — run benchmarks, calculate metrics, generate report
  - `run_benchmarks()` — run specific benchmarks
  - `compare_models()` — run same benchmarks across multiple providers
  - `EvaluationConfig`, `EvaluationReport`, `CategoryScore`, `DetailedMetrics`, `CostAnalysis`, `ComparisonReport` structs
- [ ] Create `models-core/src/evaluation/mod.rs` re-exporting types

#### 1.9 CLI Rewrite (`models-cli`)

- [ ] Add `evaluate` subcommand:
  - `--provider`, `--model`, `--benchmarks`, `--output`, `--endpoint`, `--api-key`
  - Instantiate provider via `ProviderFactory`
  - Run `EvaluationOrchestrator::run_full_evaluation()`
  - Print formatted report to terminal
- [ ] Add `list-benchmarks` subcommand (with `--category` filter)
- [ ] Preserve existing `ollama` subcommand for backward compat
- [ ] Add progress indicators (`indicatif` crate recommended)

#### 1.10 API Extension (`models-api`)

- [ ] Add new routes:
  - `POST /api/v1/evaluate` — create evaluation
  - `GET /api/v1/evaluations` — list evaluations
  - `GET /api/v1/evaluations/:id` — get evaluation
  - `POST /api/v1/compare` — compare models
  - `GET /api/v1/benchmarks` — list benchmarks
- [ ] Update `AppState` to include `EvaluationOrchestrator` and `EvaluationStorage`
- [ ] Keep existing Ollama-specific endpoints for backward compatibility

#### 1.11 Config & Infrastructure Updates

- [ ] Extend `config/default.toml` with:
  - `[evaluation]` section (default_temperature, default_max_tokens, default_seed)
  - `[providers.ollama]`, `[providers.vllm]`, `[providers.openai]` sections
  - `[benchmarks]` section (data_directory, cache_directory)
- [ ] Extend `Config` struct in `models-core/src/config.rs` to parse new sections
- [ ] Update Dockerfile for multi-target build (API + CLI stages)
- [ ] Update `docker-compose.yml` if needed

#### 1.12 Phase 1 Testing

- [ ] Unit tests for `GenerateRequest`/`GenerateResponse` serialization
- [ ] Unit tests for `OllamaProvider` (mock HTTP or integration)
- [ ] Unit tests for `ExactMatchMetric`, `F1ScoreMetric`, `TokensPerSecondMetric`
- [ ] Unit tests for `BenchmarkRegistry` (register, get, list)
- [ ] Integration test: `models evaluate --provider ollama --model llama3.2 --benchmarks mmlu`
- [ ] Verify `cargo build --release` succeeds for all crates
- [ ] Verify `cargo test` passes

---

### Phase 2 — Comprehensive Evaluation (More Benchmarks + Metrics + Reports)

**Goal:** 10+ benchmarks, full metric coverage, model comparison, report generation.

---

#### 2.1 Additional Providers

- [ ] Implement `VLLMProvider` in `models-providers/src/vllm.rs` (OpenAI-compatible API)
- [ ] Implement `AnthropicProvider` in `models-providers/src/anthropic.rs`
- [ ] Implement `LMStudioProvider` (OpenAI-compatible, shares logic with vLLM)
- [ ] Register all in `ProviderFactory`

#### 2.2 Additional Benchmarks

- [ ] **Reasoning:** BBH benchmark (`models-benchmark/src/reasoning/bbh.rs`)
- [ ] **Reasoning:** ARC benchmark (`models-benchmark/src/reasoning/arc.rs`)
- [ ] **Math:** GSM8K benchmark (`models-benchmark/src/math/gsm8k.rs`)
- [ ] **Math:** MATH benchmark (`models-benchmark/src/math/math.rs`)
- [ ] **Coding:** HumanEval benchmark (`models-benchmark/src/coding/humaneval.rs`)
- [ ] **Coding:** MBPP benchmark (`models-benchmark/src/coding/mbpp.rs`)
- [ ] **Hallucination:** TruthfulQA benchmark (`models-benchmark/src/hallucination/truthfulqa.rs`)
- [ ] **Context:** LongContext benchmark (`models-benchmark/src/context/long_context.rs`)
- [ ] **Context:** NeedleInHaystack benchmark (`models-benchmark/src/context/needle.rs`)
- [ ] **Instruction:** AlpacaEval benchmark (`models-benchmark/src/instruction/alpaca_eval.rs`)
- [ ] **Multi-turn:** MT-Bench benchmark (`models-benchmark/src/multiturn/mt_bench.rs`)
- [ ] **Safety:** Safety benchmark (`models-benchmark/src/safety/safety.rs`)
- [ ] Register all in `BenchmarkRegistry::register_all_builtin()`
- [ ] Create/download dataset files for each benchmark in `datasets/`

#### 2.3 Additional Metrics

- [ ] BLEU score metric (`models-metrics/src/accuracy.rs`)
- [ ] ROUGE score metric (ROUGE-1, ROUGE-2, ROUGE-L)
- [ ] Hallucination rate metric (`models-metrics/src/hallucination.rs`)
- [ ] Factual consistency metric
- [ ] Output drift metric (`models-metrics/src/drift.rs`)
- [ ] Temperature variance metric
- [ ] Cost per token metric (`models-metrics/src/efficiency.rs`)
- [ ] Quality per token metric
- [ ] Register all in `MetricsEngine::register_all_builtin()`

#### 2.4 Report Generation

- [ ] JSON report export (already via `serde_json::to_string_pretty`)
- [ ] HTML report template (embedded or Askama/Tera template)
- [ ] CLI `--output` flag support (auto-detect format from extension)
- [ ] `view-report` CLI command implementation
- [ ] `list-reports` CLI command implementation

#### 2.5 Model Comparison

- [ ] `compare` CLI command implementation
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

### New Files

| File | Phase | Purpose |
|---|---|---|
| `models-providers/Cargo.toml` | 1 | New crate manifest |
| `models-providers/src/lib.rs` | 1 | Provider re-exports |
| `models-providers/src/ollama.rs` | 1 | Ollama adapter |
| `models-providers/src/openai.rs` | 1 | OpenAI adapter |
| `models-providers/src/vllm.rs` | 2 | vLLM adapter |
| `models-providers/src/anthropic.rs` | 2 | Anthropic adapter |
| `models-benchmark/Cargo.toml` | 1 | New crate manifest |
| `models-benchmark/src/lib.rs` | 1 | Benchmark traits + types |
| `models-benchmark/src/registry.rs` | 1 | BenchmarkRegistry |
| `models-benchmark/src/reasoning/mmlu.rs` | 1 | MMLU benchmark |
| `models-benchmark/src/hallucination/truthfulqa.rs` | 2 | TruthfulQA |
| `models-benchmark/src/math/gsm8k.rs` | 2 | GSM8K |
| `models-benchmark/src/coding/humaneval.rs` | 2 | HumanEval |
| _(+ 8 more benchmark files)_ | 2 | See Phase 2 list |
| `models-metrics/Cargo.toml` | 1 | New crate manifest |
| `models-metrics/src/lib.rs` | 1 | Metric traits + engine |
| `models-metrics/src/speed.rs` | 1 | Speed metrics |
| `models-metrics/src/accuracy.rs` | 1 | Accuracy metrics |
| `models-metrics/src/hallucination.rs` | 2 | Hallucination metrics |
| `models-metrics/src/drift.rs` | 2 | Drift metrics |
| `models-metrics/src/efficiency.rs` | 2 | Efficiency metrics |
| `models-storage/Cargo.toml` | 1 | New crate manifest |
| `models-storage/src/lib.rs` | 1 | Storage re-exports |
| `models-storage/src/postgres.rs` | 1 | PostgreSQL implementation |
| `models-core/src/providers/mod.rs` | 1 | Provider trait + types |
| `models-core/src/providers/config.rs` | 1 | ProviderConfig + Factory |
| `models-core/src/evaluation/mod.rs` | 1 | Evaluation module |
| `models-core/src/evaluation/orchestrator.rs` | 1 | EvaluationOrchestrator |
| `models-core/src/storage/mod.rs` | 1 | EvaluationStorage trait |
| `migrations/001_initial_schema.sql` | 1 | Database schema |
| `datasets/mmlu/*.json` | 1 | MMLU sample data |

### Modified Files

| File | Phase | Changes |
|---|---|---|
| `Cargo.toml` (workspace root) | 1 | Add new members + deps (sqlx, maplit) |
| `models-core/Cargo.toml` | 1 | Add sqlx, async-trait deps |
| `models-core/src/lib.rs` | 1 | Add `providers`, `evaluation`, `storage` modules |
| `models-core/src/domain/mod.rs` | 1 | Rename `ModelProvider` → `ProviderKind`, `Dataset` → `DatasetEntity` |
| `models-core/src/domain/model.rs` | 1 | Rename enum `ModelProvider` → `ProviderKind` |
| `models-core/src/domain/dataset.rs` | 1 | Rename struct `Dataset` → `DatasetEntity` |
| `models-core/src/config.rs` | 1 | Extend with evaluation, provider, benchmark sections |
| `models-core/src/error.rs` | 1 | Add benchmark, storage, evaluation error variants |
| `models-core/src/inference.rs` | 1 | Deprecate or keep for backward compat |
| `models-api/src/main.rs` | 1 | Add orchestrator + storage to AppState |
| `models-api/src/handlers.rs` | 1 | Add evaluation, comparison, benchmark handlers |
| `models-api/src/routes.rs` | 1 | Add evaluation API routes |
| `models-api/Cargo.toml` | 1 | Add models-storage, models-benchmark deps |
| `models-cli/src/main.rs` | 1 | Add evaluate, list-benchmarks, compare commands |
| `models-cli/Cargo.toml` | 1 | Add new crate deps |
| `Dockerfile` | 1 | Fix crate paths, add multi-target |
| `config/default.toml` | 1 | Add evaluation, provider, benchmark sections |
| `docker-compose.yml` | 1 | Update env vars, possibly add volumes for datasets |

---

## 4. Recommended Development Order (Within Phase 1)

> Follow this sequence to avoid circular dependencies and enable incremental testing.

```
1. Workspace restructure (Cargo.toml, new crate scaffolding)
2. Rename conflicts in models-core (ProviderKind, DatasetEntity)
3. ModelProvider trait + GenerateRequest/Response types (models-core/src/providers/)
4. ProviderConfig + ProviderFactory (models-core/src/providers/config.rs)
5. OllamaProvider adapter (models-providers/src/ollama.rs)
   └── Test: cargo test in models-providers (mock or Ollama integration)
6. Metric trait + speed/accuracy metrics (models-metrics)
   └── Test: cargo test in models-metrics
7. Benchmark trait + BenchmarkRegistry + MMLU impl (models-benchmark)
   └── Test: cargo test with sample dataset
8. EvaluationStorage trait (models-core/src/storage/)
9. PostgreSQL storage impl + migrations (models-storage)
   └── Test: integration test against local Postgres
10. EvaluationOrchestrator (models-core/src/evaluation/)
    └── Test: unit test with mock provider + mock storage
11. CLI evaluate command (models-cli)
    └── Test: end-to-end with Ollama
12. API evaluation endpoints (models-api)
    └── Test: curl commands
13. Config + Dockerfile + docker-compose updates
14. OpenAI provider adapter
```

---

## 5. Key Architectural Decisions for the Coding Agent

1. **Keep `models-ollama` crate as-is** — it's a clean, well-tested Ollama client. The new `OllamaProvider` in `models-providers` should wrap it, not duplicate it.
2. **The `models-workflow` crate can remain unchanged** for Phase 1. It may eventually be absorbed by the evaluation orchestrator but is not blocking.
3. **Use `sqlx` with compile-time checking disabled** initially (use `sqlx::query()` string queries, not `sqlx::query!()`). Compile-time checking requires a live database at build time.
4. **Start with 2 subjects × 10 questions for MMLU** to validate the pipeline without downloading the full dataset (14,000+ questions).
5. **Use `f64` everywhere for scores** (the PRD uses both `f64` and `f32` — standardize on `f64`).
6. **Add `maplit` crate** to workspace dependencies for the `hashmap!` macro used extensively in the PRD.
7. **Use `async-trait` crate** (already in workspace) for all async trait definitions.
