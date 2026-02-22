-- Models Lab Database Schema
-- Migration 001: Initial Schema
-- Created: 2026-02-22

-- ============================================================================
-- Provider Configurations
-- Stores provider settings for different LLM backends
-- ============================================================================

CREATE TABLE IF NOT EXISTS provider_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    provider_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_provider_configs_type ON provider_configs(provider_type);
CREATE INDEX idx_provider_configs_active ON provider_configs(is_active);

-- ============================================================================
-- Evaluation Reports
-- Stores aggregate evaluation results for a model
-- ============================================================================

CREATE TABLE IF NOT EXISTS evaluation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    provider_name VARCHAR(255) NOT NULL,
    overall_score DOUBLE PRECISION NOT NULL,
    benchmark_count INTEGER NOT NULL DEFAULT 0,
    category_scores JSONB NOT NULL DEFAULT '{}',
    raw_report JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evaluation_reports_model ON evaluation_reports(model_name);
CREATE INDEX idx_evaluation_reports_provider ON evaluation_reports(provider_name);
CREATE INDEX idx_evaluation_reports_created ON evaluation_reports(created_at DESC);
CREATE INDEX idx_evaluation_reports_score ON evaluation_reports(overall_score DESC);

-- ============================================================================
-- Benchmark Results
-- Stores individual benchmark run results
-- ============================================================================

CREATE TABLE IF NOT EXISTS benchmark_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    provider_name VARCHAR(255) NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    total_samples INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    mean_latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    raw_result JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_benchmark_results_benchmark ON benchmark_results(benchmark_id);
CREATE INDEX idx_benchmark_results_model ON benchmark_results(model_name);
CREATE INDEX idx_benchmark_results_provider ON benchmark_results(provider_name);
CREATE INDEX idx_benchmark_results_created ON benchmark_results(created_at DESC);
CREATE INDEX idx_benchmark_results_accuracy ON benchmark_results(accuracy DESC);

-- ============================================================================
-- Sample Results
-- Stores individual sample-level results for detailed analysis
-- ============================================================================

CREATE TABLE IF NOT EXISTS sample_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    benchmark_result_id UUID NOT NULL REFERENCES benchmark_results(id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    input_text TEXT NOT NULL,
    expected_output TEXT,
    actual_output TEXT,
    is_correct BOOLEAN,
    latency_ms DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sample_results_benchmark_result ON sample_results(benchmark_result_id);
CREATE INDEX idx_sample_results_correct ON sample_results(is_correct);

-- ============================================================================
-- Model Comparison History
-- Stores model comparison results
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_a VARCHAR(255) NOT NULL,
    model_b VARCHAR(255) NOT NULL,
    comparison_data JSONB NOT NULL DEFAULT '{}',
    winner VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_model_comparisons_models ON model_comparisons(model_a, model_b);
CREATE INDEX idx_model_comparisons_created ON model_comparisons(created_at DESC);

-- ============================================================================
-- Benchmark Metadata
-- Stores metadata about available benchmarks
-- ============================================================================

CREATE TABLE IF NOT EXISTS benchmark_metadata (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT,
    default_samples INTEGER DEFAULT 100,
    metrics JSONB DEFAULT '[]',
    config_schema JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_benchmark_metadata_category ON benchmark_metadata(category);

-- ============================================================================
-- Audit Log
-- Tracks changes and operations for debugging and compliance
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(50) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID,
    old_data JSONB,
    new_data JSONB,
    user_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_operation ON audit_log(operation);
CREATE INDEX idx_audit_log_table ON audit_log(table_name);
CREATE INDEX idx_audit_log_created ON audit_log(created_at DESC);

-- ============================================================================
-- Utility Functions
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_provider_configs_updated_at
    BEFORE UPDATE ON provider_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_benchmark_metadata_updated_at
    BEFORE UPDATE ON benchmark_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Insert default benchmark metadata
INSERT INTO benchmark_metadata (id, name, category, description, default_samples) VALUES
    ('mmlu', 'Massive Multitask Language Understanding', 'reasoning', 'Measures knowledge across 57 subjects', 100),
    ('bbh', 'Big-Bench Hard', 'reasoning', 'Challenging reasoning tasks', 100),
    ('gsm8k', 'Grade School Math 8K', 'math', 'Multi-step arithmetic reasoning', 100),
    ('humaneval', 'HumanEval', 'coding', 'Code generation benchmark', 164),
    ('truthfulqa', 'TruthfulQA', 'hallucination', 'Measures truthfulness of responses', 100)
ON CONFLICT (id) DO NOTHING;