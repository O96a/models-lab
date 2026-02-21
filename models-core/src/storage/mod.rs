//! Evaluation storage trait and types

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;

// ============================================================================
// Storage Types
// ============================================================================

/// Stored benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredBenchmarkResult {
    /// Unique ID
    pub id: Uuid,
    /// Benchmark ID
    pub benchmark_id: String,
    /// Model name
    pub model_name: String,
    /// Provider name
    pub provider_name: String,
    /// Accuracy score
    pub accuracy: f64,
    /// Number of samples
    pub total_samples: usize,
    /// Number of correct
    pub correct_count: usize,
    /// Mean latency in ms
    pub mean_latency_ms: f64,
    /// Raw result JSON
    pub raw_result: serde_json::Value,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// Stored evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvaluationReport {
    /// Unique ID
    pub id: Uuid,
    /// Model name
    pub model_name: String,
    /// Provider name
    pub provider_name: String,
    /// Overall score
    pub overall_score: f64,
    /// Number of benchmarks run
    pub benchmark_count: usize,
    /// Category scores
    pub category_scores: serde_json::Value,
    /// Raw report JSON
    pub raw_report: serde_json::Value,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// Query parameters for listing results
#[derive(Debug, Clone, Default)]
pub struct QueryParams {
    /// Filter by model name
    pub model_name: Option<String>,
    /// Filter by benchmark ID
    pub benchmark_id: Option<String>,
    /// Limit results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_name = Some(model.into());
        self
    }

    pub fn with_benchmark(mut self, benchmark: impl Into<String>) -> Self {
        self.benchmark_id = Some(benchmark.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

// ============================================================================
// Storage Trait
// ============================================================================

/// Trait for evaluation storage backends
#[async_trait]
pub trait EvaluationStorage: Send + Sync {
    /// Save a benchmark result
    async fn save_benchmark_result(
        &self,
        result: &StoredBenchmarkResult,
    ) -> Result<Uuid>;

    /// Save an evaluation report
    async fn save_evaluation_report(
        &self,
        report: &StoredEvaluationReport,
    ) -> Result<Uuid>;

    /// Get a specific benchmark result by ID
    async fn get_benchmark_result(&self, id: Uuid) -> Result<Option<StoredBenchmarkResult>>;

    /// Get a specific evaluation report by ID
    async fn get_evaluation_report(&self, id: Uuid) -> Result<Option<StoredEvaluationReport>>;

    /// List benchmark results with optional filters
    async fn list_benchmark_results(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredBenchmarkResult>>;

    /// List evaluation reports with optional filters
    async fn list_evaluation_reports(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredEvaluationReport>>;

    /// Get benchmark history for a model
    async fn get_benchmark_history(
        &self,
        model_name: &str,
        benchmark_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StoredBenchmarkResult>>;

    /// Delete a benchmark result
    async fn delete_benchmark_result(&self, id: Uuid) -> Result<bool>;

    /// Delete an evaluation report
    async fn delete_evaluation_report(&self, id: Uuid) -> Result<bool>;

    /// Check if the storage is healthy
    async fn health_check(&self) -> Result<bool>;
}

// ============================================================================
// In-Memory Storage (for testing)
// ============================================================================

/// In-memory storage implementation for testing
pub struct InMemoryStorage {
    benchmark_results: std::sync::RwLock<Vec<StoredBenchmarkResult>>,
    evaluation_reports: std::sync::RwLock<Vec<StoredEvaluationReport>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            benchmark_results: std::sync::RwLock::new(Vec::new()),
            evaluation_reports: std::sync::RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EvaluationStorage for InMemoryStorage {
    async fn save_benchmark_result(
        &self,
        result: &StoredBenchmarkResult,
    ) -> Result<Uuid> {
        let mut results = self.benchmark_results.write().unwrap();
        let id = result.id;
        results.push(result.clone());
        Ok(id)
    }

    async fn save_evaluation_report(
        &self,
        report: &StoredEvaluationReport,
    ) -> Result<Uuid> {
        let mut reports = self.evaluation_reports.write().unwrap();
        let id = report.id;
        reports.push(report.clone());
        Ok(id)
    }

    async fn get_benchmark_result(&self, id: Uuid) -> Result<Option<StoredBenchmarkResult>> {
        let results = self.benchmark_results.read().unwrap();
        Ok(results.iter().find(|r| r.id == id).cloned())
    }

    async fn get_evaluation_report(&self, id: Uuid) -> Result<Option<StoredEvaluationReport>> {
        let reports = self.evaluation_reports.read().unwrap();
        Ok(reports.iter().find(|r| r.id == id).cloned())
    }

    async fn list_benchmark_results(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredBenchmarkResult>> {
        let results = self.benchmark_results.read().unwrap();
        let mut filtered: Vec<_> = results
            .iter()
            .filter(|r| {
                query.model_name.as_ref().map_or(true, |m| r.model_name == *m)
                    && query.benchmark_id.as_ref().map_or(true, |b| r.benchmark_id == *b)
            })
            .cloned()
            .collect();

        if let Some(offset) = query.offset {
            filtered = filtered.into_iter().skip(offset).collect();
        }
        if let Some(limit) = query.limit {
            filtered.truncate(limit);
        }

        Ok(filtered)
    }

    async fn list_evaluation_reports(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredEvaluationReport>> {
        let reports = self.evaluation_reports.read().unwrap();
        let mut filtered: Vec<_> = reports
            .iter()
            .filter(|r| {
                query.model_name.as_ref().map_or(true, |m| r.model_name == *m)
            })
            .cloned()
            .collect();

        if let Some(offset) = query.offset {
            filtered = filtered.into_iter().skip(offset).collect();
        }
        if let Some(limit) = query.limit {
            filtered.truncate(limit);
        }

        Ok(filtered)
    }

    async fn get_benchmark_history(
        &self,
        model_name: &str,
        benchmark_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StoredBenchmarkResult>> {
        let results = self.benchmark_results.read().unwrap();
        let mut history: Vec<_> = results
            .iter()
            .filter(|r| {
                r.model_name == model_name
                    && benchmark_id.map_or(true, |b| r.benchmark_id == b)
            })
            .cloned()
            .collect();

        history.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        history.truncate(limit);

        Ok(history)
    }

    async fn delete_benchmark_result(&self, id: Uuid) -> Result<bool> {
        let mut results = self.benchmark_results.write().unwrap();
        let len_before = results.len();
        results.retain(|r| r.id != id);
        Ok(results.len() < len_before)
    }

    async fn delete_evaluation_report(&self, id: Uuid) -> Result<bool> {
        let mut reports = self.evaluation_reports.write().unwrap();
        let len_before = reports.len();
        reports.retain(|r| r.id != id);
        Ok(reports.len() < len_before)
    }

    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryStorage::new();

        let result = StoredBenchmarkResult {
            id: Uuid::new_v4(),
            benchmark_id: "mmlu".to_string(),
            model_name: "llama3.2".to_string(),
            provider_name: "ollama".to_string(),
            accuracy: 0.75,
            total_samples: 100,
            correct_count: 75,
            mean_latency_ms: 500.0,
            raw_result: serde_json::json!({}),
            created_at: Utc::now(),
        };

        let id = storage.save_benchmark_result(&result).await.unwrap();
        let retrieved = storage.get_benchmark_result(id).await.unwrap();
        assert!(retrieved.is_some());
    }
}
