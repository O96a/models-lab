//! PostgreSQL storage implementation for evaluation persistence

use async_trait::async_trait;
use sqlx::postgres::{PgPool, PgPoolOptions};
use tracing::{debug, info, warn};
use uuid::Uuid;

use models_core::storage::{EvaluationStorage, StoredBenchmarkResult, StoredEvaluationReport, QueryParams};
use models_core::error::{Error, Result};

/// PostgreSQL storage backend configuration
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    /// Database connection URL
    pub url: String,
    /// Maximum number of connections in the pool
    pub max_connections: u32,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            url: "postgres://models:models_secret@localhost:5432/models_lab".to_string(),
            max_connections: 10,
        }
    }
}

impl PostgresConfig {
    /// Create a new configuration with the given URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }

    /// Set the maximum number of connections
    pub fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }
}

/// PostgreSQL storage implementation
pub struct PostgresStorage {
    pool: PgPool,
}

impl PostgresStorage {
    /// Create a new PostgreSQL storage instance
    pub async fn new(config: PostgresConfig) -> Result<Self> {
        info!("Connecting to PostgreSQL at {}", config.url.replace(|c: char| c.is_alphanumeric() || c == ':' || c == '@' || c == '/' || c == '.', "*"));

        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&config.url)
            .await
            .map_err(|e| Error::Database(format!("Failed to connect to PostgreSQL: {}", e)))?;

        info!("Successfully connected to PostgreSQL");

        Ok(Self { pool })
    }

    /// Create a new PostgreSQL storage instance from an existing pool
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run database migrations
    pub async fn run_migrations(&self) -> Result<()> {
        info!("Running database migrations...");

        // Read and execute migration files
        let migration_sql = include_str!("../../migrations/001_initial_schema.sql");

        sqlx::raw_sql(migration_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Database(format!("Failed to run migrations: {}", e)))?;

        info!("Database migrations completed successfully");
        Ok(())
    }

    /// Close the connection pool
    pub async fn close(&self) {
        self.pool.close().await;
        debug!("PostgreSQL connection pool closed");
    }
}

#[async_trait]
impl EvaluationStorage for PostgresStorage {
    async fn save_benchmark_result(
        &self,
        result: &StoredBenchmarkResult,
    ) -> Result<Uuid> {
        debug!("Saving benchmark result for model {} on {}", result.model_name, result.benchmark_id);

        let id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO benchmark_results
                (id, benchmark_id, model_name, provider_name, accuracy, total_samples,
                 correct_count, mean_latency_ms, raw_result, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            "#,
        )
        .bind(result.id)
        .bind(&result.benchmark_id)
        .bind(&result.model_name)
        .bind(&result.provider_name)
        .bind(result.accuracy)
        .bind(result.total_samples as i32)
        .bind(result.correct_count as i32)
        .bind(result.mean_latency_ms)
        .bind(&result.raw_result)
        .bind(result.created_at)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to save benchmark result: {}", e)))?;

        debug!("Saved benchmark result with id {}", id);
        Ok(id)
    }

    async fn save_evaluation_report(
        &self,
        report: &StoredEvaluationReport,
    ) -> Result<Uuid> {
        debug!("Saving evaluation report for model {}", report.model_name);

        let id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO evaluation_reports
                (id, model_name, provider_name, overall_score, benchmark_count,
                 category_scores, raw_report, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            "#,
        )
        .bind(report.id)
        .bind(&report.model_name)
        .bind(&report.provider_name)
        .bind(report.overall_score)
        .bind(report.benchmark_count as i32)
        .bind(&report.category_scores)
        .bind(&report.raw_report)
        .bind(report.created_at)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to save evaluation report: {}", e)))?;

        debug!("Saved evaluation report with id {}", id);
        Ok(id)
    }

    async fn get_benchmark_result(&self, id: Uuid) -> Result<Option<StoredBenchmarkResult>> {
        debug!("Fetching benchmark result {}", id);

        let result = sqlx::query_as::<_, StoredBenchmarkResultRow>(
            r#"
            SELECT id, benchmark_id, model_name, provider_name, accuracy,
                   total_samples, correct_count, mean_latency_ms, raw_result, created_at
            FROM benchmark_results
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to fetch benchmark result: {}", e)))?;

        Ok(result.map(|r| r.into()))
    }

    async fn get_evaluation_report(&self, id: Uuid) -> Result<Option<StoredEvaluationReport>> {
        debug!("Fetching evaluation report {}", id);

        let result = sqlx::query_as::<_, StoredEvaluationReportRow>(
            r#"
            SELECT id, model_name, provider_name, overall_score, benchmark_count,
                   category_scores, raw_report, created_at
            FROM evaluation_reports
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to fetch evaluation report: {}", e)))?;

        Ok(result.map(|r| r.into()))
    }

    async fn list_benchmark_results(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredBenchmarkResult>> {
        debug!("Listing benchmark results with query: model={:?}, benchmark={:?}",
               query.model_name, query.benchmark_id);

        let limit = query.limit.unwrap_or(100) as i32;
        let offset = query.offset.unwrap_or(0) as i32;

        let results = match (&query.model_name, &query.benchmark_id) {
            (Some(model), Some(benchmark)) => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    WHERE model_name = $1 AND benchmark_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    "#,
                )
                .bind(model)
                .bind(benchmark)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (Some(model), None) => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(model)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, Some(benchmark)) => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    WHERE benchmark_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(benchmark)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            (None, None) => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    "#,
                )
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| Error::Database(format!("Failed to list benchmark results: {}", e)))?;

        Ok(results.into_iter().map(|r| r.into()).collect())
    }

    async fn list_evaluation_reports(
        &self,
        query: QueryParams,
    ) -> Result<Vec<StoredEvaluationReport>> {
        debug!("Listing evaluation reports with query: model={:?}", query.model_name);

        let limit = query.limit.unwrap_or(100) as i32;
        let offset = query.offset.unwrap_or(0) as i32;

        let results = match &query.model_name {
            Some(model) => {
                sqlx::query_as::<_, StoredEvaluationReportRow>(
                    r#"
                    SELECT id, model_name, provider_name, overall_score, benchmark_count,
                           category_scores, raw_report, created_at
                    FROM evaluation_reports
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    "#,
                )
                .bind(model)
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query_as::<_, StoredEvaluationReportRow>(
                    r#"
                    SELECT id, model_name, provider_name, overall_score, benchmark_count,
                           category_scores, raw_report, created_at
                    FROM evaluation_reports
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    "#,
                )
                .bind(limit)
                .bind(offset)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| Error::Database(format!("Failed to list evaluation reports: {}", e)))?;

        Ok(results.into_iter().map(|r| r.into()).collect())
    }

    async fn get_benchmark_history(
        &self,
        model_name: &str,
        benchmark_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<StoredBenchmarkResult>> {
        debug!("Fetching benchmark history for model {}, benchmark {:?}", model_name, benchmark_id);

        let results = match benchmark_id {
            Some(benchmark) => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    WHERE model_name = $1 AND benchmark_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    "#,
                )
                .bind(model_name)
                .bind(benchmark)
                .bind(limit as i32)
                .fetch_all(&self.pool)
                .await
            }
            None => {
                sqlx::query_as::<_, StoredBenchmarkResultRow>(
                    r#"
                    SELECT id, benchmark_id, model_name, provider_name, accuracy,
                           total_samples, correct_count, mean_latency_ms, raw_result, created_at
                    FROM benchmark_results
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    "#,
                )
                .bind(model_name)
                .bind(limit as i32)
                .fetch_all(&self.pool)
                .await
            }
        }
        .map_err(|e| Error::Database(format!("Failed to fetch benchmark history: {}", e)))?;

        Ok(results.into_iter().map(|r| r.into()).collect())
    }

    async fn delete_benchmark_result(&self, id: Uuid) -> Result<bool> {
        debug!("Deleting benchmark result {}", id);

        let result = sqlx::query(
            "DELETE FROM benchmark_results WHERE id = $1"
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to delete benchmark result: {}", e)))?;

        let deleted = result.rows_affected() > 0;
        if deleted {
            debug!("Deleted benchmark result {}", id);
        } else {
            warn!("Benchmark result {} not found for deletion", id);
        }

        Ok(deleted)
    }

    async fn delete_evaluation_report(&self, id: Uuid) -> Result<bool> {
        debug!("Deleting evaluation report {}", id);

        let result = sqlx::query(
            "DELETE FROM evaluation_reports WHERE id = $1"
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| Error::Database(format!("Failed to delete evaluation report: {}", e)))?;

        let deleted = result.rows_affected() > 0;
        if deleted {
            debug!("Deleted evaluation report {}", id);
        } else {
            warn!("Evaluation report {} not found for deletion", id);
        }

        Ok(deleted)
    }

    async fn health_check(&self) -> Result<bool> {
        let result: Option<(i32,)> = sqlx::query_as("SELECT 1")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| Error::Database(format!("Health check failed: {}", e)))?;

        Ok(result.is_some())
    }
}

// ============================================================================
// Database Row Types (for sqlx mapping)
// ============================================================================

/// Database row type for benchmark results
#[derive(sqlx::FromRow)]
struct StoredBenchmarkResultRow {
    id: Uuid,
    benchmark_id: String,
    model_name: String,
    provider_name: String,
    accuracy: f64,
    total_samples: i32,
    correct_count: i32,
    mean_latency_ms: f64,
    raw_result: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl From<StoredBenchmarkResultRow> for StoredBenchmarkResult {
    fn from(row: StoredBenchmarkResultRow) -> Self {
        StoredBenchmarkResult {
            id: row.id,
            benchmark_id: row.benchmark_id,
            model_name: row.model_name,
            provider_name: row.provider_name,
            accuracy: row.accuracy,
            total_samples: row.total_samples as usize,
            correct_count: row.correct_count as usize,
            mean_latency_ms: row.mean_latency_ms,
            raw_result: row.raw_result,
            created_at: row.created_at,
        }
    }
}

/// Database row type for evaluation reports
#[derive(sqlx::FromRow)]
struct StoredEvaluationReportRow {
    id: Uuid,
    model_name: String,
    provider_name: String,
    overall_score: f64,
    benchmark_count: i32,
    category_scores: serde_json::Value,
    raw_report: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl From<StoredEvaluationReportRow> for StoredEvaluationReport {
    fn from(row: StoredEvaluationReportRow) -> Self {
        StoredEvaluationReport {
            id: row.id,
            model_name: row.model_name,
            provider_name: row.provider_name,
            overall_score: row.overall_score,
            benchmark_count: row.benchmark_count as usize,
            category_scores: row.category_scores,
            raw_report: row.raw_report,
            created_at: row.created_at,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PostgresConfig::default();
        assert_eq!(config.max_connections, 10);
    }

    #[test]
    fn test_config_builder() {
        let config = PostgresConfig::new("postgres://user:pass@localhost/db")
            .with_max_connections(20);

        assert_eq!(config.url, "postgres://user:pass@localhost/db");
        assert_eq!(config.max_connections, 20);
    }
}