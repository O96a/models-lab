//! Report Generation Tests
//!
//! Tests for generating and validating evaluation reports.

use models_benchmark::{BenchmarkRegistry, BenchmarkConfig, BenchmarkResult, BenchmarkStatistics, SampleResult};
use models_core::{
    EvaluationReport, EvaluationConfig, CategoryScore, DetailedMetrics,
    ComparisonReport, ComparisonConfig, ModelComparisonResult, ModelCategoryScore, ModelBenchmarkResult,
    ModelConfig as ComparisonModelConfig,
    generate_html_report, generate_comparison_html_report, generate_html_from_stored,
    HtmlReportConfig, StoredEvaluationReport, InMemoryStorage,
};
use models_lab_tests::{MockProvider, test_utils};
use chrono::Utc;
use std::collections::HashMap;

/// Test JSON report generation
#[test]
fn test_json_report_generation() {
    // Create a mock evaluation report
    let report = create_test_evaluation_report();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&report)
        .expect("Should serialize to JSON");

    // Verify JSON structure
    assert!(json.contains("\"id\""), "JSON should have id");
    assert!(json.contains("\"model_name\""), "JSON should have model_name");
    assert!(json.contains("\"provider_name\""), "JSON should have provider_name");
    assert!(json.contains("\"overall_score\""), "JSON should have overall_score");
    assert!(json.contains("\"category_scores\""), "JSON should have category_scores");
    assert!(json.contains("\"benchmark_results\""), "JSON should have benchmark_results");
    assert!(json.contains("\"metrics\""), "JSON should have metrics");
    assert!(json.contains("\"config\""), "JSON should have config");
    assert!(json.contains("\"timestamp\""), "JSON should have timestamp");

    println!("✅ JSON report structure verified");
}

/// Test JSON report deserialization
#[test]
fn test_json_report_deserialization() {
    let report = create_test_evaluation_report();

    // Serialize
    let json = serde_json::to_string(&report).unwrap();

    // Deserialize
    let deserialized: EvaluationReport = serde_json::from_str(&json)
        .expect("Should deserialize from JSON");

    // Verify fields match
    assert_eq!(deserialized.model_name, report.model_name);
    assert_eq!(deserialized.provider_name, report.provider_name);
    assert_eq!(deserialized.overall_score, report.overall_score);
    assert_eq!(deserialized.benchmark_results.len(), report.benchmark_results.len());
    assert_eq!(deserialized.category_scores.len(), report.category_scores.len());

    println!("✅ JSON report round-trip verified");
}

/// Test report fields are valid
#[test]
fn test_report_fields_valid() {
    let report = create_test_evaluation_report();

    // Verify ID is valid UUID
    assert!(!report.id.to_string().is_empty(), "Should have ID");

    // Verify model and provider names
    assert!(!report.model_name.is_empty(), "Should have model name");
    assert!(!report.provider_name.is_empty(), "Should have provider name");

    // Verify overall score is valid
    assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0,
        "Overall score should be between 0 and 1");

    // Verify timestamps
    assert!(report.timestamp <= Utc::now(), "Timestamp should not be in the future");

    // Verify metrics
    assert!(report.metrics.mean_latency_ms >= 0.0, "Mean latency should be non-negative");
    assert!(report.metrics.p95_latency_ms >= 0.0, "P95 latency should be non-negative");
    assert!(report.metrics.error_count >= 0, "Error count should be non-negative");

    // Verify category scores
    for cat in &report.category_scores {
        assert!(!cat.category.is_empty(), "Category should have name");
        assert!(cat.score >= 0.0 && cat.score <= 1.0,
            "Category score should be between 0 and 1");
        assert!(cat.sample_count > 0, "Category should have samples");
    }

    // Verify benchmark results
    for result in &report.benchmark_results {
        assert!(!result.benchmark_id.is_empty(), "Result should have benchmark_id");
        assert!(!result.benchmark_name.is_empty(), "Result should have benchmark_name");
        assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0,
            "Accuracy should be between 0 and 1");
        assert!(result.sample_count > 0, "Result should have samples");
    }

    println!("✅ Report fields validated");
}

/// Test HTML report generation
#[tokio::test]
async fn test_html_report_generation() {
    let report = create_test_evaluation_report();

    // Generate HTML
    let html = generate_html_report(&report)
        .expect("Should generate HTML");

    // Verify HTML structure
    assert!(html.contains("<!DOCTYPE html>"), "HTML should have DOCTYPE");
    assert!(html.contains("<html"), "HTML should have html tag");
    assert!(html.contains("<head>"), "HTML should have head");
    assert!(html.contains("<body>"), "HTML should have body");

    // Verify content
    assert!(html.contains(&report.model_name), "HTML should contain model name");
    assert!(html.contains(&report.provider_name), "HTML should contain provider name");
    assert!(html.contains("Chart.js"), "HTML should include Chart.js");

    // Verify overall score is included
    let score_str = format!("{:.1}", report.overall_score * 100.0);
    assert!(html.contains(&score_str), "HTML should contain overall score");

    println!("✅ HTML report generation verified");
}

/// Test HTML report with custom configuration
#[tokio::test]
async fn test_html_report_custom_config() {
    let report = create_test_evaluation_report();

    let config = HtmlReportConfig {
        title: "Custom Test Report".to_string(),
        include_charts: true,
        theme: "dark".to_string(),
        custom_css: Some("body { color: red; }".to_string()),
    };

    // Generate HTML with config
    let html = models_core::generate_html_report_with_config(&report, config)
        .expect("Should generate HTML with config");

    // Verify custom title
    assert!(html.contains("Custom Test Report"), "HTML should contain custom title");

    // Verify dark theme
    assert!(html.contains("#1a1a2e"), "HTML should have dark theme styles");

    // Verify custom CSS
    assert!(html.contains("body { color: red; }"), "HTML should contain custom CSS");

    println!("✅ HTML report custom config verified");
}

/// Test HTML report from stored report
#[tokio::test]
async fn test_html_from_stored_report() {
    let report = create_test_evaluation_report();

    // Create stored report
    let stored = StoredEvaluationReport {
        id: report.id,
        model_name: report.model_name.clone(),
        provider_name: report.provider_name.clone(),
        overall_score: report.overall_score,
        benchmark_count: report.benchmark_results.len(),
        category_scores: serde_json::to_value(&report.category_scores).unwrap(),
        raw_report: serde_json::to_value(&report).unwrap(),
        created_at: report.timestamp,
    };

    // Generate HTML from stored
    let html = generate_html_from_stored(&stored)
        .expect("Should generate HTML from stored report");

    // Verify HTML contains report data
    assert!(html.contains(&stored.model_name), "HTML should contain stored model name");
    assert!(html.contains("Chart.js"), "HTML should include Chart.js");

    println!("✅ HTML from stored report verified");
}

/// Test comparison report JSON structure
#[test]
fn test_comparison_report_json_structure() {
    let report = create_test_comparison_report();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&report)
        .expect("Should serialize");

    // Verify structure
    assert!(json.contains("\"id\""), "JSON should have id");
    assert!(json.contains("\"model_results\""), "JSON should have model_results");
    assert!(json.contains("\"benchmarks\""), "JSON should have benchmarks");
    assert!(json.contains("\"benchmark_winners\""), "JSON should have benchmark_winners");
    assert!(json.contains("\"category_winners\""), "JSON should have category_winners");
    assert!(json.contains("\"statistical_tests\""), "JSON should have statistical_tests");
    assert!(json.contains("\"overall_winner\""), "JSON should have overall_winner");
    assert!(json.contains("\"timestamp\""), "JSON should have timestamp");
    assert!(json.contains("\"config\""), "JSON should have config");

    println!("✅ Comparison report JSON structure verified");
}

/// Test comparison report HTML generation
#[tokio::test]
async fn test_comparison_html_report() {
    let report = create_test_comparison_report();

    // Generate HTML
    let html = generate_comparison_html_report(&report)
        .expect("Should generate comparison HTML");

    // Verify HTML structure
    assert!(html.contains("<!DOCTYPE html>"), "HTML should have DOCTYPE");
    assert!(html.contains(&report.model_results[0].model_name),
        "HTML should contain first model name");
    assert!(html.contains(&report.overall_winner.model),
        "HTML should contain overall winner");
    assert!(html.contains("Chart.js"), "HTML should include Chart.js");

    println!("✅ Comparison HTML report verified");
}

/// Test report with empty results
#[test]
fn test_report_empty_results() {
    let report = EvaluationReport {
        id: uuid::Uuid::new_v4(),
        model_name: "empty-test".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.0,
        category_scores: vec![],
        benchmark_results: vec![],
        metrics: DetailedMetrics {
            mean_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            total_tokens: 0,
            tokens_per_second: None,
            error_count: 0,
        },
        cost_analysis: None,
        config: EvaluationConfig::default(),
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    };

    // Should serialize correctly
    let json = serde_json::to_string(&report).expect("Should serialize empty report");
    assert!(!json.is_empty(), "JSON should not be empty");

    // Should deserialize correctly
    let _: EvaluationReport = serde_json::from_str(&json).expect("Should deserialize");

    println!("✅ Empty report handling verified");
}

/// Test full evaluation report generation flow
#[tokio::test]
async fn test_full_report_generation_flow() {
    // Create provider and run evaluation
    let provider = MockProvider::for_mbpp();
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").expect("MBPP should exist");

    let config = BenchmarkConfig {
        max_samples: Some(2),
        ..Default::default()
    };

    // Run benchmark
    let bench_result = benchmark.run(&provider, config).await
        .expect("Should complete");

    // Create evaluation report from result
    let report = create_report_from_benchmark(&bench_result);

    // Generate JSON
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(!json.is_empty(), "JSON should not be empty");

    // Verify report structure
    assert_eq!(report.model_name, bench_result.model_name);
    assert_eq!(report.provider_name, bench_result.provider_name);
    assert!(!report.benchmark_results.is_empty(), "Should have benchmark results");

    // Generate HTML
    let html = generate_html_report(&report).expect("Should generate HTML");
    assert!(html.contains("<!DOCTYPE html>"), "HTML should be valid");

    println!("✅ Full report generation flow verified");
}

/// Test comparison report with statistical tests
#[test]
fn test_comparison_report_statistical_tests() {
    let mut report = create_test_comparison_report();

    // Verify statistical tests exist
    assert!(!report.statistical_tests.is_empty(), "Should have statistical tests");

    // Verify each test has required fields
    for test in &report.statistical_tests {
        assert!(!test.benchmark_id.is_empty(), "Test should have benchmark_id");
        assert!(!test.model_a.is_empty(), "Test should have model_a");
        assert!(!test.model_b.is_empty(), "Test should have model_b");
        assert!(test.score_difference >= 0.0, "Score difference should be non-negative");
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0, "P-value should be valid");
        assert!(test.confidence_level > 0.0 && test.confidence_level < 1.0,
            "Confidence level should be valid");
    }

    println!("✅ Statistical tests verified");
}

/// Test report metadata handling
#[test]
fn test_report_metadata() {
    let mut report = create_test_evaluation_report();

    // Add custom metadata
    report.metadata.insert("custom_key".to_string(), serde_json::json!("custom_value"));
    report.metadata.insert("numeric_key".to_string(), serde_json::json!(42));

    // Serialize and verify metadata is preserved
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("custom_key"), "JSON should contain custom metadata key");
    assert!(json.contains("custom_value"), "JSON should contain custom metadata value");

    // Deserialize and verify
    let deserialized: EvaluationReport = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.metadata.get("custom_key"),
        Some(&serde_json::json!("custom_value"))
    );

    println!("✅ Report metadata handling verified");
}

/// Test evaluation config serialization
#[test]
fn test_evaluation_config_serialization() {
    let config = EvaluationConfig {
        temperature: 0.7,
        max_tokens: 2048,
        num_few_shot: 5,
        max_samples_per_benchmark: Some(100),
        benchmarks: vec!["mbpp".to_string(), "long_context".to_string()],
        save_results: true,
        detailed_report: true,
    };

    let json = serde_json::to_string_pretty(&config).unwrap();

    // Verify all fields are present
    assert!(json.contains("temperature"), "JSON should have temperature");
    assert!(json.contains("0.7"), "JSON should have temperature value");
    assert!(json.contains("max_tokens"), "JSON should have max_tokens");
    assert!(json.contains("2048"), "JSON should have max_tokens value");
    assert!(json.contains("num_few_shot"), "JSON should have num_few_shot");
    assert!(json.contains("save_results"), "JSON should have save_results");
    assert!(json.contains("true"), "JSON should have boolean values");

    // Deserialize and verify
    let deserialized: EvaluationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.temperature, config.temperature);
    assert_eq!(deserialized.max_tokens, config.max_tokens);
    assert_eq!(deserialized.num_few_shot, config.num_few_shot);
    assert_eq!(deserialized.save_results, config.save_results);

    println!("✅ Evaluation config serialization verified");
}

/// Test benchmark result to report conversion
#[tokio::test]
async fn test_benchmark_result_to_report() {
    let provider = MockProvider::for_mbpp();
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").unwrap();

    let config = BenchmarkConfig {
        max_samples: Some(3),
        ..Default::default()
    };

    let bench_result = benchmark.run(&provider, config).await.unwrap();

    // Create category scores from benchmark result
    let mut category_scores = Vec::new();
    for (category, stats) in &bench_result.category_stats {
        category_scores.push(CategoryScore {
            category: category.clone(),
            score: stats.accuracy,
            sample_count: stats.total_samples,
            benchmarks: vec![bench_result.benchmark_id.clone()],
        });
    }

    // Create benchmark result summary
    let benchmark_summary = models_core::evaluation::BenchmarkResultSummary {
        benchmark_id: bench_result.benchmark_id.clone(),
        benchmark_name: bench_result.benchmark_name.clone(),
        accuracy: bench_result.statistics.accuracy,
        sample_count: bench_result.statistics.total_samples,
        correct_count: bench_result.statistics.correct_count,
        mean_latency_ms: bench_result.statistics.mean_latency_ms,
    };

    // Create evaluation report
    let report = EvaluationReport {
        id: uuid::Uuid::new_v4(),
        model_name: bench_result.model_name.clone(),
        provider_name: bench_result.provider_name.clone(),
        overall_score: bench_result.statistics.accuracy,
        category_scores,
        benchmark_results: vec![benchmark_summary],
        metrics: DetailedMetrics {
            mean_latency_ms: bench_result.statistics.mean_latency_ms,
            p95_latency_ms: bench_result.statistics.p95_latency_ms,
            total_tokens: 0,
            tokens_per_second: None,
            error_count: 0,
        },
        cost_analysis: None,
        config: EvaluationConfig::default(),
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    };

    // Verify report is valid
    assert!(!report.model_name.is_empty());
    assert_eq!(report.overall_score, bench_result.statistics.accuracy);

    // Serialize and verify
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains(&bench_result.benchmark_id), "Should contain benchmark ID");

    println!("✅ Benchmark result to report conversion verified");
}

// Helper functions

fn create_test_evaluation_report() -> EvaluationReport {
    EvaluationReport {
        id: uuid::Uuid::new_v4(),
        model_name: "test-model".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.75,
        category_scores: vec![
            CategoryScore {
                category: "coding".to_string(),
                score: 0.8,
                sample_count: 10,
                benchmarks: vec!["mbpp".to_string()],
            },
            CategoryScore {
                category: "context".to_string(),
                score: 0.7,
                sample_count: 5,
                benchmarks: vec!["long_context".to_string()],
            },
        ],
        benchmark_results: vec![
            models_core::evaluation::BenchmarkResultSummary {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                accuracy: 0.8,
                sample_count: 10,
                correct_count: 8,
                mean_latency_ms: 100.0,
            },
            models_core::evaluation::BenchmarkResultSummary {
                benchmark_id: "long_context".to_string(),
                benchmark_name: "Long Context".to_string(),
                accuracy: 0.7,
                sample_count: 5,
                correct_count: 3,
                mean_latency_ms: 150.0,
            },
        ],
        metrics: DetailedMetrics {
            mean_latency_ms: 120.0,
            p95_latency_ms: 180.0,
            total_tokens: 10000,
            tokens_per_second: Some(50.0),
            error_count: 0,
        },
        cost_analysis: Some(models_core::evaluation::CostAnalysis {
            cost_per_1k_tokens: Some(0.002),
            total_cost: Some(0.02),
            total_tokens: 10000,
        }),
        config: EvaluationConfig {
            temperature: 0.0,
            max_tokens: 1024,
            num_few_shot: 0,
            max_samples_per_benchmark: Some(10),
            benchmarks: vec!["mbpp".to_string(), "long_context".to_string()],
            save_results: true,
            detailed_report: true,
        },
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    }
}

fn create_test_comparison_report() -> ComparisonReport {
    let config = ComparisonConfig {
        models: vec![
            ComparisonModelConfig {
                provider: "ollama".to_string(),
                model: "model_a".to_string(),
                base_url: None,
                api_key: None,
            },
            ComparisonModelConfig {
                provider: "ollama".to_string(),
                model: "model_b".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(10),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    ComparisonReport {
        id: uuid::Uuid::new_v4(),
        model_results: vec![
            ModelComparisonResult {
                model_name: "model_a".to_string(),
                provider_name: "ollama".to_string(),
                overall_score: 0.8,
                category_scores: vec![
                    ModelCategoryScore {
                        category: "coding".to_string(),
                        score: 0.8,
                        benchmark_count: 1,
                        benchmark_scores: [("mbpp".to_string(), 0.8)].into_iter().collect(),
                    },
                ],
                benchmark_results: vec![
                    ModelBenchmarkResult {
                        benchmark_id: "mbpp".to_string(),
                        benchmark_name: "MBPP".to_string(),
                        category: "coding".to_string(),
                        accuracy: 0.8,
                        sample_count: 10,
                        mean_latency_ms: 100.0,
                        std_dev: None,
                    },
                ],
                rank: 1,
                total_samples: 10,
                avg_latency_ms: 100.0,
            },
            ModelComparisonResult {
                model_name: "model_b".to_string(),
                provider_name: "ollama".to_string(),
                overall_score: 0.7,
                category_scores: vec![
                    ModelCategoryScore {
                        category: "coding".to_string(),
                        score: 0.7,
                        benchmark_count: 1,
                        benchmark_scores: [("mbpp".to_string(), 0.7)].into_iter().collect(),
                    },
                ],
                benchmark_results: vec![
                    ModelBenchmarkResult {
                        benchmark_id: "mbpp".to_string(),
                        benchmark_name: "MBPP".to_string(),
                        category: "coding".to_string(),
                        accuracy: 0.7,
                        sample_count: 10,
                        mean_latency_ms: 120.0,
                        std_dev: None,
                    },
                ],
                rank: 2,
                total_samples: 10,
                avg_latency_ms: 120.0,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        benchmark_winners: vec![
            models_core::evaluation::BenchmarkWinner {
                benchmark_id: "mbpp".to_string(),
                model: "model_a".to_string(),
                score: 0.8,
                margin: 0.1,
                is_significant: true,
            },
        ],
        category_winners: [("coding".to_string(), "model_a".to_string())].into_iter().collect(),
        statistical_tests: vec![
            models_core::evaluation::StatisticalTest {
                benchmark_id: "mbpp".to_string(),
                model_a: "model_a".to_string(),
                model_b: "model_b".to_string(),
                score_difference: 0.1,
                p_value: 0.02,
                is_significant: true,
                confidence_level: 0.95,
                winner: Some("model_a".to_string()),
            },
        ],
        overall_winner: models_core::evaluation::WinnerInfo {
            model: "model_a".to_string(),
            provider: "ollama".to_string(),
            overall_score: 0.8,
            benchmarks_won: 1,
            categories_won: 1,
            margin: 0.1,
            reasoning: "model_a achieved higher score".to_string(),
        },
        timestamp: Utc::now(),
        config,
    }
}

fn create_report_from_benchmark(bench_result: &BenchmarkResult) -> EvaluationReport {
    let category_scores: Vec<CategoryScore> = bench_result.category_stats.iter()
        .map(|(cat, stats)| CategoryScore {
            category: cat.clone(),
            score: stats.accuracy,
            sample_count: stats.total_samples,
            benchmarks: vec![bench_result.benchmark_id.clone()],
        })
        .collect();

    EvaluationReport {
        id: uuid::Uuid::new_v4(),
        model_name: bench_result.model_name.clone(),
        provider_name: bench_result.provider_name.clone(),
        overall_score: bench_result.statistics.accuracy,
        category_scores,
        benchmark_results: vec![
            models_core::evaluation::BenchmarkResultSummary {
                benchmark_id: bench_result.benchmark_id.clone(),
                benchmark_name: bench_result.benchmark_name.clone(),
                accuracy: bench_result.statistics.accuracy,
                sample_count: bench_result.statistics.total_samples,
                correct_count: bench_result.statistics.correct_count,
                mean_latency_ms: bench_result.statistics.mean_latency_ms,
            },
        ],
        metrics: DetailedMetrics {
            mean_latency_ms: bench_result.statistics.mean_latency_ms,
            p95_latency_ms: bench_result.statistics.p95_latency_ms,
            total_tokens: bench_result.statistics.total_tokens,
            tokens_per_second: bench_result.statistics.tokens_per_second,
            error_count: 0,
        },
        cost_analysis: None,
        config: EvaluationConfig::default(),
        timestamp: Utc::now(),
        metadata: bench_result.metadata.clone(),
    }
}

