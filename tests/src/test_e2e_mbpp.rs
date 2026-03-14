//! Model Comparison Integration Test
//!
//! This test verifies the model comparison functionality including
//! statistical analysis and report generation.

use models_benchmark::BenchmarkRegistry;
use models_core::{
    ComparisonConfig, ComparisonReportBuilder, ModelBenchmarkResult, ModelCategoryScore,
    ModelComparisonResult, ModelConfig, BenchmarkWinner, StatisticalTest, WinnerInfo,
};
use std::collections::HashMap;

use crate::mock_provider::{MockProvider, MockProviderBuilder};

/// Test basic model comparison between two models
#[tokio::test]
async fn test_model_comparison_basic() {
    // Create two different mock providers
    let provider1 = MockProvider::perfect_coder();
    let provider2 = MockProvider::poor_coder();

    // Create comparison config
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "perfect-coder".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "poor-coder".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(3),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    // Build comparison report
    let mut report_builder = ComparisonReportBuilder::new(config);

    // Simulate results for model 1 (perfect coder)
    let result1 = ModelComparisonResult {
        model_name: "perfect-coder".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.95,
        category_scores: vec![
            ModelCategoryScore {
                category: "coding".to_string(),
                score: 0.95,
                benchmark_count: 1,
                benchmark_scores: [("mbpp".to_string(), 0.95)].into_iter().collect(),
            },
        ],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.95,
                sample_count: 3,
                mean_latency_ms: 100.0,
                std_dev: None,
            },
        ],
        rank: 1,
        total_samples: 3,
        avg_latency_ms: 100.0,
    };

    // Simulate results for model 2 (poor coder)
    let result2 = ModelComparisonResult {
        model_name: "poor-coder".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.15,
        category_scores: vec![
            ModelCategoryScore {
                category: "coding".to_string(),
                score: 0.15,
                benchmark_count: 1,
                benchmark_scores: [("mbpp".to_string(), 0.15)].into_iter().collect(),
            },
        ],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.15,
                sample_count: 3,
                mean_latency_ms: 50.0,
                std_dev: None,
            },
        ],
        rank: 2,
        total_samples: 3,
        avg_latency_ms: 50.0,
    };

    report_builder = report_builder.add_result(result1);
    report_builder = report_builder.add_result(result2);

    let report = report_builder.build();

    // Verify report structure
    assert_eq!(report.models.len(), 2);
    assert_eq!(report.models[0].model_name, "perfect-coder");
    assert_eq!(report.models[1].model_name, "poor-coder");

    // Verify overall scores
    assert!(report.models[0].overall_score > report.models[1].overall_score,
        "Perfect coder should have higher score than poor coder");

    // Verify winner
    assert!(report.winner.is_some(), "Should have a winner");
    let winner = report.winner.unwrap();
    assert_eq!(winner.model_name, "perfect-coder");
}

/// Test statistical analysis in comparison reports
#[tokio::test]
async fn test_comparison_statistical_analysis() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-b".to_string(),
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

    let mut report_builder = ComparisonReportBuilder::new(config);

    // Add results with different accuracies
    let result_a = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.85,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.85,
                sample_count: 10,
                mean_latency_ms: 100.0,
                std_dev: Some(10.0),
            },
        ],
        rank: 1,
        total_samples: 10,
        avg_latency_ms: 100.0,
    };

    let result_b = ModelComparisonResult {
        model_name: "model-b".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.75,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.75,
                sample_count: 10,
                mean_latency_ms: 150.0,
                std_dev: Some(15.0),
            },
        ],
        rank: 2,
        total_samples: 10,
        avg_latency_ms: 150.0,
    };

    report_builder = report_builder.add_result(result_a);
    report_builder = report_builder.add_result(result_b);

    let report = report_builder.build();

    // Verify statistical analysis
    assert!(report.statistical_tests.is_some(), "Should have statistical tests");
    let tests = report.statistical_tests.unwrap();
    assert!(!tests.is_empty(), "Should have at least one statistical test");

    // Verify winner info
    assert!(report.winner.is_some(), "Should have a winner");
    let winner = report.winner.unwrap();
    assert_eq!(winner.model_name, "model-a");
    assert!(winner.margin > 0.0, "Winner should have positive margin");
}

/// Test comparison with three models
#[tokio::test]
async fn test_comparison_three_models() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-b".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-c".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    // Add results for three models with different scores
    let scores = vec![0.9, 0.7, 0.8];
    let model_names = vec!["model-a", "model-b", "model-c"];

    for (i, (model_name, score)) in model_names.iter().zip(scores.iter()).enumerate() {
        let result = ModelComparisonResult {
            model_name: model_name.to_string(),
            provider_name: "mock".to_string(),
            overall_score: *score,
            category_scores: vec![],
            benchmark_results: vec![
                ModelBenchmarkResult {
                    benchmark_id: "mbpp".to_string(),
                    benchmark_name: "MBPP".to_string(),
                    category: "coding".to_string(),
                    accuracy: *score,
                    sample_count: 5,
                    mean_latency_ms: 100.0 + (i as f64 * 10.0),
                    std_dev: None,
                },
            ],
            rank: 0, // Will be calculated
            total_samples: 5,
            avg_latency_ms: 100.0 + (i as f64 * 10.0),
        };
        report_builder = report_builder.add_result(result);
    }

    let report = report_builder.build();

    // Verify all three models are in the report
    assert_eq!(report.models.len(), 3);

    // Verify ranking (model-a should be first with score 0.9)
    assert_eq!(report.models[0].model_name, "model-a");
    assert_eq!(report.models[0].rank, 1);

    // Verify winner
    assert!(report.winner.is_some());
    let winner = report.winner.unwrap();
    assert_eq!(winner.model_name, "model-a");
}

/// Test comparison report serialization
#[tokio::test]
async fn test_comparison_report_serialization() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(3),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    let result = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.85,
        category_scores: vec![
            ModelCategoryScore {
                category: "coding".to_string(),
                score: 0.85,
                benchmark_count: 1,
                benchmark_scores: [("mbpp".to_string(), 0.85)].into_iter().collect(),
            },
        ],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.85,
                sample_count: 3,
                mean_latency_ms: 100.0,
                std_dev: Some(10.0),
            },
        ],
        rank: 1,
        total_samples: 3,
        avg_latency_ms: 100.0,
    };

    report_builder = report_builder.add_result(result);
    let report = report_builder.build();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&report).expect("Should serialize to JSON");
    assert!(!json.is_empty());

    // Verify JSON contains expected fields
    assert!(json.contains("models"));
    assert!(json.contains("model-a"));
    assert!(json.contains("overall_score"));
    assert!(json.contains("benchmark_results"));
}

/// Test comparison with latency analysis
#[tokio::test]
async fn test_comparison_latency_analysis() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "fast-model".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "slow-model".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    // Fast model: high accuracy, low latency
    let fast_result = ModelComparisonResult {
        model_name: "fast-model".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.9,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.9,
                sample_count: 5,
                mean_latency_ms: 50.0,
                std_dev: Some(5.0),
            },
        ],
        rank: 1,
        total_samples: 5,
        avg_latency_ms: 50.0,
    };

    // Slow model: lower accuracy, high latency
    let slow_result = ModelComparisonResult {
        model_name: "slow-model".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.7,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.7,
                sample_count: 5,
                mean_latency_ms: 200.0,
                std_dev: Some(20.0),
            },
        ],
        rank: 2,
        total_samples: 5,
        avg_latency_ms: 200.0,
    };

    report_builder = report_builder.add_result(fast_result);
    report_builder = report_builder.add_result(slow_result);

    let report = report_builder.build();

    // Verify latency comparison
    let fast_model = report.models.iter().find(|m| m.model_name == "fast-model").unwrap();
    let slow_model = report.models.iter().find(|m| m.model_name == "slow-model").unwrap();

    assert!(fast_model.avg_latency_ms < slow_model.avg_latency_ms,
        "Fast model should have lower latency than slow model");

    // Verify fast model is ranked higher
    assert!(fast_model.rank < slow_model.rank,
        "Fast model should be ranked higher than slow model");
}

/// Test comparison with category scores
#[tokio::test]
async fn test_comparison_category_scores() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    let category_scores = vec![
        ModelCategoryScore {
            category: "coding".to_string(),
            score: 0.85,
            benchmark_count: 1,
            benchmark_scores: [("mbpp".to_string(), 0.85)].into_iter().collect(),
        },
        ModelCategoryScore {
            category: "reasoning".to_string(),
            score: 0.75,
            benchmark_count: 0,
            benchmark_scores: HashMap::new(),
        },
    ];

    let result = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.80,
        category_scores,
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.85,
                sample_count: 5,
                mean_latency_ms: 100.0,
                std_dev: None,
            },
        ],
        rank: 1,
        total_samples: 5,
        avg_latency_ms: 100.0,
    };

    report_builder = report_builder.add_result(result);
    let report = report_builder.build();

    // Verify category scores are preserved
    assert!(!report.models[0].category_scores.is_empty(),
        "Should have category scores");

    let coding_score = report.models[0].category_scores.iter()
        .find(|cs| cs.category == "coding")
        .expect("Should have coding category");
    assert_eq!(coding_score.score, 0.85);
}

/// Test comparison with statistical significance
#[tokio::test]
async fn test_comparison_statistical_significance() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-b".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(30),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    // Add results with significant difference
    let result_a = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.90,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.90,
                sample_count: 30,
                mean_latency_ms: 100.0,
                std_dev: Some(5.0),
            },
        ],
        rank: 1,
        total_samples: 30,
        avg_latency_ms: 100.0,
    };

    let result_b = ModelComparisonResult {
        model_name: "model-b".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.70,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.70,
                sample_count: 30,
                mean_latency_ms: 120.0,
                std_dev: Some(8.0),
            },
        ],
        rank: 2,
        total_samples: 30,
        avg_latency_ms: 120.0,
    };

    report_builder = report_builder.add_result(result_a);
    report_builder = report_builder.add_result(result_b);

    let report = report_builder.build();

    // Verify statistical tests exist
    assert!(report.statistical_tests.is_some(), "Should have statistical tests");

    // Verify winner is model-a
    assert!(report.winner.is_some());
    assert_eq!(report.winner.unwrap().model_name, "model-a");
}

/// Test comparison report JSON serialization
#[tokio::test]
async fn test_comparison_report_serialization() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    let result = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.85,
        category_scores: vec![],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.85,
                sample_count: 5,
                mean_latency_ms: 100.0,
                std_dev: None,
            },
        ],
        rank: 1,
        total_samples: 5,
        avg_latency_ms: 100.0,
    };

    report_builder = report_builder.add_result(result);
    let report = report_builder.build();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&report).expect("Should serialize to JSON");
    assert!(!json.is_empty());

    // Verify JSON contains expected fields
    assert!(json.contains("models"));
    assert!(json.contains("model-a"));
    assert!(json.contains("overall_score"));
    assert!(json.contains("0.85"));

    // Deserialize and verify
    let deserialized: models_core::ComparisonReport = serde_json::from_str(&json)
        .expect("Should deserialize from JSON");
    assert_eq!(deserialized.models.len(), 1);
    assert_eq!(deserialized.models[0].model_name, "model-a");
}

/// Test comparison with multiple benchmarks
#[tokio::test]
async fn test_comparison_multiple_benchmarks() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string(), "humaneval".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    let result = ModelComparisonResult {
        model_name: "model-a".to_string(),
        provider_name: "mock".to_string(),
        overall_score: 0.80,
        category_scores: vec![
            ModelCategoryScore {
                category: "coding".to_string(),
                score: 0.80,
                benchmark_count: 2,
                benchmark_scores: [
                    ("mbpp".to_string(), 0.85),
                    ("humaneval".to_string(), 0.75),
                ].into_iter().collect(),
            },
        ],
        benchmark_results: vec![
            ModelBenchmarkResult {
                benchmark_id: "mbpp".to_string(),
                benchmark_name: "MBPP".to_string(),
                category: "coding".to_string(),
                accuracy: 0.85,
                sample_count: 5,
                mean_latency_ms: 100.0,
                std_dev: None,
            },
            ModelBenchmarkResult {
                benchmark_id: "humaneval".to_string(),
                benchmark_name: "HumanEval".to_string(),
                category: "coding".to_string(),
                accuracy: 0.75,
                sample_count: 5,
                mean_latency_ms: 120.0,
                std_dev: None,
            },
        ],
        rank: 1,
        total_samples: 10,
        avg_latency_ms: 110.0,
    };

    report_builder = report_builder.add_result(result);
    let report = report_builder.build();

    // Verify multiple benchmarks are tracked
    assert_eq!(report.models[0].benchmark_results.len(), 2);

    // Verify category scores aggregate correctly
    let coding_score = report.models[0].category_scores.iter()
        .find(|cs| cs.category == "coding")
        .expect("Should have coding category");
    assert_eq!(coding_score.benchmark_count, 2);
}

/// Test comparison with tied scores
#[tokio::test]
async fn test_comparison_tied_scores() {
    let config = ComparisonConfig {
        models: vec![
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-a".to_string(),
                base_url: None,
                api_key: None,
            },
            ModelConfig {
                provider: "mock".to_string(),
                model: "model-b".to_string(),
                base_url: None,
                api_key: None,
            },
        ],
        benchmarks: vec!["mbpp".to_string()],
        max_samples: Some(5),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        confidence_level: 0.95,
    };

    let mut report_builder = ComparisonReportBuilder::new(config);

    // Add two models with identical scores
    for model_name in ["model-a", "model-b"] {
        let result = ModelComparisonResult {
            model_name: model_name.to_string(),
            provider_name: "mock".to_string(),
            overall_score: 0.80,
            category_scores: vec![],
            benchmark_results: vec![
                ModelBenchmarkResult {
                    benchmark_id: "mbpp".to_string(),
                    benchmark_name: "MBPP".to_string(),
                    category: "coding".to_string(),
                    accuracy: 0.80,
                    sample_count: 5,
                    mean_latency_ms: 100.0,
                    std_dev: None,
                },
            ],
            rank: 0,
            total_samples: 5,
            avg_latency_ms: 100.0,
        };
        report_builder = report_builder.add_result(result);
    }

    let report = report_builder.build();

    // With tied scores, both should have the same rank
    assert_eq!(report.models[0].rank, report.models[1].rank);

    // Winner might be None or one of them (implementation dependent)
    // Just verify the report is valid
    assert_eq!(report.models.len(), 2);
}
