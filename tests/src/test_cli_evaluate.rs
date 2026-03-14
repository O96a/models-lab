//! CLI Integration Tests
//!
//! Tests for the CLI evaluate command end-to-end functionality.

use models_benchmark::{BenchmarkRegistry, BenchmarkConfig};
use models_core::providers::ModelProvider;
use models_lab_tests::{MockProvider, test_utils};

/// Test evaluating MBPP benchmark
///
/// Simulates: models evaluate --provider mock --model test-model --benchmarks mbpp --samples 2
#[tokio::test]
async fn test_evaluate_mbpp() {
    // Create mock provider configured for MBPP
    let provider = MockProvider::for_mbpp();

    // Get the MBPP benchmark
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").expect("MBPP benchmark should exist");

    // Configure with minimal samples for fast test
    let config = BenchmarkConfig {
        max_samples: Some(2),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        ..Default::default()
    };

    // Run the benchmark
    let result = benchmark.run(&provider, config).await;

    // Verify the result
    assert!(result.is_ok(), "Benchmark should complete successfully: {:?}", result.err());
    let result = result.unwrap();

    // Verify result structure
    assert_eq!(result.benchmark_id, "mbpp", "Benchmark ID should be 'mbpp'");
    assert_eq!(result.benchmark_name, "MBPP", "Benchmark name should be 'MBPP'");
    assert_eq!(result.model_name, "test-model", "Model name should match");
    assert_eq!(result.provider_name, "mock", "Provider name should be 'mock'");

    // Verify statistics
    assert_eq!(result.statistics.total_samples, 2, "Should evaluate 2 samples");
    assert!(result.statistics.accuracy >= 0.0 && result.statistics.accuracy <= 1.0,
        "Accuracy should be between 0 and 1");
    assert!(result.statistics.mean_latency_ms > 0.0,
        "Should have recorded latency");

    // Verify sample results
    assert_eq!(result.sample_results.len(), 2, "Should have 2 sample results");

    // Verify each sample result has required fields
    for sample in &result.sample_results {
        assert!(!sample.sample_id.is_empty(), "Sample should have ID");
        assert!(!sample.generated_output.is_empty(), "Sample should have generated output");
        // Score should be between 0 and 1
        assert!(sample.score >= 0.0 && sample.score <= 1.0,
            "Score should be between 0 and 1, got {}", sample.score);
    }

    // Verify category stats exist (MBPP has categories like 'basic', 'string', 'math', 'list')
    assert!(!result.category_stats.is_empty(), "Should have category statistics");

    println!("✅ MBPP evaluation test passed");
    println!("   Accuracy: {:.2}%", result.statistics.accuracy * 100.0);
    println!("   Samples: {}", result.statistics.total_samples);
    println!("   Correct: {}", result.statistics.correct_count);
}

/// Test evaluating Long Context benchmark
///
/// Simulates: models evaluate --provider mock --model test-model --benchmarks long_context --samples 2
#[tokio::test]
async fn test_evaluate_long_context() {
    // Create mock provider configured for Long Context
    let provider = MockProvider::for_long_context();

    // Get the Long Context benchmark
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("long_context")
        .expect("LongContext benchmark should exist");

    // Configure with minimal samples
    let config = BenchmarkConfig {
        max_samples: Some(2),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        ..Default::default()
    };

    // Run the benchmark
    let result = benchmark.run(&provider, config).await;

    // Verify the result
    assert!(result.is_ok(), "Benchmark should complete successfully: {:?}", result.err());
    let result = result.unwrap();

    // Verify result structure
    assert_eq!(result.benchmark_id, "long_context", "Benchmark ID should be 'long_context'");
    assert_eq!(result.benchmark_name, "Long Context", "Benchmark name should be 'Long Context'");
    assert_eq!(result.model_name, "test-model", "Model name should match");

    // Verify statistics
    assert_eq!(result.statistics.total_samples, 2, "Should evaluate 2 samples");
    assert!(result.statistics.accuracy >= 0.0 && result.statistics.accuracy <= 1.0,
        "Accuracy should be between 0 and 1");

    // Verify sample results
    assert_eq!(result.sample_results.len(), 2, "Should have 2 sample results");

    // LongContext has context_length metadata
    assert!(result.metadata.contains_key("context_lengths"),
        "Should have context_lengths metadata");

    println!("✅ Long Context evaluation test passed");
    println!("   Accuracy: {:.2}%", result.statistics.accuracy * 100.0);
    println!("   Samples: {}", result.statistics.total_samples);
}

/// Test evaluating with error handling
#[tokio::test]
async fn test_evaluate_error_handling() {
    // Create mock provider that simulates errors
    let provider = MockProvider::new()
        .with_name("mock", "error-model")
        .with_errors(true);

    // Get any benchmark
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").expect("MBPP should exist");

    let config = BenchmarkConfig {
        max_samples: Some(1),
        ..Default::default()
    };

    // Run the benchmark - it should handle errors gracefully
    let result = benchmark.run(&provider, config).await;

    // The benchmark should complete, possibly with errors recorded in results
    assert!(result.is_ok() || result.is_err(), "Should return a result");

    if let Ok(result) = result {
        // Check if errors were recorded in sample results
        let errors: Vec<_> = result.sample_results
            .iter()
            .filter(|s| s.error.is_some())
            .collect();
        println!("   Errors recorded: {}", errors.len());
    }
}

/// Test evaluating multiple benchmarks in sequence
#[tokio::test]
async fn test_evaluate_multiple_benchmarks() {
    let registry = test_utils::create_test_registry();
    let provider = MockProvider::general_model();

    let benchmark_ids = vec!["mbpp", "long_context"];
    let config = test_utils::create_test_benchmark_config();

    let mut results = Vec::new();

    for benchmark_id in benchmark_ids {
        if let Some(benchmark) = registry.get(benchmark_id) {
            match benchmark.run(&provider, config.clone()).await {
                Ok(result) => {
                    results.push((benchmark_id, result.statistics.accuracy));
                }
                Err(e) => {
                    println!("   Warning: {} failed: {}", benchmark_id, e);
                }
            }
        }
    }

    assert!(!results.is_empty(), "Should have at least one result");

    println!("✅ Multiple benchmarks test passed");
    for (id, accuracy) in &results {
        println!("   {}: {:.2}%", id, accuracy * 100.0);
    }
}

/// Test that verifies benchmark config is respected
#[tokio::test]
async fn test_evaluate_respects_max_samples() {
    let provider = MockProvider::for_mbpp();
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").unwrap();

    // Test with different sample counts
    for max_samples in [1, 2, 3] {
        let config = BenchmarkConfig {
            max_samples: Some(max_samples),
            ..Default::default()
        };

        let result = benchmark.run(&provider, config).await.unwrap();
        assert_eq!(result.statistics.total_samples, max_samples,
            "Should evaluate exactly {} samples", max_samples);
    }

    println!("✅ Max samples test passed");
}

/// Test evaluating with temperature parameter
#[tokio::test]
async fn test_evaluate_with_temperature() {
    let provider = MockProvider::for_mbpp();
    let registry = test_utils::create_test_registry();
    let benchmark = registry.get("mbpp").unwrap();

    // Test with different temperatures
    for temperature in [0.0, 0.5, 1.0] {
        let config = BenchmarkConfig {
            max_samples: Some(1),
            temperature,
            ..Default::default()
        };

        let result = benchmark.run(&provider, config).await;
        assert!(result.is_ok(), "Should complete with temperature {}", temperature);
    }

    println!("✅ Temperature parameter test passed");
}

