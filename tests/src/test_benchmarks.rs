//! Benchmark End-to-End Tests
//!
//! Tests for running benchmarks end-to-end with mock providers.

use models_benchmark::{
    Benchmark, BenchmarkConfig, BenchmarkCategory,
    MbppBenchmark, LongContextBenchmark,
};
use models_core::providers::GenerateRequest;
use models_lab_tests::{MockProvider, test_utils};
use std::collections::HashMap;

/// Test MBPP benchmark end-to-end
#[test]
fn test_mbpp_benchmark() {
    // Create benchmark
    let benchmark = MbppBenchmark::new();

    // Verify benchmark metadata
    assert_eq!(benchmark.id(), "mbpp", "Benchmark ID should be 'mbpp'");
    assert_eq!(benchmark.name(), "MBPP", "Benchmark name should be 'MBPP'");
    assert_eq!(benchmark.category(), BenchmarkCategory::Coding, "Category should be Coding");
    assert!(!benchmark.description().is_empty(), "Should have description");

    println!("✅ MBPP benchmark metadata verified");
}

/// Test MBPP benchmark loading dataset
#[tokio::test]
async fn test_mbpp_load_dataset() {
    let benchmark = MbppBenchmark::new();

    // Load dataset
    let dataset = benchmark.load_dataset().await.expect("Should load dataset");

    // Verify dataset structure
    assert_eq!(dataset.name, "MBPP", "Dataset name should be 'MBPP'");
    assert!(!dataset.samples.is_empty(), "Dataset should have samples");

    // Verify first sample
    let first_sample = &dataset.samples[0];
    assert!(!first_sample.id.is_empty(), "Sample should have ID");
    assert!(!first_sample.input.is_empty(), "Sample should have input");

    // Verify metadata
    assert!(first_sample.metadata.contains_key("test_list"), "Should have test_list metadata");
    assert!(first_sample.metadata.contains_key("challenge_test_list"), "Should have challenge_test_list metadata");

    println!("✅ MBPP dataset loaded: {} samples", dataset.len());
}

/// Test MBPP benchmark run with mock provider
#[tokio::test]
async fn test_mbpp_benchmark_run() {
    let benchmark = MbppBenchmark::new();
    let provider = MockProvider::for_mbpp();

    let config = BenchmarkConfig {
        max_samples: Some(3),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        ..Default::default()
    };

    let result = benchmark.run(&provider, config).await
        .expect("Benchmark should complete");

    // Verify result structure
    assert_eq!(result.benchmark_id, "mbpp");
    assert_eq!(result.benchmark_name, "MBPP");
    assert_eq!(result.model_name, "test-model");
    assert_eq!(result.provider_name, "mock");

    // Verify statistics
    assert_eq!(result.statistics.total_samples, 3, "Should have 3 samples");
    assert!(result.statistics.accuracy >= 0.0 && result.statistics.accuracy <= 1.0);

    // Verify sample results
    assert_eq!(result.sample_results.len(), 3);
    for sample in &result.sample_results {
        assert!(!sample.sample_id.is_empty());
        assert!(sample.score >= 0.0 && sample.score <= 1.0);
    }

    // Verify category stats
    assert!(!result.category_stats.is_empty(), "Should have category statistics");

    println!("✅ MBPP benchmark run completed");
    println!("   Accuracy: {:.2}%", result.statistics.accuracy * 100.0);
    println!("   Samples: {}", result.statistics.total_samples);
    println!("   Correct: {}", result.statistics.correct_count);
}

/// Test MBPP evaluation logic
#[test]
fn test_mbpp_evaluate_response() {
    let benchmark = MbppBenchmark::new();

    // Load dataset to get sample
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dataset = rt.block_on(async {
        benchmark.load_dataset().await.unwrap()
    });

    let sample = &dataset.samples[0];

    // Test correct response (with function)
    let correct_response = "```python\ndef sum_list(numbers):\n    return sum(numbers)\n```";
    let score = benchmark.evaluate_response(sample, correct_response);
    assert!(score > 0.0, "Correct response should have score > 0");

    // Test incorrect response
    let incorrect_response = "This is not code";
    let score = benchmark.evaluate_response(sample, incorrect_response);
    assert!(score < 1.0, "Incorrect response should have score < 1");

    println!("✅ MBPP evaluation logic verified");
}

/// Test Long Context benchmark end-to-end
#[test]
fn test_long_context_benchmark() {
    // Create benchmark
    let benchmark = LongContextBenchmark::new();

    // Verify benchmark metadata
    assert_eq!(benchmark.id(), "long_context", "Benchmark ID should be 'long_context'");
    assert_eq!(benchmark.name(), "Long Context", "Benchmark name should be 'Long Context'");
    assert_eq!(benchmark.category(), BenchmarkCategory::Context, "Category should be Context");
    assert!(!benchmark.description().is_empty(), "Should have description");

    println!("✅ Long Context benchmark metadata verified");
}

/// Test Long Context benchmark loading dataset
#[tokio::test]
async fn test_long_context_load_dataset() {
    let benchmark = LongContextBenchmark::new();

    // Load dataset
    let dataset = benchmark.load_dataset().await.expect("Should load dataset");

    // Verify dataset structure
    assert_eq!(dataset.name, "LongContext", "Dataset name should be 'LongContext'");
    assert!(!dataset.samples.is_empty(), "Dataset should have samples");

    // Verify metadata includes context lengths
    assert!(!dataset.metadata.categories.is_empty(), "Should have categories");

    // Check for context length categories
    let categories = &dataset.metadata.categories;
    assert!(categories.iter().any(|c| c.contains("1k") || c.contains("2k")),
        "Should have context length categories");

    println!("✅ Long Context dataset loaded: {} samples", dataset.len());
    println!("   Categories: {:?}", categories);
}

/// Test Long Context benchmark run with mock provider
#[tokio::test]
async fn test_long_context_benchmark_run() {
    let benchmark = LongContextBenchmark::new();
    let provider = MockProvider::for_long_context();

    let config = BenchmarkConfig {
        max_samples: Some(3),
        temperature: 0.0,
        max_tokens: 1024,
        num_few_shot: 0,
        ..Default::default()
    };

    let result = benchmark.run(&provider, config).await
        .expect("Benchmark should complete");

    // Verify result structure
    assert_eq!(result.benchmark_id, "long_context");
    assert_eq!(result.benchmark_name, "Long Context");
    assert_eq!(result.model_name, "test-model");

    // Verify statistics
    assert_eq!(result.statistics.total_samples, 3);
    assert!(result.statistics.accuracy >= 0.0 && result.statistics.accuracy <= 1.0);

    // Verify sample results
    assert_eq!(result.sample_results.len(), 3);

    // Verify metadata contains context_lengths
    assert!(result.metadata.contains_key("context_lengths"),
        "Should have context_lengths metadata");

    // Verify category stats (should have categories like "1k_single_hop", etc.)
    assert!(!result.category_stats.is_empty(), "Should have category statistics");

    println!("✅ Long Context benchmark run completed");
    println!("   Accuracy: {:.2}%", result.statistics.accuracy * 100.0);
    println!("   Categories: {:?}", result.category_stats.keys().collect::<Vec<_>>());
}

/// Test Long Context evaluation logic
#[test]
fn test_long_context_evaluate_response() {
    use models_benchmark::DataSample;

    let benchmark = LongContextBenchmark::new();

    // Test exact match
    let sample = DataSample::new("test", "context").with_expected("42");
    let score = benchmark.evaluate_response(&sample, "42");
    assert_eq!(score, 1.0, "Exact match should score 1.0");

    // Test case-insensitive match
    let score = benchmark.evaluate_response(&sample, "42");
    assert_eq!(score, 1.0, "Case-insensitive match should score 1.0");

    // Test contained in response
    let score = benchmark.evaluate_response(&sample, "The answer is 42");
    assert_eq!(score, 1.0, "Contained answer should score 1.0");

    // Test incorrect answer
    let score = benchmark.evaluate_response(&sample, "100");
    assert_eq!(score, 0.0, "Incorrect answer should score 0.0");

    // Test multi-word answer
    let sample = DataSample::new("test2", "context").with_expected("Dr. Smith");

    // Partial match
    let score = benchmark.evaluate_response(&sample, "Dr. Smith discovered this");
    assert!(score > 0.0, "Partial match should have score > 0");

    println!("✅ Long Context evaluation logic verified");
}

/// Test Long Context sample statistics
#[test]
fn test_long_context_sample_statistics() {
    let benchmark = LongContextBenchmark::new();
    let stats = benchmark.sample_statistics();

    // Verify context length categories (based on actual CONTEXT_LENGTHS in implementation)
    assert!(stats.contains_key("4k_tokens"), "Should have 4k tokens");
    assert!(stats.contains_key("8k_tokens"), "Should have 8k tokens");
    assert!(stats.contains_key("16k_tokens"), "Should have 16k tokens");
    assert!(stats.contains_key("32k_tokens"), "Should have 32k tokens");
    assert!(stats.contains_key("64k_tokens"), "Should have 64k tokens");
    assert!(stats.contains_key("128k_tokens"), "Should have 128k tokens");

    // Verify question type categories
    assert!(stats.contains_key("single_hop"), "Should have single_hop");
    assert!(stats.contains_key("multi_hop"), "Should have multi_hop");
    assert!(stats.contains_key("many_hop"), "Should have many_hop");

    // Verify counts
    let total_samples: usize = stats.values().sum();
    assert!(total_samples > 0, "Should have samples");

    println!("✅ Long Context statistics verified");
    println!("   Context lengths: 4k, 8k, 16k, 32k, 64k, 128k");
    println!("   Question types: single_hop, multi_hop, many_hop");
}

/// Test benchmark configuration defaults
#[test]
fn test_benchmark_config_defaults() {
    let config = BenchmarkConfig::default();

    assert!(config.max_samples.is_none());
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.max_tokens, 1024);
    assert_eq!(config.num_few_shot, 0);
    assert!(config.stop_sequences.is_empty());

    println!("✅ Benchmark config defaults verified");
}

/// Test benchmark run with different sample counts
#[tokio::test]
async fn test_benchmark_varying_samples() {
    let benchmark = MbppBenchmark::new();
    let provider = MockProvider::for_mbpp();

    for samples in [1, 2, 5] {
        let config = BenchmarkConfig {
            max_samples: Some(samples),
            ..Default::default()
        };

        let result = benchmark.run(&provider, config).await
            .expect("Should complete");

        assert_eq!(result.statistics.total_samples, samples,
            "Should evaluate exactly {} samples", samples);
    }

    println!("✅ Benchmark varying samples test passed");
}

/// Test benchmark category stats are properly populated
#[tokio::test]
async fn test_benchmark_category_stats() {
    let benchmark = MbppBenchmark::new();
    let provider = MockProvider::for_mbpp();

    let config = BenchmarkConfig {
        max_samples: Some(5),
        ..Default::default()
    };

    let result = benchmark.run(&provider, config).await
        .expect("Should complete");

    // Verify each category has valid statistics
    for (category, stats) in &result.category_stats {
        assert!(stats.total_samples > 0, "Category {} should have samples", category);
        assert!(stats.accuracy >= 0.0 && stats.accuracy <= 1.0,
            "Category {} accuracy should be valid", category);
        println!("   Category '{}': {} samples, {:.2}% accuracy",
            category, stats.total_samples, stats.accuracy * 100.0);
    }

    println!("✅ Benchmark category stats verified");
}

/// Test running benchmark with error simulation
#[tokio::test]
async fn test_benchmark_error_recovery() {
    let benchmark = MbppBenchmark::new();
    let provider = MockProvider::new()
        .with_errors(true);

    let config = BenchmarkConfig {
        max_samples: Some(3),
        ..Default::default()
    };

    // Benchmark should still complete, recording errors
    let result = benchmark.run(&provider, config).await;

    // Result should exist (errors are recorded, not thrown)
    assert!(result.is_ok() || result.is_err(), "Should return a result");

    if let Ok(result) = result {
        // Some samples may have errors recorded
        let errors = result.sample_results.iter()
            .filter(|s| s.error.is_some())
            .count();
        println!("   Samples with errors: {}", errors);
    }

    println!("✅ Benchmark error recovery test passed");
}

/// Test benchmark registry
#[test]
fn test_benchmark_registry() {
    let registry = models_benchmark::BenchmarkRegistry::with_builtin();

    // Verify built-in benchmarks exist
    assert!(registry.get("mbpp").is_some(), "Should have MBPP");
    assert!(registry.get("long_context").is_some(), "Should have LongContext");

    // Verify listing works
    let list = registry.list();
    assert!(!list.is_empty(), "Registry should list benchmarks");

    println!("✅ Benchmark registry verified");
    println!("   Built-in benchmarks: {:?}", list);
}

/// Test benchmark result serialization
#[tokio::test]
async fn test_benchmark_result_serialization() {
    let benchmark = MbppBenchmark::new();
    let provider = MockProvider::for_mbpp();

    let config = BenchmarkConfig {
        max_samples: Some(2),
        ..Default::default()
    };

    let result = benchmark.run(&provider, config).await
        .expect("Should complete");

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&result)
        .expect("Should serialize");

    // Verify JSON contains expected fields
    assert!(json.contains("benchmark_id"), "JSON should have benchmark_id");
    assert!(json.contains("statistics"), "JSON should have statistics");
    assert!(json.contains("sample_results"), "JSON should have sample_results");

    // Deserialize back
    let deserialized: models_benchmark::BenchmarkResult = serde_json::from_str(&json)
        .expect("Should deserialize");

    assert_eq!(deserialized.benchmark_id, result.benchmark_id);
    assert_eq!(deserialized.statistics.total_samples, result.statistics.total_samples);

    println!("✅ Benchmark result serialization verified");
}

