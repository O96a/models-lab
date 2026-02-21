//! Benchmark registry

use std::collections::HashMap;

use crate::{Benchmark, BenchmarkCategory};

/// Registry for benchmarks
pub struct BenchmarkRegistry {
    benchmarks: HashMap<String, Box<dyn Benchmark>>,
}

impl BenchmarkRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            benchmarks: HashMap::new(),
        }
    }

    /// Create a registry with all built-in benchmarks
    pub fn with_builtin() -> Self {
        let mut registry = Self::new();
        registry.register_all_builtin();
        registry
    }

    /// Register a benchmark
    pub fn register<B: Benchmark + 'static>(&mut self, benchmark: B) {
        self.benchmarks
            .insert(benchmark.id().to_string(), Box::new(benchmark));
    }

    /// Register all built-in benchmarks
    pub fn register_all_builtin(&mut self) {
        self.register(crate::reasoning::mmlu::MMLUBenchmark::new());
    }

    /// Get a benchmark by ID
    pub fn get(&self, id: &str) -> Option<&dyn Benchmark> {
        self.benchmarks.get(id).map(|b| b.as_ref())
    }

    /// List all benchmark IDs
    pub fn list(&self) -> Vec<&str> {
        self.benchmarks.keys().map(|s| s.as_str()).collect()
    }

    /// List benchmarks by category
    pub fn list_by_category(&self, category: BenchmarkCategory) -> Vec<&dyn Benchmark> {
        self.benchmarks
            .values()
            .filter(|b| b.category() == category)
            .map(|b| b.as_ref())
            .collect()
    }

    /// Check if a benchmark exists
    pub fn contains(&self, id: &str) -> bool {
        self.benchmarks.contains_key(id)
    }

    /// Get the number of registered benchmarks
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }
}

impl Default for BenchmarkRegistry {
    fn default() -> Self {
        Self::with_builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = BenchmarkRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_builtin_registry() {
        let registry = BenchmarkRegistry::with_builtin();
        assert!(registry.contains("mmlu"));
        assert!(!registry.is_empty());
    }
}
