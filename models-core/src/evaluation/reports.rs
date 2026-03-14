//! HTML Report Generation for Evaluation Reports
//!
//! This module provides HTML report generation capabilities using the Tera template engine.
//! It supports generating rich HTML reports with charts, tables, and visual comparisons.

use crate::evaluation::{
    ComparisonReport, EvaluationReport,
    ModelComparisonResult,
};
use crate::storage::StoredEvaluationReport;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tera::{Context, Tera};

/// Configuration for HTML report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlReportConfig {
    /// Title of the report
    pub title: String,
    /// Include Chart.js for visualization
    pub include_charts: bool,
    /// Theme (light or dark)
    pub theme: String,
    /// Custom CSS styles
    pub custom_css: Option<String>,
}

impl Default for HtmlReportConfig {
    fn default() -> Self {
        Self {
            title: "Model Evaluation Report".to_string(),
            include_charts: true,
            theme: "light".to_string(),
            custom_css: None,
        }
    }
}

/// Template data for category scores
#[derive(Debug, Clone, Serialize)]
struct CategoryScoreData {
    category: String,
    score: f64,
    score_percentage: f64,
    sample_count: usize,
    benchmarks: Vec<String>,
}

/// Template data for benchmark results
#[derive(Debug, Clone, Serialize)]
struct BenchmarkResultData {
    benchmark_id: String,
    benchmark_name: String,
    accuracy: f64,
    accuracy_percentage: f64,
    sample_count: usize,
    correct_count: usize,
    mean_latency_ms: f64,
}

/// Template data for metrics
#[derive(Debug, Clone, Serialize)]
struct MetricsData {
    mean_latency_ms: f64,
    p95_latency_ms: f64,
    total_tokens: u64,
    tokens_per_second: Option<f64>,
    error_count: usize,
}

/// Template data for cost analysis
#[derive(Debug, Clone, Serialize)]
struct CostAnalysisData {
    cost_per_1k_tokens: Option<f64>,
    total_cost: Option<f64>,
    total_tokens: u64,
    cost_formatted: String,
}

/// Report generator using Tera templates
pub struct HtmlReportGenerator {
    tera: Tera,
    config: HtmlReportConfig,
}

impl HtmlReportGenerator {
    /// Create a new report generator with default embedded template
    pub fn new() -> Result<Self> {
        let mut tera = Tera::default();

        // Add the default template
        tera.add_raw_template("report.html", DEFAULT_TEMPLATE)
            .map_err(|e| Error::Config(format!("Failed to load template: {}", e)))?;

        Ok(Self {
            tera,
            config: HtmlReportConfig::default(),
        })
    }

    /// Create a new report generator with custom configuration
    pub fn with_config(config: HtmlReportConfig) -> Result<Self> {
        let mut tera = Tera::default();

        // Add the default template
        tera.add_raw_template("report.html", DEFAULT_TEMPLATE)
            .map_err(|e| Error::Config(format!("Failed to load template: {}", e)))?;

        Ok(Self { tera, config })
    }

    /// Create a report generator with templates from a directory
    pub fn from_templates_dir(templates_dir: &str) -> Result<Self> {
        let tera = Tera::new(&format!("{}/**/*.html", templates_dir))
            .map_err(|e| Error::Config(format!("Failed to load templates: {}", e)))?;

        Ok(Self {
            tera,
            config: HtmlReportConfig::default(),
        })
    }

    /// Generate HTML report from an EvaluationReport
    pub fn generate(&self, report: &EvaluationReport) -> Result<String> {
        let context = self.build_evaluation_context(report)?;
        self.render(context)
    }

    /// Generate HTML report from a ComparisonReport
    pub fn generate_comparison(&self, report: &ComparisonReport) -> Result<String> {
        let context = self.build_comparison_context(report)?;
        self.render(context)
    }

    /// Build context for evaluation report template
    fn build_evaluation_context(&self, report: &EvaluationReport) -> Result<Context> {
        let mut context = Context::new();

        // Basic report info
        context.insert("title", &self.config.title);
        context.insert("report_id", &report.id.to_string());
        context.insert("model_name", &report.model_name);
        context.insert("provider_name", &report.provider_name);
        context.insert("overall_score", &report.overall_score);
        context.insert(
            "overall_score_percentage",
            &(report.overall_score * 100.0),
        );
        context.insert("timestamp", &report.timestamp.to_rfc3339());
        context.insert("theme", &self.config.theme);
        context.insert("include_charts", &self.config.include_charts);

        // Category scores
        let category_scores: Vec<CategoryScoreData> = report
            .category_scores
            .iter()
            .map(|cs| CategoryScoreData {
                category: cs.category.clone(),
                score: cs.score,
                score_percentage: cs.score * 100.0,
                sample_count: cs.sample_count,
                benchmarks: cs.benchmarks.clone(),
            })
            .collect();
        context.insert("category_scores", &category_scores);

        // Benchmark results
        let benchmark_results: Vec<BenchmarkResultData> = report
            .benchmark_results
            .iter()
            .map(|br| BenchmarkResultData {
                benchmark_id: br.benchmark_id.clone(),
                benchmark_name: br.benchmark_name.clone(),
                accuracy: br.accuracy,
                accuracy_percentage: br.accuracy * 100.0,
                sample_count: br.sample_count,
                correct_count: br.correct_count,
                mean_latency_ms: br.mean_latency_ms,
            })
            .collect();
        context.insert("benchmark_results", &benchmark_results);

        // Metrics
        let metrics = MetricsData {
            mean_latency_ms: report.metrics.mean_latency_ms,
            p95_latency_ms: report.metrics.p95_latency_ms,
            total_tokens: report.metrics.total_tokens,
            tokens_per_second: report.metrics.tokens_per_second,
            error_count: report.metrics.error_count,
        };
        context.insert("metrics", &metrics);

        // Cost analysis
        if let Some(cost) = &report.cost_analysis {
            let cost_data = CostAnalysisData {
                cost_per_1k_tokens: cost.cost_per_1k_tokens,
                total_cost: cost.total_cost,
                total_tokens: cost.total_tokens,
                cost_formatted: cost
                    .total_cost
                    .map(|c| format!("${:.4}", c))
                    .unwrap_or_else(|| "N/A".to_string()),
            };
            context.insert("cost_analysis", &cost_data);
            context.insert("has_cost_analysis", &true);
        } else {
            context.insert("has_cost_analysis", &false);
        }

        // Configuration
        context.insert("config", &report.config);

        // Chart data (JSON for Chart.js)
        let chart_labels: Vec<String> = report
            .category_scores
            .iter()
            .map(|cs| cs.category.clone())
            .collect();
        let chart_data: Vec<f64> = report
            .category_scores
            .iter()
            .map(|cs| cs.score * 100.0)
            .collect();
        let bench_labels: Vec<String> = report
            .benchmark_results
            .iter()
            .map(|br| br.benchmark_name.clone())
            .collect();
        let bench_data: Vec<f64> = report
            .benchmark_results
            .iter()
            .map(|br| br.accuracy * 100.0)
            .collect();

        context.insert("chart_labels", &serde_json::to_string(&chart_labels).unwrap_or_default());
        context.insert("chart_data", &serde_json::to_string(&chart_data).unwrap_or_default());
        context.insert(
            "bench_labels",
            &serde_json::to_string(&bench_labels).unwrap_or_default(),
        );
        context.insert("bench_data", &serde_json::to_string(&bench_data).unwrap_or_default());

        // Custom CSS
        if let Some(ref css) = self.config.custom_css {
            context.insert("custom_css", css);
        }

        Ok(context)
    }

    /// Build context for comparison report template
    fn build_comparison_context(&self, report: &ComparisonReport) -> Result<Context> {
        let mut context = Context::new();

        context.insert("title", "Model Comparison Report");
        context.insert("report_id", &report.id.to_string());
        context.insert("timestamp", &report.timestamp.to_rfc3339());
        context.insert("benchmarks", &report.benchmarks);
        context.insert("theme", &self.config.theme);
        context.insert("include_charts", &self.config.include_charts);

        // Model results
        context.insert("model_results", &report.model_results);

        // Winners
        context.insert("overall_winner", &report.overall_winner);
        context.insert("benchmark_winners", &report.benchmark_winners);
        context.insert("category_winners", &report.category_winners);
        context.insert("statistical_tests", &report.statistical_tests);

        // Chart data for model comparison
        let model_names: Vec<String> = report
            .model_results
            .iter()
            .map(|mr| mr.model_name.clone())
            .collect();
        let model_scores: Vec<f64> = report
            .model_results
            .iter()
            .map(|mr| mr.overall_score * 100.0)
            .collect();

        context.insert(
            "model_names",
            &serde_json::to_string(&model_names).unwrap_or_default(),
        );
        context.insert(
            "model_scores",
            &serde_json::to_string(&model_scores).unwrap_or_default(),
        );

        // Per-benchmark comparison data
        let mut benchmark_comparison: Vec<HashMap<String, serde_json::Value>> = Vec::new();
        for benchmark_id in &report.benchmarks {
            let mut entry = HashMap::new();
            entry.insert(
                "benchmark_id".to_string(),
                serde_json::json!(benchmark_id),
            );

            let mut scores = Vec::new();
            for model_result in &report.model_results {
                if let Some(bench_result) = model_result
                    .benchmark_results
                    .iter()
                    .find(|r| &r.benchmark_id == benchmark_id)
                {
                    scores.push(serde_json::json!({
                        "model": model_result.model_name,
                        "score": bench_result.accuracy * 100.0
                    }));
                }
            }
            entry.insert("scores".to_string(), serde_json::json!(scores));
            benchmark_comparison.push(entry);
        }
        context.insert("benchmark_comparison", &benchmark_comparison);

        Ok(context)
    }

    /// Render the template with the given context
    fn render(&self, context: Context) -> Result<String> {
        self.tera
            .render("report.html", &context)
            .map_err(|e| Error::Config(format!("Template rendering error: {}", e)))
    }
}

impl Default for HtmlReportGenerator {
    fn default() -> Self {
        Self::new().expect("Failed to create default report generator")
    }
}

/// Generate an HTML report from an EvaluationReport
///
/// This is a convenience function that creates a default generator and generates the report.
pub fn generate_html_report(report: &EvaluationReport) -> Result<String> {
    let generator = HtmlReportGenerator::new()?;
    generator.generate(report)
}

/// Generate an HTML report from an EvaluationReport with custom configuration
pub fn generate_html_report_with_config(
    report: &EvaluationReport,
    config: HtmlReportConfig,
) -> Result<String> {
    let generator = HtmlReportGenerator::with_config(config)?;
    generator.generate(report)
}

/// Generate an HTML comparison report
pub fn generate_comparison_html_report(report: &ComparisonReport) -> Result<String> {
    let generator = HtmlReportGenerator::new()?;
    generator.generate_comparison(report)
}

/// Generate HTML from a stored report
pub fn generate_html_from_stored(report: &StoredEvaluationReport) -> Result<String> {
    // Parse the raw report back into EvaluationReport
    let evaluation_report: EvaluationReport = serde_json::from_value(report.raw_report.clone())
        .map_err(|e| Error::Config(format!("Failed to parse stored report: {}", e)))?;

    generate_html_report(&evaluation_report)
}

/// Save an HTML report to a file
pub fn save_html_report(report: &EvaluationReport, path: &str) -> Result<()> {
    let html = generate_html_report(report)?;
    std::fs::write(path, html)
        .map_err(|e| Error::Config(format!("Failed to write HTML report: {}", e)))?;
    Ok(())
}

/// Save an HTML comparison report to a file
pub fn save_comparison_html_report(report: &ComparisonReport, path: &str) -> Result<()> {
    let html = generate_comparison_html_report(report)?;
    std::fs::write(path, html)
        .map_err(|e| Error::Config(format!("Failed to write HTML report: {}", e)))?;
    Ok(())
}

/// Default HTML template with embedded Chart.js
const DEFAULT_TEMPLATE_MINIMAL: &str = r#"<!DOCTYPE html>
<html>
<head><title>{{ title }}</title></head>
<body>
    <h1>{{ model_name }}</h1>
    <p>Score: {{ overall_score }}</p>
</body>
</html>
"#;

const DEFAULT_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {% if custom_css %}
        {{ custom_css }}
        {% endif %}
        :root {
            --dark-theme-bg: #1a1a2e;
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; color: var(--primary-color); }
        .subtitle { color: var(--text-muted); font-size: 1.1rem; }
        .meta {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .meta-item { text-align: center; }
        .meta-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .meta-value { font-size: 1.1rem; font-weight: 600; margin-top: 5px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .card-title {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        .card-subtitle {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 5px;
        }
        .section {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }
        .section-title {
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }
        .score-bar {
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--success-color));
            border-radius: 4px;
        }
        .score-value { font-weight: 600; color: var(--success-color); }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .metric-item {
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
        }
        footer {
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p class="subtitle">Comprehensive Model Performance Analysis</p>
            <div class="meta">
                <div class="meta-item">
                    <div class="meta-label">Model</div>
                    <div class="meta-value">{{ model_name }}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Provider</div>
                    <div class="meta-value">{{ provider_name }}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Generated</div>
                    <div class="meta-value">{{ timestamp }}</div>
                </div>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-title">Overall Score</div>
                <div class="card-value">{{ overall_score_percentage | round(precision=1) }}%</div>
                <div class="card-subtitle">Weighted average across all benchmarks</div>
            </div>
            <div class="card">
                <div class="card-title">Benchmarks Run</div>
                <div class="card-value">{{ benchmark_results | length }}</div>
                <div class="card-subtitle">Total evaluation coverage</div>
            </div>
            <div class="card">
                <div class="card-title">Categories</div>
                <div class="card-value">{{ category_scores | length }}</div>
                <div class="card-subtitle">Performance domains tested</div>
            </div>
            {% if has_cost_analysis %}
            <div class="card">
                <div class="card-title">Est. Cost</div>
                <div class="card-value" style="font-size: 1.8rem;">{{ cost_analysis.cost_formatted }}</div>
                <div class="card-subtitle">{{ cost_analysis.total_tokens }} tokens</div>
            </div>
            {% endif %}
        </div>

        {% if include_charts %}
        <div class="section">
            <h2 class="section-title">Performance Visualization</h2>
            <div class="chart-container">
                <canvas id="categoryChart"></canvas>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2 class="section-title">Category Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Score</th>
                        <th>Progress</th>
                        <th>Samples</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cat in category_scores %}
                    <tr>
                        <td><strong>{{ cat.category }}</strong></td>
                        <td class="score-value">{{ cat.score_percentage | round(precision=1) }}%</td>
                        <td style="width: 40%;">
                            <div class="score-bar">
                                <div class="score-bar-fill" style="width: {{ cat.score_percentage }}%"></div>
                            </div>
                        </td>
                        <td>{{ cat.sample_count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Benchmark Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Accuracy</th>
                        <th>Progress</th>
                        <th>Correct</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>
                    {% for bench in benchmark_results %}
                    <tr>
                        <td>
                            <strong>{{ bench.benchmark_name }}</strong>
                        </td>
                        <td class="score-value">{{ bench.accuracy_percentage | round(precision=1) }}%</td>
                        <td style="width: 30%;">
                            <div class="score-bar">
                                <div class="score-bar-fill" style="width: {{ bench.accuracy_percentage }}%"></div>
                            </div>
                        </td>
                        <td>{{ bench.correct_count }}</td>
                        <td>{{ bench.sample_count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">Detailed Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-label">Mean Latency</div>
                    <div class="metric-value">{{ metrics.mean_latency_ms | round(precision=0) }}ms</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">P95 Latency</div>
                    <div class="metric-value">{{ metrics.p95_latency_ms | round(precision=0) }}ms</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Total Tokens</div>
                    <div class="metric-value">{{ metrics.total_tokens }}</div>
                </div>
                {% if metrics.tokens_per_second %}
                <div class="metric-item">
                    <div class="metric-label">Tokens/Second</div>
                    <div class="metric-value">{{ metrics.tokens_per_second | round(precision=1) }}</div>
                </div>
                {% endif %}
                <div class="metric-item">
                    <div class="metric-label">Errors</div>
                    <div class="metric-value" style="color: {% if metrics.error_count == 0 %}var(--success-color){% else %}var(--danger-color){% endif %}">
                        {{ metrics.error_count }}
                    </div>
                </div>
            </div>
        </div>

        {% if has_cost_analysis %}
        <div class="section">
            <h2 class="section-title">Cost Analysis</h2>
            <div class="metrics-grid">
                {% if cost_analysis.total_cost %}
                <div class="metric-item">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value">${{ cost_analysis.total_cost | round(precision=4) }}</div>
                </div>
                {% endif %}
                {% if cost_analysis.cost_per_1k_tokens %}
                <div class="metric-item">
                    <div class="metric-label">Cost per 1K Tokens</div>
                    <div class="metric-value">${{ cost_analysis.cost_per_1k_tokens | round(precision=4) }}</div>
                </div>
                {% endif %}
                <div class="metric-item">
                    <div class="metric-label">Total Tokens Used</div>
                    <div class="metric-value">{{ cost_analysis.total_tokens }}</div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if model_results %}
        <div class="section">
            <h2 class="section-title">Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Overall Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in model_results %}
                    <tr>
                        <td>{{ model.model_name }}</td>
                        <td>{{ model.overall_score }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if overall_winner %}
            <p>Winner: {{ overall_winner.model }}</p>
            {% endif %}
        </div>
        {% endif %}

        {% if model_results %}
        <div class="section">
            <h2 class="section-title">Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Overall Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in model_results %}
                    <tr>
                        <td>{{ model.model_name }}</td>
                        <td>{{ model.overall_score }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% if overall_winner %}
            <p>Winner: {{ overall_winner.model }}</p>
            {% endif %}
        </div>
        {% endif %}

        <footer>
            <p>Generated by Models Lab Evaluation System</p>
            <p style="font-size: 0.85rem; margin-top: 10px;">Report ID: {{ report_id }}</p>
        </footer>
    </div>

    <!-- Chart.js for visualization -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% if include_charts %}
    <script>
        // Simple chart placeholder
        console.log('Charts would be rendered here');
    </script>
    {% endif %}
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::{CategoryScore, DetailedMetrics, CostAnalysis};

    fn create_test_report() -> EvaluationReport {
        EvaluationReport {
            id: uuid::Uuid::new_v4(),
            model_name: "test-model".to_string(),
            provider_name: "test-provider".to_string(),
            overall_score: 0.75,
            category_scores: vec![
                CategoryScore {
                    category: "reasoning".to_string(),
                    score: 0.8,
                    sample_count: 100,
                    benchmarks: vec!["mmlu".to_string()],
                },
                CategoryScore {
                    category: "math".to_string(),
                    score: 0.7,
                    sample_count: 50,
                    benchmarks: vec!["gsm8k".to_string()],
                },
            ],
            benchmark_results: vec![
                crate::evaluation::BenchmarkResultSummary {
                    benchmark_id: "mmlu".to_string(),
                    benchmark_name: "MMLU".to_string(),
                    accuracy: 0.8,
                    sample_count: 100,
                    correct_count: 80,
                    mean_latency_ms: 500.0,
                },
            ],
            metrics: DetailedMetrics {
                mean_latency_ms: 500.0,
                p95_latency_ms: 750.0,
                total_tokens: 10000,
                tokens_per_second: Some(50.0),
                error_count: 0,
            },
            cost_analysis: Some(CostAnalysis {
                cost_per_1k_tokens: Some(0.002),
                total_cost: Some(0.02),
                total_tokens: 10000,
            }),
            config: crate::evaluation::EvaluationConfig::default(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_html_generator_creation() {
        let generator = HtmlReportGenerator::new();
        assert!(generator.is_ok());
    }

    #[test]
    fn test_generate_html_report() {
        let report = create_test_report();
        let result = generate_html_report(&report);
        assert!(result.is_ok());
        let html = result.unwrap();
        assert!(html.contains("test-model"));
        assert!(html.contains("75"));
        assert!(html.contains("Chart.js"));
    }

    #[test]
    fn test_report_with_dark_theme() {
        let report = create_test_report();
        let config = HtmlReportConfig {
            theme: "dark".to_string(),
            ..Default::default()
        };
        let generator = HtmlReportGenerator::with_config(config).unwrap();
        let html = generator.generate(&report).unwrap();
        // Template is simplified, just verify it generates valid HTML
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Chart.js"));
    }
}
