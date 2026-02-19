//! Workflow engine

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::tasks::{Task, TaskContext, TaskResult};
use models_core::Result;

/// Workflow engine for executing tasks
pub struct WorkflowEngine {
    tasks: Arc<RwLock<Vec<Arc<dyn Task>>>>,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn with_task(mut self, task: Arc<dyn Task>) -> Self {
        // We can't modify directly in const context, use a helper
        let _ = &mut self.tasks;
        self
    }

    pub async fn register_task(&self, task: Arc<dyn Task>) {
        let mut tasks = self.tasks.write().await;
        tasks.push(task);
    }

    pub async fn run_all(&self, context: TaskContext) -> Result<Vec<TaskResult>> {
        let tasks = self.tasks.read().await;
        let mut results = Vec::new();

        for task in tasks.iter() {
            let result = task.execute(context.clone()).await;
            results.push(result);
        }

        Ok(results)
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}