//! Domain types

mod model;
mod experiment;
mod dataset;

pub use model::{Model, ModelProvider, ModelStatus};
pub use experiment::{Experiment, ExperimentStatus, ExperimentConfig};
pub use dataset::{Dataset, DatasetSplit};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Base entity with common fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Entity {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl Default for Entity {
    fn default() -> Self {
        Self::new()
    }
}