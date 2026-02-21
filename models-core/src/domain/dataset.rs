//! Dataset domain types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Dataset split type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

/// Dataset entity (represents a dataset record in the database)
/// Renamed from `Dataset` to avoid collision with the benchmark framework's `Dataset` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntity {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub num_samples: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DatasetEntity {
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            num_samples: 0,
            created_at: now,
            updated_at: now,
        }
    }
}