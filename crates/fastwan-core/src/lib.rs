//! `fastwan-core`: everything about FastWan that is not a GPU.
//!
//! The model, sampler and decoder in `fastwan-model` are written once against
//! [`backend::Backend`]; the CPU reference, WebGPU and CUDA backends implement
//! that one op vocabulary. Nothing here allocates device memory.

pub mod backend;
pub mod config;
pub mod geometry;
pub mod plan;
pub mod rng;
pub mod rope;
pub mod schedule;
pub mod timestep;
pub mod weights;

pub use backend::{Act, Backend, ConvEpilogue, NormKind};
pub use config::{TaeConfig, WanConfig};
pub use geometry::Geometry;
pub use schedule::DmdSchedule;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("weights: {0}")]
    Weights(String),
    #[error("geometry: {0}")]
    Geometry(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;
