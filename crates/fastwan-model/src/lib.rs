//! FastWan, written once against [`fastwan_core::Backend`]: the Wan 2.1 DiT
//! ([`dit`]), the TAEHV decoder ([`taehv`]), the DMD sampler ([`pipeline`])
//! and the op-level checker every backend must pass ([`check`]).

pub mod check;
pub mod dit;
pub mod pipeline;
pub mod taehv;

pub use dit::WanDit;
pub use pipeline::{Event, Options, Pipeline, Timings, Video};
pub use taehv::TaeHv;
