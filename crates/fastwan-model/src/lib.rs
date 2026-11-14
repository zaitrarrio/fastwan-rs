//! FastWan, written once against [`fastwan_core::Backend`]: the Wan 2.1 DiT
//! ([`dit`]), the TAEHV decoder ([`taehv`]), the ×4 upscaler ([`upscale`]), the DMD sampler ([`pipeline`])
//! and the op-level checker every backend must pass ([`check`]).

pub mod causal;
pub mod check;
pub mod dit;
pub mod pipeline;
pub mod taehv;
pub mod upscale;

pub use causal::{CausalConfig, CausalRollout};
pub use dit::{Attention, WanDit};
pub use pipeline::{Event, Options, Pipeline, Timings, Video};
pub use taehv::TaeHv;
pub use upscale::Upscaler;
