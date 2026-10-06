use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Face crop / preprocessing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreprocessMode {
    Crop,
    Resize,
    Full,
    ExtCrop,
    ExtFull,
}

impl Default for PreprocessMode {
    fn default() -> Self {
        Self::Crop
    }
}

/// Face enhancer selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnhancerType {
    None,
    Gfpgan,
    RestoreFormer,
    CodeFormer,
}

impl Default for EnhancerType {
    fn default() -> Self {
        Self::None
    }
}

/// Generation configuration options for SadTalker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SadTalkerConfig {
    /// Video frame resolution (256 or 512)
    pub size: u32,
    /// Video frame rate (default 25 fps)
    pub fps: f32,
    /// Style index for head pose dynamics (0 to 45)
    pub pose_style: usize,
    /// Expression intensity multiplier (e.g. 1.0)
    pub expression_scale: f32,
    /// Still mode (minimizes unnatural head wobble while preserving lip sync)
    pub still: bool,
    /// Face preprocessing mode
    pub preprocess: PreprocessMode,
    /// Face enhancement model
    pub enhancer: EnhancerType,
    /// Batch size for neural inference
    pub batch_size: usize,
    /// Custom checkpoint directory path (optional)
    pub checkpoint_dir: Option<PathBuf>,
    /// Optional reference video for eye blink animation
    pub ref_eyeblink: Option<PathBuf>,
    /// Optional reference video for head pose dynamics
    pub ref_pose: Option<PathBuf>,
    /// Verbose logging flag
    pub verbose: bool,
}

impl Default for SadTalkerConfig {
    fn default() -> Self {
        Self {
            size: 256,
            fps: 25.0,
            pose_style: 0,
            expression_scale: 1.0,
            still: false,
            preprocess: PreprocessMode::Crop,
            enhancer: EnhancerType::None,
            batch_size: 2,
            checkpoint_dir: None,
            ref_eyeblink: None,
            ref_pose: None,
            verbose: false,
        }
    }
}
