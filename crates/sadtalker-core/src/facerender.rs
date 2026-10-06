use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageReader};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Metadata for face bounding box and alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceCropInfo {
    pub orig_width: u32,
    pub orig_height: u32,
    pub crop_box: [u32; 4], // [x1, y1, x2, y2]
    pub target_size: u32,
}

pub struct FaceRenderer {
    pub target_size: u32,
}

impl FaceRenderer {
    pub fn new(target_size: u32) -> Self {
        Self { target_size }
    }

    /// Load and inspect source portrait image
    pub fn load_source_image<P: AsRef<Path>>(&self, path: P) -> Result<(DynamicImage, FaceCropInfo)> {
        let img = ImageReader::open(path.as_ref())
            .with_context(|| format!("Failed to open image at {:?}", path.as_ref()))?
            .decode()
            .with_context(|| "Failed to decode image")?;

        let (orig_width, orig_height) = img.dimensions();

        // Standard center crop calculation around face
        let min_dim = orig_width.min(orig_height);
        let x1 = (orig_width - min_dim) / 2;
        let y1 = (orig_height - min_dim) / 2;
        let x2 = x1 + min_dim;
        let y2 = y1 + min_dim;

        let crop_info = FaceCropInfo {
            orig_width,
            orig_height,
            crop_box: [x1, y1, x2, y2],
            target_size: self.target_size,
        };

        Ok((img, crop_info))
    }

    /// Check if ffmpeg is installed and accessible in system PATH
    pub fn check_ffmpeg() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Render video by assembling image frames and muxing audio track via FFmpeg
    pub fn mux_video_audio(
        frames_pattern: &str,
        audio_path: &Path,
        output_mp4: &Path,
        fps: f32,
    ) -> Result<()> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-r",
            &fps.to_string(),
            "-i",
            frames_pattern,
            "-i",
            audio_path.to_str().unwrap_or(""),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            output_mp4.to_str().unwrap_or(""),
        ]);

        let output = cmd.output().with_context(|| "Failed to execute ffmpeg process")?;
        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("FFmpeg encoding failed: {}", err_msg);
        }

        Ok(())
    }
}
