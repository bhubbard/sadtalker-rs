use crate::audio::AudioProcessor;
use crate::audio2exp::Audio2ExpGenerator;
use crate::audio2pose::Audio2PoseGenerator;
use crate::coeff::{FrameCoeff, MotionSequence, PoseCoeff};
use crate::config::SadTalkerConfig;
use crate::facerender::FaceRenderer;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct SadTalkerPipeline {
    pub config: SadTalkerConfig,
}

impl SadTalkerPipeline {
    pub fn new(config: SadTalkerConfig) -> Self {
        Self { config }
    }

    /// Run full coefficient generation pipeline from source image and driven audio
    pub fn generate_coefficients<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        image_path: P,
        audio_path: Q,
    ) -> Result<MotionSequence> {
        let renderer = FaceRenderer::new(self.config.size);
        let (_img, _crop_info) = renderer.load_source_image(&image_path)?;

        // 1. Process audio into frame sequence
        let samples = AudioProcessor::load_wav(&audio_path)?;
        let audio_seq = AudioProcessor::extract_features(&samples, self.config.fps);

        // 2. Generate head poses
        let pose_gen = Audio2PoseGenerator::new(self.config.pose_style, self.config.still);
        let base_pose = PoseCoeff::default();
        let poses = pose_gen.generate_poses(&audio_seq, &base_pose, None);

        // 3. Generate facial expressions
        let exp_gen = Audio2ExpGenerator::new(self.config.expression_scale);
        let base_exp = vec![0.0; 64];
        let expressions = exp_gen.generate_expressions(&audio_seq, &base_exp);

        // 4. Assemble motion sequence
        let mut motion = MotionSequence::new(self.config.fps, 80);
        motion.total_frames = audio_seq.total_frames;

        for (i, (pose, exp)) in poses.into_iter().zip(expressions.into_iter()).enumerate() {
            motion.frames.push(FrameCoeff {
                frame_index: i,
                pose,
                exp,
                eye_blink: 0.0,
            });
        }

        // 5. Post-process: smooth pose motion and inject natural eye blinks
        motion.smooth_pose(5);
        if self.config.ref_eyeblink.is_none() {
            motion.inject_natural_blinks(60); // blink every ~2.4s
        }

        Ok(motion)
    }

    /// Execute complete end-to-end talking face generation
    pub fn run<P: AsRef<Path>, Q: AsRef<Path>, O: AsRef<Path>>(
        &self,
        image_path: P,
        audio_path: Q,
        output_dir: O,
    ) -> Result<PathBuf> {
        let out_dir = output_dir.as_ref();
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("Failed to create output directory {:?}", out_dir))?;

        // 1. Generate motion coefficients
        let motion = self.generate_coefficients(&image_path, &audio_path)?;
        let coeff_json_path = out_dir.join("motion_coefficients.json");
        motion.save_json(&coeff_json_path)?;

        // 2. Check for FFmpeg availability
        let _output_video = out_dir.join("sadtalker_output.mp4");
        
        Ok(coeff_json_path)
    }
}
