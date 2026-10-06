use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// 3D Head Pose: Euler angles (pitch, yaw, roll) and 3D translation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseCoeff {
    /// Pitch angle (nodding up/down) in degrees
    pub pitch: f32,
    /// Yaw angle (turning left/right) in degrees
    pub yaw: f32,
    /// Roll angle (tilting sideways) in degrees
    pub roll: f32,
    /// Translation X (horizontal offset)
    pub tx: f32,
    /// Translation Y (vertical offset)
    pub ty: f32,
    /// Translation Z (depth scale)
    pub tz: f32,
}

/// 3D Morphable Model (3DMM) Coefficients for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameCoeff {
    pub frame_index: usize,
    /// 3D head pose orientation and translation
    pub pose: PoseCoeff,
    /// 64-dimensional facial expression blendshapes
    pub exp: Vec<f32>,
    /// Eye-blink coefficient scalar [0.0 = open, 1.0 = closed]
    pub eye_blink: f32,
}

impl FrameCoeff {
    pub fn new_neutral(frame_index: usize) -> Self {
        Self {
            frame_index,
            pose: PoseCoeff::default(),
            exp: vec![0.0; 64],
            eye_blink: 0.0,
        }
    }
}

/// Full temporal sequence of 3D motion coefficients for an animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionSequence {
    pub fps: f32,
    pub total_frames: usize,
    /// 80-dimensional face identity coefficients (constant for identity)
    pub id_coeff: Vec<f32>,
    /// Per-frame dynamic expression and pose
    pub frames: Vec<FrameCoeff>,
}

impl MotionSequence {
    pub fn new(fps: f32, id_dim: usize) -> Self {
        Self {
            fps,
            total_frames: 0,
            id_coeff: vec![0.0; id_dim],
            frames: Vec::new(),
        }
    }

    /// Apply temporal Gaussian/moving average smoothing to head pose to eliminate jitter
    pub fn smooth_pose(&mut self, window_size: usize) {
        if self.frames.len() <= window_size || window_size <= 1 {
            return;
        }
        let half = window_size / 2;
        let orig_poses: Vec<PoseCoeff> = self.frames.iter().map(|f| f.pose.clone()).collect();

        for i in 0..self.frames.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(orig_poses.len());
            let count = (end - start) as f32;

            let mut sum_pitch = 0.0;
            let mut sum_yaw = 0.0;
            let mut sum_roll = 0.0;
            let mut sum_tx = 0.0;
            let mut sum_ty = 0.0;
            let mut sum_tz = 0.0;

            for p in &orig_poses[start..end] {
                sum_pitch += p.pitch;
                sum_yaw += p.yaw;
                sum_roll += p.roll;
                sum_tx += p.tx;
                sum_ty += p.ty;
                sum_tz += p.tz;
            }

            let p = &mut self.frames[i].pose;
            p.pitch = sum_pitch / count;
            p.yaw = sum_yaw / count;
            p.roll = sum_roll / count;
            p.tx = sum_tx / count;
            p.ty = sum_ty / count;
            p.tz = sum_tz / count;
        }
    }

    /// Synthesize natural periodic eye blinks every ~2-4 seconds if no reference provided
    pub fn inject_natural_blinks(&mut self, blink_interval_frames: usize) {
        let blink_duration = 5; // 5 frames @ 25fps = 200ms blink duration
        for (i, frame) in self.frames.iter_mut().enumerate() {
            let cycle = i % blink_interval_frames;
            if cycle < blink_duration {
                // Sinusoidal blink curve [0 -> 1 -> 0]
                let t = (cycle as f32 / blink_duration as f32) * std::f32::consts::PI;
                let blink_val = t.sin();
                frame.eye_blink = frame.eye_blink.max(blink_val);
                // Also modulate upper/lower eyelid blendshapes
                if frame.exp.len() >= 14 {
                    frame.exp[11] += blink_val * 0.8; // Left eye closure
                    frame.exp[12] += blink_val * 0.8; // Right eye closure
                }
            }
        }
    }

    /// Save motion sequence to a JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load motion sequence from a JSON file
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let seq = serde_json::from_reader(reader)?;
        Ok(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_smoothing_and_blinks() {
        let mut seq = MotionSequence::new(25.0, 80);
        for i in 0..50 {
            let mut f = FrameCoeff::new_neutral(i);
            f.pose.yaw = (i as f32 * 0.5) + (if i % 2 == 0 { 2.0 } else { -2.0 });
            seq.frames.push(f);
        }
        seq.total_frames = seq.frames.len();

        seq.smooth_pose(5);
        seq.inject_natural_blinks(25);

        assert_eq!(seq.frames.len(), 50);
        assert!(seq.frames[2].eye_blink > 0.0);
    }
}
