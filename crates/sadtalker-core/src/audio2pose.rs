use crate::audio::AudioSequence;
use crate::coeff::PoseCoeff;
use std::f32::consts::PI;

pub struct Audio2PoseGenerator {
    pub pose_style: usize,
    pub still_mode: bool,
}

impl Audio2PoseGenerator {
    pub fn new(pose_style: usize, still_mode: bool) -> Self {
        Self {
            pose_style,
            still_mode,
        }
    }

    /// Generate head pose sequence from audio features
    pub fn generate_poses(
        &self,
        audio_seq: &AudioSequence,
        base_pose: &PoseCoeff,
        ref_poses: Option<&[PoseCoeff]>,
    ) -> Vec<PoseCoeff> {
        let total_frames = audio_seq.total_frames;
        let mut poses = Vec::with_capacity(total_frames);

        if let Some(refs) = ref_poses {
            if !refs.is_empty() {
                // Loop or interpolate reference poses
                for i in 0..total_frames {
                    let ref_idx = i % refs.len();
                    poses.push(refs[ref_idx].clone());
                }
                return poses;
            }
        }

        if self.still_mode {
            // Still mode: minimal organic micro-movements around base pose
            for i in 0..total_frames {
                let t = i as f32 / audio_seq.frames.len().max(1) as f32;
                let micro_pitch = (t * 2.0 * PI).sin() * 0.2;
                let micro_yaw = (t * 1.5 * PI).cos() * 0.15;
                poses.push(PoseCoeff {
                    pitch: base_pose.pitch + micro_pitch,
                    yaw: base_pose.yaw + micro_yaw,
                    roll: base_pose.roll,
                    tx: base_pose.tx,
                    ty: base_pose.ty,
                    tz: base_pose.tz,
                });
            }
            return poses;
        }

        // Stylized dynamic pose generation conditioned on audio rhythm & energy
        let style_factor = (self.pose_style % 5 + 1) as f32;
        let mut curr_pitch = base_pose.pitch;
        let mut curr_yaw = base_pose.yaw;
        let mut curr_roll = base_pose.roll;

        for (i, frame) in audio_seq.frames.iter().enumerate() {
            let energy = frame.rms_energy;
            let time = i as f32 / 25.0;

            // Audio cadence creates head nod on energetic beats
            let cadence = (time * 3.0 * PI).sin();
            let nod_amplitude = (energy * 15.0 * style_factor).min(6.0);
            let turn_amplitude = (energy * 10.0 * style_factor).min(5.0);

            let target_pitch = base_pose.pitch + (cadence * nod_amplitude);
            let target_yaw = base_pose.yaw + ((time * 1.2 * PI).cos() * turn_amplitude);
            let target_roll = base_pose.roll + ((time * 0.8 * PI).sin() * (turn_amplitude * 0.3));

            // Smooth momentum step towards targets
            curr_pitch += (target_pitch - curr_pitch) * 0.15;
            curr_yaw += (target_yaw - curr_yaw) * 0.12;
            curr_roll += (target_roll - curr_roll) * 0.12;

            poses.push(PoseCoeff {
                pitch: curr_pitch,
                yaw: curr_yaw,
                roll: curr_roll,
                tx: base_pose.tx + ((time * 0.5).sin() * 0.005),
                ty: base_pose.ty + ((time * 0.7).cos() * 0.004),
                tz: base_pose.tz,
            });
        }

        poses
    }
}
