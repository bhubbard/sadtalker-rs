use crate::audio::AudioSequence;

pub struct Audio2ExpGenerator {
    pub expression_scale: f32,
}

impl Audio2ExpGenerator {
    pub fn new(expression_scale: f32) -> Self {
        Self { expression_scale }
    }

    /// Generate 64-dim facial expression coefficients from audio features
    pub fn generate_expressions(&self, audio_seq: &AudioSequence, base_exp: &[f32]) -> Vec<Vec<f32>> {
        let total_frames = audio_seq.total_frames;
        let mut expressions = Vec::with_capacity(total_frames);

        for (_f, frame) in audio_seq.frames.iter().enumerate() {
            let mut exp = if base_exp.len() >= 64 {
                base_exp.to_vec()
            } else {
                vec![0.0; 64]
            };

            if !frame.is_silent {
                // Map audio energy bands to mouth and articulation blendshapes:
                // exp[0..5]: jaw open, lower lip depression
                // exp[6..10]: lip corner pull, mouth stretch
                // exp[14..18]: cheek puff, nostril flare, brow motion
                let energy = frame.rms_energy.min(1.0);
                let low_freq_energy = frame.mel_energies.iter().take(4).sum::<f32>() / 4.0;
                let mid_freq_energy = frame.mel_energies.iter().skip(4).take(8).sum::<f32>() / 8.0;

                let jaw_open = (low_freq_energy * 2.5 * self.expression_scale).min(1.5);
                let mouth_spread = (mid_freq_energy * 1.8 * self.expression_scale).min(1.2);
                let brow_raise = (energy * 0.4 * self.expression_scale).min(0.5);

                exp[0] = (exp[0] + jaw_open).clamp(-1.0, 2.0);
                exp[1] = (exp[1] + mouth_spread).clamp(-1.0, 1.5);
                exp[2] = (exp[2] + brow_raise).clamp(-0.5, 1.0);

                // Articulation blendshapes
                for (idx, &mel) in frame.mel_energies.iter().take(12).enumerate() {
                    let target_idx = 3 + idx;
                    if target_idx < 64 {
                        exp[target_idx] += mel * 0.6 * self.expression_scale;
                    }
                }
            }

            expressions.push(exp);
        }

        expressions
    }
}
