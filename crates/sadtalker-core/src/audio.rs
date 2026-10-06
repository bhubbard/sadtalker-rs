use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const STANDARD_SAMPLE_RATE: u32 = 16000;
pub const STANDARD_FPS: f32 = 25.0;
pub const SAMPLES_PER_FRAME: usize = (STANDARD_SAMPLE_RATE as f32 / STANDARD_FPS) as usize; // 640

/// Audio features representing a single video frame window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrameFeature {
    pub frame_index: usize,
    pub timestamp_sec: f32,
    pub rms_energy: f32,
    pub is_silent: bool,
    /// 16 mel-frequency energy values
    pub mel_energies: Vec<f32>,
}

/// Extracted audio track metadata and per-frame features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSequence {
    pub sample_rate: u32,
    pub duration_seconds: f32,
    pub total_samples: usize,
    pub total_frames: usize,
    pub frames: Vec<AudioFrameFeature>,
}

pub struct AudioProcessor;

impl AudioProcessor {
    /// Load WAV audio file and convert to 16kHz mono float buffer
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to open WAV audio file: {:?}", path.as_ref()))?;
        
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        let raw_samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap_or(0) as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        // Convert multi-channel to mono
        let mono_samples = if channels > 1 {
            raw_samples
                .chunks(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            raw_samples
        };

        // Resample if necessary (linear interpolation to 16,000 Hz)
        if source_rate != STANDARD_SAMPLE_RATE {
            Ok(Self::resample(&mono_samples, source_rate, STANDARD_SAMPLE_RATE))
        } else {
            Ok(mono_samples)
        }
    }

    /// Resample audio using linear interpolation
    pub fn resample(input: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
        if src_rate == target_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = target_rate as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut output = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            let val = input[low] * (1.0 - frac) + input[high] * frac;
            output.push(val);
        }
        output
    }

    /// Process audio into frame-synchronized features at standard 25 FPS
    pub fn extract_features(samples: &[f32], fps: f32) -> AudioSequence {
        let samples_per_frame = (STANDARD_SAMPLE_RATE as f32 / fps).round() as usize;
        let total_samples = samples.len();
        let total_frames = if total_samples == 0 {
            0
        } else {
            ((total_samples as f32) / (samples_per_frame as f32)).ceil() as usize
        };
        let duration = total_samples as f32 / STANDARD_SAMPLE_RATE as f32;

        let mut frames = Vec::with_capacity(total_frames);

        for f in 0..total_frames {
            let start = f * samples_per_frame;
            let end = (start + samples_per_frame).min(total_samples);
            let chunk = if start < total_samples {
                &samples[start..end]
            } else {
                &[]
            };

            // Calculate RMS energy
            let sum_sq: f32 = chunk.iter().map(|&x| x * x).sum();
            let rms = if !chunk.is_empty() {
                (sum_sq / chunk.len() as f32).sqrt()
            } else {
                0.0
            };
            let is_silent = rms < 0.005;

            // Simplified mel filterbank energy estimation (16 channels)
            let mel_energies = Self::compute_mel_energies(chunk, 16);

            frames.push(AudioFrameFeature {
                frame_index: f,
                timestamp_sec: f as f32 / fps,
                rms_energy: rms,
                is_silent,
                mel_energies,
            });
        }

        AudioSequence {
            sample_rate: STANDARD_SAMPLE_RATE,
            duration_seconds: duration,
            total_samples,
            total_frames,
            frames,
        }
    }

    /// Approximate 16-band mel spectrum energy for phoneme estimation
    fn compute_mel_energies(chunk: &[f32], bands: usize) -> Vec<f32> {
        if chunk.is_empty() {
            return vec![0.0; bands];
        }
        let mut energies = vec![0.0; bands];
        let sub_chunk_size = (chunk.len() / bands).max(1);

        for (b, energy) in energies.iter_mut().enumerate() {
            let start = (b * sub_chunk_size).min(chunk.len());
            let end = ((b + 1) * sub_chunk_size).min(chunk.len());
            let slice = &chunk[start..end];
            if !slice.is_empty() {
                // Apply Hamming window
                let mut sum = 0.0;
                for (i, &s) in slice.iter().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / slice.len() as f32).cos();
                    sum += (s * w).powi(2);
                }
                *energy = (sum / slice.len() as f32).sqrt();
            }
        }
        energies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let input = vec![0.1, 0.5, -0.3, 0.8];
        let out = AudioProcessor::resample(&input, 16000, 16000);
        assert_eq!(input, out);
    }

    #[test]
    fn test_extract_features() {
        // 1 second of 16kHz sine wave
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let seq = AudioProcessor::extract_features(&samples, 25.0);
        assert_eq!(seq.total_frames, 25);
        assert!(seq.frames[0].rms_energy > 0.1);
        assert!(!seq.frames[0].is_silent);
        assert_eq!(seq.frames[0].mel_energies.len(), 16);
    }
}
