use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use sadtalker_core::{
    AudioProcessor, EnhancerType, PreprocessMode, SadTalkerConfig, SadTalkerPipeline,
    WeightsInspector,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "sadtalker",
    author = "SadTalker Contributors",
    version = "0.1.0",
    about = "High-performance Rust CLI for SadTalker: 3D audio-driven stylized talking face animation"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate realistic talking face animation from source portrait and driving audio
    Generate {
        /// Path to source portrait image (PNG/JPEG)
        #[arg(short = 'i', long = "source-image", required = true)]
        source_image: PathBuf,

        /// Path to driving audio WAV file
        #[arg(short = 'a', long = "driven-audio", required = true)]
        driven_audio: PathBuf,

        /// Destination directory for generated results
        #[arg(short = 'o', long = "result-dir", default_value = "./results")]
        result_dir: PathBuf,

        /// Video resolution size (256 or 512)
        #[arg(long = "size", default_value_t = 256)]
        size: u32,

        /// Pose style preset index (0 to 45)
        #[arg(long = "pose-style", default_value_t = 0)]
        pose_style: usize,

        /// Expression intensity multiplier
        #[arg(long = "expression-scale", default_value_t = 1.0)]
        expression_scale: f32,

        /// Enable still mode to minimize head wobble
        #[arg(long = "still")]
        still: bool,

        /// Verbose output logging
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },

    /// Extract audio-driven 3DMM motion coefficients without rendering video
    Audio2coeff {
        /// Path to source portrait image
        #[arg(short = 'i', long = "source-image", required = true)]
        source_image: PathBuf,

        /// Path to driving audio WAV file
        #[arg(short = 'a', long = "driven-audio", required = true)]
        driven_audio: PathBuf,

        /// Destination path for generated motion coefficients JSON
        #[arg(short = 'o', long = "output", default_value = "motion_coefficients.json")]
        output: PathBuf,

        /// Pose style preset index
        #[arg(long = "pose-style", default_value_t = 0)]
        pose_style: usize,

        /// Still mode flag
        #[arg(long = "still")]
        still: bool,
    },

    /// Inspect audio tracks, motion coefficient files, or SafeTensors checkpoints
    Inspect {
        /// Path to audio WAV, motion JSON, or SafeTensors file
        #[arg(required = true)]
        file: PathBuf,
    },

    /// Benchmark audio feature extraction and motion generation pipeline
    Bench {
        /// Optional audio file to benchmark against (defaults to synthetic 30s tone)
        #[arg(short = 'a', long = "audio")]
        audio: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            source_image,
            driven_audio,
            result_dir,
            size,
            pose_style,
            expression_scale,
            still,
            verbose,
        } => {
            println!("{}", "=== SadTalker Native Rust Pipeline ===".bold().cyan());
            println!("  Source Image:  {}", source_image.display().to_string().yellow());
            println!("  Driven Audio:  {}", driven_audio.display().to_string().yellow());
            println!("  Resolution:    {}x{}", size, size);
            println!("  Pose Style:    {}", pose_style);
            println!("  Still Mode:    {}", still);

            let config = SadTalkerConfig {
                size,
                fps: 25.0,
                pose_style,
                expression_scale,
                still,
                preprocess: PreprocessMode::Crop,
                enhancer: EnhancerType::None,
                batch_size: 2,
                checkpoint_dir: None,
                ref_eyeblink: None,
                ref_pose: None,
                verbose,
            };

            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap(),
            );
            pb.set_message("Processing audio and calculating 3DMM coefficients...");

            let start = Instant::now();
            let pipeline = SadTalkerPipeline::new(config);
            let out_coeff = pipeline.run(&source_image, &driven_audio, &result_dir)?;

            pb.finish_with_message(format!(
                "Done in {:.2?}! Coefficients saved to {}",
                start.elapsed(),
                out_coeff.display().to_string().green()
            ));
        }

        Commands::Audio2coeff {
            source_image,
            driven_audio,
            output,
            pose_style,
            still,
        } => {
            println!("{}", "=== SadTalker Audio2Coeff Extractor ===".bold().cyan());
            let config = SadTalkerConfig {
                pose_style,
                still,
                ..Default::default()
            };

            let pipeline = SadTalkerPipeline::new(config);
            let motion = pipeline.generate_coefficients(&source_image, &driven_audio)?;
            motion.save_json(&output)?;

            println!(
                "{} Generated {} frames of 3D motion coefficients at {} FPS -> {}",
                "✓".green(),
                motion.total_frames,
                motion.fps,
                output.display().to_string().bold()
            );
        }

        Commands::Inspect { file } => {
            println!("{} {}", "Inspecting:".bold().cyan(), file.display());
            let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");

            match ext.to_lowercase().as_str() {
                "wav" => {
                    let samples = AudioProcessor::load_wav(&file)?;
                    let seq = AudioProcessor::extract_features(&samples, 25.0);
                    println!("  Type:          WAV Audio");
                    println!("  Sample Rate:   {} Hz", seq.sample_rate);
                    println!("  Duration:      {:.2} seconds", seq.duration_seconds);
                    println!("  Total Samples: {}", seq.total_samples);
                    println!("  Video Frames:  {} (@ 25 FPS)", seq.total_frames);
                    let silent_count = seq.frames.iter().filter(|f| f.is_silent).count();
                    println!("  Silent Frames: {} / {}", silent_count, seq.total_frames);
                }
                "json" => {
                    let motion = sadtalker_core::MotionSequence::load_json(&file)?;
                    println!("  Type:          3DMM Motion Sequence");
                    println!("  FPS:           {}", motion.fps);
                    println!("  Total Frames:  {}", motion.total_frames);
                    println!("  ID Dimensions: {}", motion.id_coeff.len());
                    let max_yaw = motion
                        .frames
                        .iter()
                        .map(|f| f.pose.yaw.abs())
                        .fold(0.0f32, f32::max);
                    let blinks = motion.frames.iter().filter(|f| f.eye_blink > 0.5).count();
                    println!("  Max Yaw Angle: {:.2}°", max_yaw);
                    println!("  Blink Frames:  {}", blinks);
                }
                "safetensors" => {
                    let tensors = WeightsInspector::inspect_safetensors(&file)?;
                    println!("  Type:          SafeTensors Checkpoint");
                    println!("  Tensors Count: {}", tensors.len());
                    let total_params: usize = tensors.iter().map(|t| t.num_elements).sum();
                    println!("  Total Params:  {:.2} M", total_params as f64 / 1_000_000.0);
                    for t in tensors.iter().take(10) {
                        println!("    • {:<35} {:?} ({})", t.name, t.shape, t.dtype);
                    }
                    if tensors.len() > 10 {
                        println!("    ... and {} more tensors", tensors.len() - 10);
                    }
                }
                _ => {
                    println!("Unsupported file extension: .{}", ext);
                }
            }
        }

        Commands::Bench { audio } => {
            println!("{}", "=== Benchmarking SadTalker Core Pipeline ===".bold().cyan());
            let samples = if let Some(p) = audio {
                AudioProcessor::load_wav(&p)?
            } else {
                // Generate 10 seconds of synthetic audio @ 16kHz
                (0..160_000)
                    .map(|i| (i as f32 * 0.05).sin() * 0.3)
                    .collect()
            };

            let duration = samples.len() as f32 / 16000.0;
            println!("Testing on {:.1} seconds of audio...", duration);

            let t0 = Instant::now();
            let seq = AudioProcessor::extract_features(&samples, 25.0);
            let audio_dur = t0.elapsed();

            let pose_gen = sadtalker_core::Audio2PoseGenerator::new(0, false);
            let base_pose = sadtalker_core::PoseCoeff::default();
            let t1 = Instant::now();
            let poses = pose_gen.generate_poses(&seq, &base_pose, None);
            let pose_dur = t1.elapsed();

            let exp_gen = sadtalker_core::Audio2ExpGenerator::new(1.0);
            let base_exp = vec![0.0; 64];
            let t2 = Instant::now();
            let exps = exp_gen.generate_expressions(&seq, &base_exp);
            let exp_dur = t2.elapsed();

            println!("\nBenchmark Results:");
            println!("  Audio Feature Extraction: {:.2?} ({} frames, {:.1} fps throughput)",
                audio_dur, seq.total_frames, seq.total_frames as f64 / audio_dur.as_secs_f64());
            println!("  3D Pose Trajectory Gen:   {:.2?} ({} poses, {:.1} fps throughput)",
                pose_dur, poses.len(), poses.len() as f64 / pose_dur.as_secs_f64());
            println!("  3D Expression Blendshapes: {:.2?} ({} frames, {:.1} fps throughput)",
                exp_dur, exps.len(), exps.len() as f64 / exp_dur.as_secs_f64());
        }
    }

    Ok(())
}
