use sadtalker_core::{
    Audio2ExpGenerator, Audio2PoseGenerator, AudioProcessor, EnhancerType, FaceRenderer,
    FrameCoeff, MotionSequence, PoseCoeff, PreprocessMode, SadTalkerConfig,
    WeightsInspector,
};
use tempfile::NamedTempFile;

#[test]
fn test_pose_smoothing_and_blinks() {
    let mut seq = MotionSequence::new(25.0, 80);
    for i in 0..60 {
        let mut f = FrameCoeff::new_neutral(i);
        // Add artificial high frequency noise
        f.pose.pitch = 5.0 + if i % 2 == 0 { 2.0 } else { -2.0 };
        f.pose.yaw = (i as f32) * 0.2;
        seq.frames.push(f);
    }
    seq.total_frames = seq.frames.len();

    seq.smooth_pose(5);
    seq.inject_natural_blinks(30);

    assert_eq!(seq.frames.len(), 60);
    // Noise should be smoothed
    assert!((seq.frames[10].pose.pitch - 5.0).abs() < 1.0);
    // Check eye blink injected
    assert!(seq.frames[2].eye_blink > 0.0);
}

#[test]
fn test_motion_sequence_json_roundtrip() {
    let mut seq = MotionSequence::new(25.0, 80);
    let mut f = FrameCoeff::new_neutral(0);
    f.pose.pitch = 3.5;
    f.pose.yaw = -2.1;
    f.pose.roll = 0.8;
    f.exp[0] = 0.45;
    seq.frames.push(f);
    seq.total_frames = 1;

    let temp_file = NamedTempFile::new().unwrap();
    seq.save_json(temp_file.path()).unwrap();

    let loaded = MotionSequence::load_json(temp_file.path()).unwrap();
    assert_eq!(loaded.total_frames, 1);
    assert_eq!(loaded.fps, 25.0);
    assert!((loaded.frames[0].pose.pitch - 3.5).abs() < 1e-5);
    assert!((loaded.frames[0].exp[0] - 0.45).abs() < 1e-5);
}

#[test]
fn test_audio2pose_generator() {
    let gen_moving = Audio2PoseGenerator::new(0, false);
    let gen_still = Audio2PoseGenerator::new(0, true);

    let dummy_samples = vec![0.1f32; 16000 * 2]; // 2 seconds
    let dummy_audio = AudioProcessor::extract_features(&dummy_samples, 25.0);

    let base_pose = PoseCoeff::default();
    let poses_moving = gen_moving.generate_poses(&dummy_audio, &base_pose, None);
    let poses_still = gen_still.generate_poses(&dummy_audio, &base_pose, None);

    assert_eq!(poses_moving.len(), 50);
    assert_eq!(poses_still.len(), 50);

    // Still mode should have minimal micro-movements (<= 0.25 deg)
    for p in poses_still {
        assert!(p.pitch.abs() <= 0.25);
        assert!(p.yaw.abs() <= 0.25);
        assert_eq!(p.roll, 0.0);
    }
}

#[test]
fn test_audio2exp_generator() {
    let gen = Audio2ExpGenerator::new(1.5);
    let dummy_samples = vec![0.1f32; 16000 * 2];
    let dummy_audio = AudioProcessor::extract_features(&dummy_samples, 25.0);

    let base_exp = vec![0.1f32; 64];
    let exps = gen.generate_expressions(&dummy_audio, &base_exp);
    assert_eq!(exps.len(), 50);
    assert_eq!(exps[0].len(), 64);
}

#[test]
fn test_config_and_render_modes() {
    let config = SadTalkerConfig::default();
    assert_eq!(config.size, 256);
    assert_eq!(config.fps, 25.0);
    assert_eq!(config.preprocess, PreprocessMode::Crop);
    assert_eq!(config.enhancer, EnhancerType::None);

    let renderer = FaceRenderer::new(256);
    assert_eq!(renderer.target_size, 256);
}

#[test]
fn test_weights_inspector_invalid_path() {
    let res = WeightsInspector::inspect_safetensors("/invalid/path/sadtalker.safetensors");
    assert!(res.is_err());
}
