use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub num_elements: usize,
}

pub struct WeightsInspector;

impl WeightsInspector {
    /// Inspect a SafeTensors checkpoint file and return metadata for all contained tensors
    pub fn inspect_safetensors<P: AsRef<Path>>(path: P) -> Result<Vec<TensorInfo>> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Cannot open weights file {:?}", path.as_ref()))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)
            .with_context(|| "Failed to parse SafeTensors buffer")?;

        let mut infos = Vec::new();
        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let num_elements = shape.iter().product();
            let dtype = format!("{:?}", view.dtype());

            infos.push(TensorInfo {
                name,
                shape,
                dtype,
                num_elements,
            });
        }

        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    /// Discover available model checkpoints in designated directory
    pub fn list_checkpoints<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
        let mut ckpts = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir.as_ref()) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(ext) = p.extension() {
                    if ext == "safetensors" || ext == "pth" || ext == "bin" {
                        ckpts.push(p);
                    }
                }
            }
        }
        ckpts.sort();
        ckpts
    }
}
