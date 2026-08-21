//! GPU-accelerated SoundFont/SFZ synthesizer for black MIDI.
//!
//! The architecture in short: the host parses MIDI and resolves presets, the
//! device renders voices and reduces them to a stereo block, and every GPU
//! result is checked against the single-threaded CPU reference in `cpu`.

pub mod backend;
pub mod bank;
pub mod config;
pub mod cpu;
pub mod driver;
pub mod fixed;
pub mod gpu;
pub mod limiter;
pub mod midi;
pub mod resample;
pub mod sf2;
pub mod sfz;
pub mod testkit;
pub mod voice;
pub mod wav;

pub use backend::{Backend, BlockStats};
pub use bank::Bank;
pub use config::{BackendKind, Config, EnvelopeCurve, Interpolation, StealRule};
pub use driver::Driver;

use anyhow::{bail, Result};
use std::path::Path;

/// Load a soundfont, dispatching on the file extension.
pub fn load_bank(path: impl AsRef<Path>, cfg: &Config) -> Result<Bank> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "sf2" | "sf3" | "sfbk" => sf2::load(path, cfg),
        "sfz" => sfz::load(path, cfg),
        _ => bail!(
            "{}: unrecognised soundfont extension {:?}; expected .sf2 or .sfz",
            path.display(),
            ext
        ),
    }
}
