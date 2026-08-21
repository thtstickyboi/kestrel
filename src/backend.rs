//! The interface the driver renders through.
//!
//! Deliberately narrow: the driver owns MIDI parsing, preset resolution and
//! the gate table, and a backend only has to accept spawn commands and hand
//! back mixed blocks. That is what lets the CPU reference and the GPU path be
//! swapped under the same test.

use crate::voice::SpawnCmd;
use anyhow::Result;

#[derive(Debug, Clone, Copy, Default)]
pub struct BlockStats {
    pub active_voices: u64,
    /// Voices killed to make room, cumulative.
    pub stolen: u64,
    /// Note-ons that never became voices, cumulative.
    pub dropped: u64,
    /// Peak absolute sample in the last block, before limiting.
    pub peak: f32,
}

pub trait Backend {
    /// Publish this block's note-off gate rows. `rows` is `tiles * GATE_SLOTS`
    /// entries, tile-major.
    fn set_gates(&mut self, rows: &[u32]) -> Result<()>;

    /// Publish this block's per-channel controller state. `rows` is
    /// `tiles * BEND_CHANNELS * CHAN_FIELDS` words, tile-major: an 8.24 bend
    /// factor and two f32 gains per channel. The two flags are false when that
    /// control sat at unity for the whole block, which lets a backend skip the
    /// work rather than multiply by one.
    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool)
        -> Result<()>;

    /// Install one copy of the region params table. Variant 0 is the
    /// untouched one installed at construction; later ones are built when a
    /// channel's CC71-CC75 reach a combination not seen before, which is rare
    /// enough that this is a cold path.
    fn set_params_variant(&mut self, index: u32, data: &[crate::bank::RegionParams]) -> Result<()>;

    /// Add voices to the pool. May steal or drop according to the configured
    /// rule if the pool is full.
    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()>;

    /// Render one block into `out`, interleaved stereo, `block_frames * 2`
    /// samples. Advances every voice and compacts the pool.
    fn render(&mut self, out: &mut [f32]) -> Result<()>;

    fn stats(&self) -> BlockStats;

    fn name(&self) -> &'static str;

    /// Per-pass timings from the last block, for `--profile`. Empty when the
    /// backend has nothing to report.
    fn timings(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}
