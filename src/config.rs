//! All tunables live here. Nothing in this crate reads a magic constant that is
//! not derived from a `Config`.

use crate::limiter::LimiterMode;
use anyhow::{bail, Result};

/// Sample interpolation quality used by both the CPU reference and the GPU path.
///
/// The GPU shader branches on this via a pipeline-overridable constant, so both
/// backends must agree exactly for the null test to pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Interpolation {
    /// Truncate to the nearest lower sample. One fetch per voice per frame.
    Nearest = 0,
    /// Two-point linear. The default.
    Linear = 1,
    /// Four-point Catmull-Rom. Roughly 2x the sample-pool bandwidth.
    Cubic = 2,
}

impl Interpolation {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "nearest" | "none" | "0" => Some(Interpolation::Nearest),
            "linear" | "1" => Some(Interpolation::Linear),
            "cubic" | "hermite" | "2" => Some(Interpolation::Cubic),
            _ => None,
        }
    }

    /// Number of pool samples the interpolator touches per frame.
    pub fn taps(self) -> u32 {
        match self {
            Interpolation::Nearest => 1,
            Interpolation::Linear => 2,
            Interpolation::Cubic => 4,
        }
    }
}

/// Shape of the decay and release envelope segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EnvelopeCurve {
    /// Level falls by a constant number of dB per frame (multiplicative).
    /// This is what the SoundFont spec asks for.
    Exponential = 0,
    /// Level falls by a constant amount per frame (additive). Matches the
    /// "linear envelope" option XSynth exposes.
    Linear = 1,
}

impl EnvelopeCurve {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "exp" | "exponential" | "db" => Some(EnvelopeCurve::Exponential),
            "lin" | "linear" => Some(EnvelopeCurve::Linear),
            _ => None,
        }
    }
}

/// What happens when a note-on arrives and the voice pool is full.
///
/// Fixed once, on purpose. The rule must never depend on scheduling order,
/// because that would make output non-reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StealRule {
    /// Kill the voices with the smallest note id, i.e. the ones that started
    /// earliest. Note ids are assigned by the host in event order, so this is
    /// a total order with no ties and is fully reproducible.
    Oldest = 0,
    /// Refuse to start the new note instead of killing an old one.
    DropNew = 1,
    /// Kill the voices with the lowest envelope level, which are the ones
    /// contributing least to the mix. Ties are broken by note id, so the
    /// victim set is still a total order with no ties and fully reproducible.
    ///
    /// `Oldest` is the wrong rule under saturation: the oldest voices are the
    /// mature, sounding ones and the survivors are whichever were struck most
    /// recently and are therefore still silent.
    Quietest = 2,
}

/// Which of a saturated block's note-ons get admitted when they outnumber the
/// free pool slots.
///
/// Distinct from `StealRule`, which decides who *dies*. On a heavily
/// oversubscribed file dropping dominates stealing by an order of magnitude,
/// so this is the rule that decides most of what is heard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AdmitRule {
    /// Rank by `voice::admit_key`: notes that outlive the block first, then by
    /// opening amplitude, with the note id as a tiebreak. Keeps the loud
    /// sustained material and spends the budget on notes that are audible.
    Loudest = 0,
    /// Thin the block evenly by event position, ignoring what each note is.
    /// The behaviour before ranking existed; kept for comparing against it.
    Even = 1,
}

impl Config {
    /// How many of `queued` note-ons a block can admit, given `live` voices
    /// already in the pool.
    ///
    /// Mirrors the arithmetic both backends do in their own `spawn`. The
    /// driver needs the same number to know where to partition the ranked
    /// block: partition short and the backend admits notes that were never
    /// ranked, partition long and the work is wasted. Kept here so the three
    /// call sites cannot drift apart.
    pub fn admit_take(&self, live: u32, queued: u32) -> u32 {
        let cap = self.max_voices;
        let want = queued.min(cap);
        if live + want <= cap {
            return want;
        }
        let need = match self.steal_rule {
            StealRule::DropNew => 0,
            _ => (live + want - cap).min(live).min(self.max_steal()),
        };
        want.min(cap - (live - need))
    }
}

impl AdmitRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "loudest" | "loud" | "rank" | "ranked" => Some(AdmitRule::Loudest),
            "even" | "spread" | "flat" => Some(AdmitRule::Even),
            _ => None,
        }
    }
}

impl StealRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "oldest" => Some(StealRule::Oldest),
            "quietest" | "quiet" | "level" => Some(StealRule::Quietest),
            "drop" | "dropnew" | "drop-new" => Some(StealRule::DropNew),
            _ => None,
        }
    }
}

/// Which backend renders the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Single-threaded scalar Rust. The ground truth for every test.
    Cpu,
    /// wgpu compute pipeline.
    Gpu,
}

#[derive(Debug, Clone)]
pub struct Config {
    // ---- stream ----------------------------------------------------------
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Output channel count. Only 2 is implemented; kept here so the constant
    /// never gets hardcoded at a call site.
    pub channels: u32,

    // ---- block scheduling ------------------------------------------------
    /// Frames rendered per dispatch. At 48 kHz, 4096 frames is about 85 ms,
    /// which makes per-block dispatch overhead irrelevant.
    pub block_frames: u32,
    /// Frames handled per workgroup reduction step inside the render shader.
    /// `block_frames` must be a multiple of this.
    pub reduce_tile: u32,
    /// Frames between note-off gate checks, and so the granularity of a
    /// note-off. Must be a multiple of `reduce_tile`. Every voice reads one
    /// gate entry per gate tile, which measured 6.8 ms per block at a million
    /// voices when it was tied to the reduce tile; at 32 frames the check
    /// costs a quarter of that and a note still releases within 0.7 ms.
    pub gate_frames: u32,
    /// Invocations per workgroup in the render pass. One voice per invocation.
    pub workgroup_size: u32,
    /// Upper bound on workgroups dispatched by the render pass. Voices beyond
    /// `max_render_workgroups * workgroup_size` are handled by looping inside
    /// the shader. This directly sizes the partial-sum buffer.
    pub max_render_workgroups: u32,
    /// Upper bound on the grid taken by every *per-voice* pass: spawn, steal
    /// selection, compaction, and the radix sort. All of them grid-stride over
    /// the pool, so this decides how many workgroups walk it and never what
    /// they compute -- which is what lets the pool hold more WG-sized blocks
    /// than one dispatch dimension can address. `u32::MAX` means "whatever the
    /// backend's dispatch ceiling is", and the backend clamps to that either
    /// way. Only the tests set it, to force the striding path on a pool small
    /// enough to fit on a test machine.
    pub max_pool_workgroups: u32,

    // ---- voice pool ------------------------------------------------------
    /// Hard ceiling on concurrent voices.
    pub max_voices: u32,
    /// Voices per (channel, key) note-on. SoundFont layers spawn one voice
    /// each; this caps runaway presets.
    pub max_layers: u32,
    pub steal_rule: StealRule,
    /// Which note-ons survive when a block oversubscribes the pool.
    pub admit_rule: AdmitRule,
    /// Ceiling on how much of the pool one block may steal, in percent.
    ///
    /// Without this, a block whose note-ons outnumber the pool steals *every*
    /// voice and refills every slot, so each block opens with the whole pool at
    /// `env_level = 0`. That is a periodic amplitude notch at the block rate,
    /// and the limiter turns it into audible pumping. It also means no voice
    /// ever lives longer than one block, so a saturated section
    /// renders as a stream of 85 ms attack fragments rather than notes.
    ///
    /// Bounding the churn trades note count for note length: fewer note-ons
    /// survive, but the ones that do get to ring. A voice's minimum lifetime is
    /// roughly `100 / steal_percent` blocks. 100 restores the old behaviour.
    ///
    /// 25 measured best on The Nuker 4. Almost all of the benefit is in
    /// bounding the churn at all -- 100 -> 25 removes the notch entirely and
    /// takes the block-rate modulation from 32 dB to 23.6 dB, where 10 buys a
    /// further 1 dB and 4 another 1 dB. Below 25 it also gets slower: 10%
    /// measured 285 s against 208 s on the same 132 s render, because a pool
    /// whose voices come from a wider span of the file holds more distinct
    /// regions and sorts worse.
    pub max_steal_percent: u32,
    /// Frames a stolen voice fades over instead of being cut.
    ///
    /// A steal used to take effect at frame zero for every victim at once,
    /// which put a step at the block boundary however small the batch was.
    /// Victims now stop at a frame derived from their own note id, spread
    /// across the block, and fade linearly to silence over this many frames.
    /// Fixed-length rather than a forced envelope release, because the pool
    /// has to be back under `max_voices` by the end of the block and a
    /// release rate read from the region could take arbitrarily long.
    pub steal_fade_frames: u32,
    /// Re-sort the voice pool by (region, envelope stage, phase) during
    /// compaction. This is the single biggest performance lever in the whole
    /// renderer and it is nearly free, because compaction already has to copy
    /// the pool. Off only for measuring what it buys.
    pub sort_voices: bool,

    // ---- dsp -------------------------------------------------------------
    pub interpolation: Interpolation,
    pub decay_curve: EnvelopeCurve,
    pub release_curve: EnvelopeCurve,
    /// Envelope level below which a releasing voice is considered dead.
    /// -100 dB by default.
    pub env_floor: f32,
    /// Enable the per-voice low-pass filter. SoundFonts that leave the cutoff
    /// at its default (13500 cents) bypass it anyway.
    pub filter_enabled: bool,
    /// Linear gain applied to the final mix before limiting.
    pub master_volume: f32,
    /// Where CC7 sits on a channel that never sends one. General MIDI says a
    /// channel powers on at 100, which is about 4 dB down; this defaults to
    /// 127 instead, so a file that never mentions channel volume renders at
    /// the level it always did and pays nothing for the feature. Set it to
    /// 100 to match a GM synth exactly.
    pub default_channel_volume: u8,
    /// How many copies of the params table the sound controllers CC71-CC75 may
    /// build. Each is a full rebuild of the table, so this trades host and
    /// device memory for how finely a file can sweep those controllers; past
    /// the cap a channel reuses the nearest copy it already has.
    pub max_param_variants: u32,
    /// Apply the soft limiter to the mixed output.
    pub limiter: bool,
    /// Which limiter runs. `Brickwall` is the default: a lookahead true-peak
    /// limiter that never exceeds the ceiling, so `clamp_block` never has to
    /// hard-clip, and that only pulls the gain down around the peak that needs
    /// it instead of for a third of a second afterwards. `Omni` is the port of
    /// the one OmniConverter ships, kept for level-matching against BASS and
    /// XSynth but deprecated for rendering.
    pub limiter_mode: LimiterMode,
    /// Brickwall ceiling in dBFS. 0.0 is flat full scale.
    pub limiter_ceiling_db: f64,
    /// Brickwall lookahead in milliseconds. This is also the render latency.
    pub limiter_lookahead_ms: f64,
    /// Brickwall release in milliseconds. Short keeps the material after a
    /// transient at full level; too short modulates the gain at audio rate on
    /// sustained loud material, which is its own distortion.
    pub limiter_release_ms: f64,
    /// Time constant of the brickwall's sustained stage, in milliseconds.
    ///
    /// A single release time has to choose between pumping, if it is long, and
    /// modulating the gain at audio rate on sustained loud material, if it is
    /// short. This second, slower stage takes the sustained part of the
    /// reduction so the fast one only ever handles transients.
    ///
    /// Off by default, because it measured *worse* on the material this synth
    /// exists for. Program-dependent release trades on a distinction between
    /// transient and sustained passages, and a saturated black MIDI mix has
    /// none: it is sustained throughout, so the slow stage stays engaged and
    /// becomes a second slow-release compressor, putting back the breathing
    /// the brickwall was there to remove. On The Nuker 4 it took the
    /// block-rate modulation from 8.8 dB back up to 15.4 dB and the 400 ms
    /// loudness variation from 0.090 to 0.134. Kept because material with real
    /// transients is a different question.
    pub limiter_sustain_ms: f64,
    /// Detect inter-sample peaks by 4x oversampling rather than looking at the
    /// samples alone. Costs a little; catches overshoot that a sample-peak
    /// limiter lets through.
    pub limiter_true_peak: bool,
    /// Limiter attack/release in seconds.
    pub limiter_attack: f32,
    pub limiter_release: f32,

    // ---- sample pool -----------------------------------------------------
    /// Resample every sample to a single rate at load time. When false, the
    /// per-sample rate ratio is folded into
    /// the phase step instead, which is lossless but leaves the pool mixed-rate.
    pub resample_pool: bool,
    /// Soft ceiling on sample-pool bytes on the device. When the pool does not
    /// fit, its rate is halved until it does rather than failing outright.
    pub sample_pool_budget: u64,

    // ---- numerics --------------------------------------------------------
    /// Use Kahan compensation in the top levels of the reduction tree.
    pub kahan_reduce: bool,
    /// Check every output block for NaN/Inf. Always on in debug builds.
    pub nan_guard: bool,
    /// Compile the shaders without naga's automatic bounds clamps and loop
    /// bounding counters. Every index in the shaders comes from a count the
    /// host controls, so the clamps are dead weight, but turning them off
    /// means a bug becomes an out-of-bounds access rather than a clamped one.
    /// Off by default; measure before turning it on.
    pub unchecked_shaders: bool,

    // ---- diagnostics -----------------------------------------------------
    /// Log per-pass timings.
    pub profile: bool,
    /// Force a specific wgpu backend, e.g. "vulkan" or "dx12".
    pub gpu_backend: Option<String>,
    /// Substring match against the adapter name.
    pub gpu_adapter: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sample_rate: 48_000,
            channels: 2,

            block_frames: 4096,
            // Swept on an RTX 5060 at a million voices. The pairing matters
            // more than either value alone, because `workgroup_size /
            // (reduce_tile * 2)` is how many threads cooperate on one output
            // lane in the first reduction level, and 32 of them -- one warp --
            // is what measures best. 256/4 renders in 84.3 ms where 512/8 took
            // 95.6 ms and 512/4, at the same workgroup storage as 256/8, took
            // 104 ms.
            reduce_tile: 4,
            gate_frames: 32,
            workgroup_size: 256,
            // 2048 workgroups of 256 is 524288 voices in flight; past that the
            // render pass loops. Wider dispatches keep winning here (84.3 ms
            // at 512, 82.5 ms at 2048) at the cost of a partial buffer that
            // scales with the count: 2048 workgroups is 64 MiB, which the
            // reduce pass then has to stream, and which is why this stops
            // rather than doubling again.
            max_render_workgroups: 2048,
            max_pool_workgroups: u32::MAX,

            max_voices: 1 << 20,
            max_layers: 16,
            steal_rule: StealRule::Quietest,
            admit_rule: AdmitRule::Loudest,
            max_steal_percent: 25,
            steal_fade_frames: 96,
            sort_voices: true,

            interpolation: Interpolation::Linear,
            decay_curve: EnvelopeCurve::Exponential,
            release_curve: EnvelopeCurve::Exponential,
            env_floor: 1.0e-5, // -100 dB
            filter_enabled: true,
            master_volume: 1.0,
            default_channel_volume: 127,
            max_param_variants: 32,
            limiter: true,
            limiter_mode: LimiterMode::Brickwall,
            limiter_ceiling_db: 0.0,
            limiter_lookahead_ms: 2.0,
            limiter_release_ms: 60.0,
            limiter_sustain_ms: 0.0,
            limiter_true_peak: true,
            limiter_attack: 0.01,
            limiter_release: 0.1,

            resample_pool: true,
            sample_pool_budget: 2 << 30, // 2 GiB, leaves room for Cake

            kahan_reduce: false,
            nan_guard: cfg!(debug_assertions),
            unchecked_shaders: false,

            profile: false,
            gpu_backend: None,
            gpu_adapter: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.channels != 2 {
            bail!("only stereo output is implemented (channels = {})", self.channels);
        }
        if self.sample_rate < 8_000 || self.sample_rate > 768_000 {
            bail!("sample_rate {} out of range", self.sample_rate);
        }
        if self.block_frames == 0 || self.block_frames % self.reduce_tile != 0 {
            bail!(
                "block_frames ({}) must be a non-zero multiple of reduce_tile ({})",
                self.block_frames,
                self.reduce_tile
            );
        }
        if !self.workgroup_size.is_power_of_two()
            || self.workgroup_size < 32
            || self.workgroup_size > 1024
        {
            bail!(
                "workgroup_size {} must be a power of two in [32, 1024]",
                self.workgroup_size
            );
        }
        if !self.reduce_tile.is_power_of_two() {
            bail!("reduce_tile {} must be a power of two", self.reduce_tile);
        }
        if self.gate_frames == 0
            || self.gate_frames % self.reduce_tile != 0
            || self.block_frames % self.gate_frames != 0
        {
            bail!(
                "gate_frames ({}) must be a multiple of reduce_tile ({}) and divide                  block_frames ({})",
                self.gate_frames,
                self.reduce_tile,
                self.block_frames
            );
        }
        // The render shader stages `workgroup_size * reduce_tile` floats in
        // workgroup storage. wgpu guarantees at least 16 KiB.
        // The render pass stages one float per voice per channel lane, plus a
        // second level of WG floats.
        let shared_bytes = self.workgroup_size as u64 * (self.reduce_tile as u64 * 2 + 1) * 4;
        if shared_bytes > 49152 {
            bail!(
                "workgroup_size {} with reduce_tile {} needs {} bytes of workgroup storage, \
                 over the 48 KiB a compute workgroup can have",
                self.workgroup_size,
                self.reduce_tile,
                shared_bytes
            );
        }
        // The reduction assigns at least one thread to each of the
        // reduce_tile * 2 channel lanes.
        if self.workgroup_size < self.reduce_tile * 2 {
            bail!(
                "workgroup_size {} must be at least twice reduce_tile {}",
                self.workgroup_size,
                self.reduce_tile
            );
        }
        if !(-24.0..=0.0).contains(&self.limiter_ceiling_db) {
            bail!(
                "limiter_ceiling_db {} must be in -24..=0",
                self.limiter_ceiling_db
            );
        }
        if !(0.05..=100.0).contains(&self.limiter_lookahead_ms) {
            bail!(
                "limiter_lookahead_ms {} must be in 0.05..=100",
                self.limiter_lookahead_ms
            );
        }
        if !(0.0..=10000.0).contains(&self.limiter_sustain_ms) {
            bail!(
                "limiter_sustain_ms {} must be in 0..=10000",
                self.limiter_sustain_ms
            );
        }
        if !(0.1..=5000.0).contains(&self.limiter_release_ms) {
            bail!(
                "limiter_release_ms {} must be in 0.1..=5000",
                self.limiter_release_ms
            );
        }
        if self.max_voices == 0 {
            bail!("max_voices must be non-zero");
        }
        if self.steal_fade_frames == 0 || self.steal_fade_frames >= self.block_frames {
            bail!(
                "steal_fade_frames {} must be in 1..block_frames ({})",
                self.steal_fade_frames,
                self.block_frames
            );
        }
        if self.max_steal_percent == 0 || self.max_steal_percent > 100 {
            bail!(
                "max_steal_percent {} must be in 1..=100",
                self.max_steal_percent
            );
        }
        // 255 and 65535 are the widths the driver packs a note-on into while
        // it waits for admission. Well past anything musical, but a silent
        // truncation there would corrupt which voice a candidate refers to.
        if self.max_layers > 255 {
            bail!("max_layers {} must be at most 255", self.max_layers);
        }
        if self.block_frames > 65535 {
            bail!("block_frames {} must be at most 65535", self.block_frames);
        }
        if self.max_layers == 0 {
            bail!("max_layers must be non-zero");
        }
        Ok(())
    }

    /// The brickwall ceiling as a linear amplitude.
    pub fn limiter_ceiling(&self) -> f64 {
        10f64.powf(self.limiter_ceiling_db / 20.0)
    }

    /// Frames in which a steal may be scheduled. The fade has to finish inside
    /// the block, so a victim cannot start fading in the last `fade` frames.
    pub fn steal_span(&self) -> u32 {
        self.block_frames.saturating_sub(self.steal_fade_frames).max(1)
    }

    /// Voice slots to allocate. A stolen voice keeps sounding until its own
    /// stop frame, so for part of a block the pool holds the outgoing voices
    /// and their replacements at once. `max_voices` stays the ceiling on what
    /// survives a block; this is the transient headroom on top of it.
    pub fn pool_slots(&self) -> u32 {
        self.max_voices.saturating_add(self.max_steal())
    }

    /// The most voices one block may steal. Integer arithmetic, and both
    /// backends call this rather than computing it themselves, because the two
    /// have to agree exactly or the null test stops meaning anything.
    pub fn max_steal(&self) -> u32 {
        let n = self.max_voices as u64 * self.max_steal_percent as u64 / 100;
        (n as u32).max(1)
    }

    /// Bytes of one interleaved stereo output block.
    pub fn block_bytes(&self) -> u64 {
        self.block_frames as u64 * self.channels as u64 * 4
    }

    /// Samples (not frames) in one interleaved output block.
    pub fn block_samples(&self) -> usize {
        self.block_frames as usize * self.channels as usize
    }

    pub fn rate_f64(&self) -> f64 {
        self.sample_rate as f64
    }
}
