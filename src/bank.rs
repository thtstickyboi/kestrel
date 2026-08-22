//! Format-independent instrument bank.
//!
//! SF2 and SFZ both get flattened into this at load time: a flat i16 sample
//! pool plus a list of regions with every generator already resolved. Note-on
//! then costs one range lookup and a handful of multiplies. This is branchy
//! table-lookup work and it stays on the CPU permanently, by design.

use crate::config::{Config, EnvelopeCurve};
use crate::fixed::Fixed;
use std::f64::consts::PI;

/// Voice flag bits. Mirrored by `shaders/common.wgsl`.
pub const VF_LOOP: u32 = 1 << 0;
/// Loop only until the note is released, then run out the tail (SF2 mode 3).
pub const VF_LOOP_UNTIL_RELEASE: u32 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    NoLoop,
    Continuous,
    UntilRelease,
}

impl LoopMode {
    pub fn flags(self) -> u32 {
        match self {
            LoopMode::NoLoop => 0,
            LoopMode::Continuous => VF_LOOP,
            LoopMode::UntilRelease => VF_LOOP | VF_LOOP_UNTIL_RELEASE,
        }
    }
}

/// One sample in the flat pool.
#[derive(Debug, Clone)]
pub struct SampleInfo {
    /// Index of the first frame in `Bank::pool`.
    pub start: u32,
    pub len: u32,
    /// Loop points relative to `start`.
    pub loop_start: u32,
    pub loop_end: u32,
    /// Rate the data is stored at. After a pool rebuild this equals
    /// `Bank::pool_rate` for every sample.
    pub rate: u32,
    pub root_key: u8,
    /// Sample-level tuning, in cents.
    pub correction_cents: f32,
    /// Pool frames per source frame. Region address offsets are quoted in
    /// source frames, so they need this to land in the right place after a
    /// pool rebuild.
    pub resample_ratio: f32,
    pub name: String,
}

/// A key/velocity zone with every SF2 generator already applied.
#[derive(Debug, Clone)]
pub struct Region {
    pub sample: u32,
    pub key_lo: u8,
    pub key_hi: u8,
    pub vel_lo: u8,
    pub vel_hi: u8,
    /// Overrides the sample's root key when >= 0.
    pub root_key_override: i16,
    /// Fixed key/velocity for drum-style regions; -1 when unset.
    pub fixed_key: i16,
    pub fixed_vel: i16,
    pub coarse_tune: i16,
    pub fine_tune: i16,
    /// Cents of pitch change per key. 100 is normal, 0 pins the pitch.
    pub scale_tuning: i16,
    pub attenuation_cb: f32,
    /// SFZ `amp_veltrack`, in percent. 100 is full velocity tracking, 0 pins
    /// every note at full amplitude. SF2 has no equivalent and leaves it at
    /// 100, which is the SF2 default velocity-to-attenuation modulator.
    pub amp_veltrack: f32,
    /// -1.0 hard left to 1.0 hard right.
    pub pan: f32,
    pub loop_mode: LoopMode,
    /// SF2 address offset generators, in source frames, applied on top of the
    /// sample's own start/end/loop points. `addr_start` doubles as the SFZ
    /// `offset` opcode.
    pub addr_start: i32,
    pub addr_end: i32,
    pub addr_loop_start: i32,
    pub addr_loop_end: i32,
    pub exclusive_class: u8,

    // Volume envelope, in seconds, before key scaling.
    pub delay: f32,
    pub attack: f32,
    pub hold: f32,
    pub decay: f32,
    /// Sustain as a linear level in [0, 1].
    pub sustain: f32,
    pub release: f32,
    /// SF2 keynumToVolEnvHold / Decay, in timecents per key relative to key 60.
    pub keynum_to_hold: i16,
    pub keynum_to_decay: i16,

    pub filter_fc_cents: f32,
    pub filter_q_cb: f32,
    /// SFZ `fil_veltrack`: how far the cutoff opens at full velocity, in
    /// cents. Sampled piano libraries lean on this hard -- the EastWest
    /// piano library used for testing sets a cutoff of 89 Hz and a veltrack of
    /// 9600, so the
    /// velocity is doing eight octaves of the tonal work and ignoring it
    /// leaves every note under an 89 Hz lowpass.
    pub filter_veltrack_cents: f32,

    /// First entry of this region's block in `Bank::params`.
    pub params_base: u32,
    /// 0 when one entry covers every key, 1 when there is an entry per key.
    pub params_stride: u32,
    /// Entries per velocity step in this region's block, 1 when the cutoff
    /// does not track velocity. Indexed from `vel_lo`.
    pub params_vel_span: u32,
}

impl Default for Region {
    fn default() -> Self {
        Region {
            sample: 0,
            key_lo: 0,
            key_hi: 127,
            vel_lo: 0,
            vel_hi: 127,
            root_key_override: -1,
            fixed_key: -1,
            fixed_vel: -1,
            coarse_tune: 0,
            fine_tune: 0,
            scale_tuning: 100,
            attenuation_cb: 0.0,
            amp_veltrack: 100.0,
            pan: 0.0,
            loop_mode: LoopMode::NoLoop,
            addr_start: 0,
            addr_end: 0,
            addr_loop_start: 0,
            addr_loop_end: 0,
            exclusive_class: 0,
            delay: 0.0,
            attack: 0.001,
            hold: 0.0,
            decay: 0.001,
            sustain: 1.0,
            release: 0.001,
            keynum_to_hold: 0,
            keynum_to_decay: 0,
            filter_fc_cents: 13500.0,
            filter_veltrack_cents: 0.0,
            filter_q_cb: 0.0,
            params_base: 0,
            params_stride: 0,
            params_vel_span: 1,
        }
    }
}

/// Per-(region, key) DSP constants, uploaded once and read by the render pass
/// at the start of each block.
///
/// Layout is repeated verbatim in `shaders/common.wgsl`; keep them in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionParams {
    /// Per-frame increment during attack.
    pub attack_rate: f32,
    /// Attack ends when the raw level reaches this. Hold is folded in here:
    /// the level keeps climbing past 1.0 while the audible gain is clamped,
    /// which reproduces SF2 hold without needing a fifth envelope stage.
    pub attack_end: f32,
    /// Multiplier (exponential curve) or decrement (linear curve) per frame.
    pub decay_coef: f32,
    /// Decay stops here: max(sustain, env_floor).
    pub decay_target: f32,
    pub sustain: f32,
    pub release_coef: f32,
    pub b0: f32,
    pub b1: f32,
    pub a1: f32,
    pub a2: f32,
    /// bit 0: run the filter.
    pub flags: u32,
    pub _pad: u32,
}

pub const RP_FILTER: u32 = 1 << 0;

/// What a channel's sound controllers do to a region's DSP constants.
///
/// CC71 through CC75 all modify things that live in `RegionParams` -- filter
/// coefficients and envelope rates -- rather than anything a voice carries.
/// They cannot be a per-channel scalar the way volume can, because the biquad
/// for a region with a 400 Hz cutoff and one with 8 kHz are not related by a
/// multiply. So instead the whole params table is rebuilt for each distinct
/// combination of these controllers that a file actually uses, and a channel
/// selects between the copies. A voice's stored params index is an offset
/// within a copy; the channel supplies which copy.
///
/// This costs host memory and nothing at all on the device: the render pass
/// already loads one `RegionParams` per voice, and now loads it from a
/// different place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamMod {
    /// CC74 brightness, as an offset to the region's cutoff in cents.
    pub cutoff_cents: f32,
    /// CC71 resonance, as an offset in centibels of Q. Never negative: see
    /// `from_controllers`.
    pub q_cb: f32,
    /// CC73, CC75, CC72: what happens to each envelope time.
    pub attack: TimeMod,
    pub decay: TimeMod,
    pub release: TimeMod,
}

/// What a sound controller does to one envelope time.
///
/// Not a multiplier, which is what this used to be and what made CC73 do
/// nothing audible. BASSMIDI shortens by scaling and lengthens by *adding*, so
/// expressing it needs both terms: `base * scale + add`, in seconds. Below the
/// neutral position `add` is zero, above it `scale` is one, so one pair covers
/// the whole range exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeMod {
    pub scale: f32,
    /// Seconds added, on top of the scaled time.
    pub add: f32,
}

impl TimeMod {
    pub const NONE: TimeMod = TimeMod { scale: 1.0, add: 0.0 };

    #[inline]
    pub fn apply(self, secs: f32) -> f32 {
        secs * self.scale + self.add
    }

    /// One controller's contribution, 64 being neutral.
    ///
    /// See `CC_*` above for where the constants come from.
    pub fn from_controller(v: u8) -> TimeMod {
        let d = v as f32 - 64.0;
        if d <= 0.0 {
            TimeMod { scale: (d / CC_ENV_STEPS_PER_OCTAVE).exp2(), add: 0.0 }
        } else {
            TimeMod { scale: 1.0, add: CC_ENV_ADD_PER_CUBE * d * d * d }
        }
    }
}

impl Default for TimeMod {
    fn default() -> Self {
        TimeMod::NONE
    }
}

impl Default for ParamMod {
    fn default() -> Self {
        ParamMod {
            cutoff_cents: 0.0,
            q_cb: 0.0,
            attack: TimeMod::NONE,
            decay: TimeMod::NONE,
            release: TimeMod::NONE,
        }
    }
}

// ---------------------------------------------------------------------------
// Sound controller constants
//
// Measured against BASSMIDI 2.4.16 rather than guessed, by driving both synths
// over identical soundfonts and note timings and sweeping each controller
// through all 128 values. The measurements that matter ran the identical sweep
// against two soundfonts differing only in the base envelope time, which is
// what separates a mapping that scales the region's own value from one that
// adds to it. Every constant below carries the number it was fitted to.
// ---------------------------------------------------------------------------

/// Cents of cutoff offset per CC74 step.
///
/// Measured 74.92 cents/step, the same on a 1000 Hz region and a 4000 Hz one to
/// within 0.005, over a hundred controller values each. The method reads
/// Kestrel's own 38.095 as 38.089 on one font and 37.958 on another, so it
/// carries about 0.2% of slope error of its own; 75 is inside that, and is the
/// obvious design constant -- an octave every 16 steps, four octaves either way.
pub const CC_CUTOFF_CENTS_PER_STEP: f32 = 75.0;

/// Centibels of resonance per CC71 step **above 64**. Below 64 BASSMIDI does
/// nothing at all, which is the half that was wrong here.
///
/// Measured 3.674 cB/step against a method that reads Kestrel's own 3.810 as
/// 3.711, so the corrected figure is 3.77 and the existing 240 cB across the
/// upper half is inside the error. The depth was never the problem.
pub const CC_Q_CB_PER_STEP: f32 = 240.0 / 63.0;

/// Controller steps per octave of envelope time below 64. The time halves every
/// eight steps, so 0 is 1/256 of the region's own time.
///
/// Measured 7.92 to 8.00 steps per octave across release, decay and attack and
/// across both base times: `dec_long` lands on 0.03129, 0.06250, 0.12501,
/// 0.25000, 0.50000 of its base at CC 24, 32, 40, 48, 56.
pub const CC_ENV_STEPS_PER_OCTAVE: f32 = 8.0;

/// Seconds of envelope time *added* per cubed step above 64.
///
/// This is the finding the whole exercise was for. Above the neutral position
/// BASSMIDI does not scale the region's time, it adds to it -- so a soundfont
/// with a 1 ms attack and one with a 200 ms attack both gain the same 15
/// seconds at CC73=127, and a multiplicative model cannot reproduce that at any
/// depth. On the 50 ms and 1000 ms release fonts the added time agreed to
/// within 0.4% at every one of 61 controller values, and lands on
/// `0.06 ms * (v-64)^3` to the second decimal: 30.72, 245.76, 829.44, 1966.08 ms
/// at CC 72, 80, 88, 96. Release, decay and attack all share it.
pub const CC_ENV_ADD_PER_CUBE: f32 = 6.0e-5;

/// The shortest fade BASSMIDI performs, in seconds.
///
/// At the bottom of the range the scaling stops mattering and every release
/// becomes the same fixed ramp -- and a linear one: at CC72=0 the decay fits a
/// straight amplitude line to 1.06 dB where an exponential needs 3.55. Both the
/// 50 ms and the 1000 ms font land on the same 4.0 ms fade.
pub const CC_MIN_FADE_SECS: f32 = 0.004;

impl ParamMod {
    pub fn is_neutral(&self) -> bool {
        *self == ParamMod::default()
    }

    /// Build from the raw controller values, 64 being the neutral position for
    /// each. BASSMIDI powers all five up at 64, so this agrees with it about
    /// where "does nothing" is.
    pub fn from_controllers(cc71: u8, cc72: u8, cc73: u8, cc74: u8, cc75: u8) -> Self {
        ParamMod {
            cutoff_cents: (cc74 as f32 - 64.0) * CC_CUTOFF_CENTS_PER_STEP,
            // One-sided on purpose. BASSMIDI's resonance is identical to five
            // decimal places for every value from 0 to 64 and only climbs above
            // it; the old symmetric curve thinned the tone on the whole lower
            // half of a controller that should have been doing nothing there.
            q_cb: (cc71 as f32 - 64.0).max(0.0) * CC_Q_CB_PER_STEP,
            attack: TimeMod::from_controller(cc73),
            decay: TimeMod::from_controller(cc75),
            release: TimeMod::from_controller(cc72),
        }
    }
}

/// A bank/program pair with its zones flattened.
#[derive(Debug, Clone)]
pub struct Preset {
    pub bank: u16,
    pub program: u16,
    pub name: String,
    /// Indices into `Bank::regions`.
    pub regions: Vec<u32>,
    /// `key_index[k] .. key_index[k+1]` slices `key_regions` for key `k`.
    pub key_index: Vec<u32>,
    pub key_regions: Vec<u32>,
}

impl Preset {
    fn build_key_index(&mut self, regions: &[Region]) {
        let mut per_key: Vec<Vec<u32>> = vec![Vec::new(); 128];
        for &r in &self.regions {
            let reg = &regions[r as usize];
            for k in reg.key_lo..=reg.key_hi.min(127) {
                per_key[k as usize].push(r);
            }
        }
        self.key_index = Vec::with_capacity(129);
        self.key_regions = Vec::new();
        for list in per_key {
            self.key_index.push(self.key_regions.len() as u32);
            self.key_regions.extend_from_slice(&list);
        }
        self.key_index.push(self.key_regions.len() as u32);
    }
}

/// Everything the renderer needs from a soundfont.
pub struct Bank {
    pub pool: Vec<i16>,
    /// Rate every pool sample is stored at, or 0 when the pool is mixed-rate
    /// and the ratio has to be folded into the phase step per sample.
    pub pool_rate: u32,
    pub samples: Vec<SampleInfo>,
    pub regions: Vec<Region>,
    pub params: Vec<RegionParams>,
    pub presets: Vec<Preset>,
    /// Sorted (bank, program) -> preset index.
    pub(crate) index: Vec<((u16, u16), u32)>,
    pub name: String,

    // ---- admission preview tables, built by `finish` --------------------
    //
    // Admission has to rank a note-on before anything is built for it, and
    // `admit_key` ranks on the voice's opening gain. That gain depends only on
    // `(region, velocity)`, and whether a layer exists at all depends only on
    // `(region, key)`, so both are precomputed here and the ranking pass reads
    // them instead of calling `build_voice`. See `preview_note_on`.
    /// `region * 128 + vel` -> the `(gain_l, gain_r)` `build_voice` would
    /// produce. Bit-exact, not an approximation: the arithmetic below is the
    /// same sequence of f32 operations on the same inputs.
    pub(crate) gain_table: Vec<[f32; 2]>,
    /// `build_voice`'s `delay_frames` per region. A region constant, and the
    /// preview needs it to route delayed voices away from admission.
    pub(crate) delay_frames: Vec<u32>,
    /// Bitset over `region * 128 + key`, set when `build_voice` would return
    /// `Some` for that pair. Everything that can make it return `None` --
    /// a missing or empty sample, a pitch ratio that is not finite and
    /// positive -- is a function of the region and the key alone.
    pub(crate) key_ok: Vec<u64>,
}

/// One layer a note-on would produce, as far as admission needs to know it.
/// Produced by `preview_note_on`, which costs two table lookups per layer
/// against `build_voice`'s pitch math and address arithmetic.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreviewLayer {
    pub region: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// `VoiceSpawn::delay_frames`. Admission never sees a delayed voice -- it
    /// goes on the deferred queue and is ranked in the block it lands in -- so
    /// the preview has to answer this before ranking, not after.
    pub delay_frames: u32,
}

/// Everything the device needs to start one voice. Produced by `note_on`.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoiceSpawn {
    pub phase: Fixed,
    pub step: Fixed,
    pub smp_base: u32,
    pub smp_len: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub flags: u32,
    pub params: u32,
    pub region: u32,
    pub gain_l: f32,
    pub gain_r: f32,
    /// Frames to wait before the voice starts, from SF2 delayVolEnv.
    pub delay_frames: u32,
    pub exclusive_class: u8,
}

#[inline]
pub fn timecents_to_secs(tc: f32) -> f32 {
    // -32768 is the SF2 idiom for "instant".
    if tc <= -12000.0 {
        0.0
    } else {
        (2.0f32).powf(tc / 1200.0)
    }
}

#[inline]
pub fn cents_to_hz(cents: f32) -> f32 {
    8.176 * (2.0f32).powf(cents / 1200.0)
}

#[inline]
pub fn cb_to_gain(cb: f32) -> f32 {
    (10.0f32).powf(-cb / 200.0)
}

/// Velocity to attenuation, in centibels, scaled by SFZ `amp_veltrack`.
///
/// At the default `veltrack` of 100 the amplitude tracks (vel/127)^2, which is
/// the usual reading of the SF2 default velocity-to-initial-attenuation
/// modulator and matches what FluidSynth does closely enough to be
/// indistinguishable in a listening test.
///
/// `veltrack` interpolates that curve against unity gain:
///
/// ```text
/// gain = 1 - veltrack/100 * (1 - (vel/127)^2)
/// ```
///
/// so 0 leaves every note at full amplitude and 100 is the curve above. A
/// library that splits by velocity has already spent velocity on choosing the
/// layer and sets `amp_veltrack=0` so it is not spent twice; applying the
/// curve regardless is what made its quiet layers quieter than intended.
///
/// Negative values invert instead of reflecting, because reflecting would push
/// the gain above unity and no library means that by `amp_veltrack=-100`.
#[inline]
pub fn velocity_atten_cb(vel: u8, veltrack: f32) -> f32 {
    let v = vel.max(1) as f32 / 127.0;
    let t = veltrack.clamp(-100.0, 100.0) / 100.0;
    if t == 1.0 {
        // Full tracking is the SF2 path and the SFZ default, so it keeps the
        // original expression rather than the equivalent one below. They agree
        // mathematically but not to the last bit of an f32, and every render
        // made before `amp_veltrack` existed went through this line.
        return (-400.0 * v.log10()).clamp(0.0, 960.0);
    }
    let gain = if t >= 0.0 {
        1.0 - t * (1.0 - v * v)
    } else {
        1.0 + t * v * v
    };
    if gain <= 0.0 {
        return 960.0;
    }
    (-200.0 * gain.log10()).clamp(0.0, 960.0)
}

impl Bank {
    pub fn finish(&mut self) {
        for p in &mut self.presets {
            p.build_key_index(&self.regions);
        }
        self.index = self
            .presets
            .iter()
            .enumerate()
            .map(|(i, p)| ((p.bank, p.program), i as u32))
            .collect();
        self.index.sort_by_key(|e| e.0);
    }

    /// Resolve a (bank, program) pair, falling back the way hardware does:
    /// same program in bank 0, then anything at all.
    pub fn find_preset(&self, bank: u16, program: u16) -> Option<u32> {
        if let Ok(i) = self.index.binary_search_by_key(&(bank, program), |e| e.0) {
            return Some(self.index[i].1);
        }
        if let Ok(i) = self.index.binary_search_by_key(&(0, program), |e| e.0) {
            return Some(self.index[i].1);
        }
        // Drum banks conventionally live at 128; if a drum program is missing,
        // fall back to the first drum preset rather than to a piano.
        if bank == 128 {
            if let Some(e) = self.index.iter().find(|e| e.0 .0 == 128) {
                return Some(e.1);
            }
        }
        self.index.first().map(|e| e.1)
    }

    /// Build the voices for one note-on. Pushes into `out` rather than
    /// allocating, because this runs once per note in a billion-note file.
    pub fn note_on(
        &self,
        preset: u32,
        key: u8,
        vel: u8,
        cfg: &Config,
        max_layers: usize,
        out: &mut Vec<VoiceSpawn>,
    ) {
        let Some(p) = self.presets.get(preset as usize) else {
            return;
        };
        let key = key.min(127);
        let lo = p.key_index[key as usize] as usize;
        let hi = p.key_index[key as usize + 1] as usize;
        let mut layers = 0usize;

        for &ri in &p.key_regions[lo..hi] {
            if layers >= max_layers {
                break;
            }
            let r = &self.regions[ri as usize];
            if vel < r.vel_lo || vel > r.vel_hi {
                continue;
            }
            if let Some(v) = self.build_voice(r, ri, key, vel, cfg) {
                out.push(v);
                layers += 1;
            }
        }
    }

    /// Build the one layer `preview_note_on` named, by its region index.
    ///
    /// The preview has already decided this layer exists, so this is the other
    /// half of the split: admission ranks on the preview, and only the notes it
    /// admits ever reach here.
    pub fn build_layer(
        &self,
        region: u32,
        key: u8,
        vel: u8,
        cfg: &Config,
    ) -> Option<VoiceSpawn> {
        let r = self.regions.get(region as usize)?;
        self.build_voice(r, region, key.min(127), vel, cfg)
    }

    fn build_voice(
        &self,
        r: &Region,
        region_idx: u32,
        key: u8,
        vel: u8,
        cfg: &Config,
    ) -> Option<VoiceSpawn> {
        let s = self.samples.get(r.sample as usize)?;
        if s.len == 0 {
            return None;
        }

        let eff_key = if r.fixed_key >= 0 { r.fixed_key as u8 } else { key };
        let eff_vel = if r.fixed_vel >= 0 { r.fixed_vel as u8 } else { vel };

        let root = if r.root_key_override >= 0 {
            r.root_key_override as f64
        } else {
            s.root_key as f64
        };

        // Pitch, computed in f64 and only then frozen to fixed point.
        let cents = (eff_key as f64 - root) * r.scale_tuning as f64
            + r.coarse_tune as f64 * 100.0
            + r.fine_tune as f64
            + s.correction_cents as f64;
        let mut ratio = (cents / 1200.0).exp2();

        // When the pool was not rebuilt at a single rate, the per-sample rate
        // ratio rides along in the phase step. Lossless, just less uniform.
        if self.pool_rate == 0 {
            ratio *= s.rate as f64 / cfg.sample_rate as f64;
        } else {
            ratio *= self.pool_rate as f64 / cfg.sample_rate as f64;
        }
        if !(ratio.is_finite() && ratio > 0.0) {
            return None;
        }

        let atten = r.attenuation_cb + velocity_atten_cb(eff_vel, r.amp_veltrack);
        let gain = cb_to_gain(atten) * cfg.master_volume;

        // Constant-power pan.
        let theta = (r.pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * (PI as f32 * 0.5);
        let gain_l = gain * theta.cos();
        let gain_r = gain * theta.sin();

        // Address offsets are quoted in source frames; scale them if the pool
        // was rebuilt at a different rate.
        let scale = |v: i32| (v as f32 * s.resample_ratio).round() as i64;
        let start_offset = scale(r.addr_start).clamp(0, s.len as i64 - 1) as u32;
        let smp_len = (s.len as i64 + scale(r.addr_end)).clamp(1, s.len as i64) as u32;
        let loop_start = (s.loop_start as i64 + scale(r.addr_loop_start))
            .clamp(0, smp_len as i64 - 1) as u32;
        let loop_end =
            (s.loop_end as i64 + scale(r.addr_loop_end)).clamp(0, smp_len as i64) as u32;

        // A region's params block is a (velocity, key) grid, either dimension
        // collapsed to one entry when nothing varies along it.
        let keys = if r.params_stride != 0 { 128u32 } else { 1 };
        let ki = if r.params_stride != 0 { eff_key as u32 } else { 0 };
        let vi = if r.params_vel_span > 1 {
            (vel.saturating_sub(r.vel_lo) as u32).min(r.params_vel_span - 1)
        } else {
            0
        };
        let params = r.params_base + vi * keys + ki;

        let mut flags = r.loop_mode.flags();
        // A loop that does not describe at least two frames is not a loop.
        if r.loop_mode != LoopMode::NoLoop && loop_end <= loop_start + 1 {
            flags = 0;
        }

        Some(VoiceSpawn {
            phase: Fixed::from_int(start_offset),
            step: Fixed::from_f64(ratio),
            smp_base: s.start,
            smp_len,
            loop_start,
            loop_end,
            flags,
            params,
            region: region_idx,
            gain_l,
            gain_r,
            delay_frames: (r.delay * cfg.sample_rate as f32) as u32,
            exclusive_class: r.exclusive_class,
        })
    }

    /// Fill `params` from `regions`. Call once after all regions exist.
    pub fn build_params(&mut self, cfg: &Config) {
        // Lay out the table first: where each region's block starts, and how
        // far it runs along key and velocity. Every variant repeats this exact
        // layout, which is what lets a voice's stored index stay valid no
        // matter which copy its channel is reading.
        let mut n = 0u32;
        for r in &mut self.regions {
            let per_key = r.keynum_to_decay != 0 || r.keynum_to_hold != 0;
            // Only regions that actually track velocity are expanded along it,
            // and only across their own velocity range. A library that already
            // splits by velocity, as sampled pianos do, gets one entry per
            // region.
            let vel_span = if r.filter_veltrack_cents != 0.0 {
                (r.vel_hi.saturating_sub(r.vel_lo) as u32 + 1).min(128)
            } else {
                1
            };
            r.params_base = n;
            r.params_stride = if per_key { 1 } else { 0 };
            r.params_vel_span = vel_span;
            n += vel_span * if per_key { 128 } else { 1 };
        }
        self.params = self.build_variant(cfg, &ParamMod::default());
        self.build_admission_tables(cfg);
    }

    /// Precompute what admission needs to rank a note-on without building it.
    ///
    /// The whole point of culling is that a block may contain a hundred times
    /// more note-ons than the pool can take, so ranking them must not cost a
    /// `build_voice` each. Both tables are exact rather than approximate --
    /// `preview_note_on` returns the same layers with the same gains that
    /// `note_on` would -- which is what lets admission move earlier without
    /// changing which notes are admitted.
    fn build_admission_tables(&mut self, cfg: &Config) {
        let n = self.regions.len();
        self.gain_table = vec![[0.0, 0.0]; n * 128];
        self.key_ok = vec![0u64; (n * 128).div_ceil(64)];
        self.delay_frames = self
            .regions
            .iter()
            .map(|r| (r.delay * cfg.sample_rate as f32) as u32)
            .collect();

        for (ri, r) in self.regions.iter().enumerate() {
            // Gain: the same arithmetic as `build_voice`, in the same order, so
            // the f32 results are bit-identical rather than merely close.
            let theta = (r.pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * (PI as f32 * 0.5);
            let (pan_l, pan_r) = (theta.cos(), theta.sin());
            for v in 0..128u8 {
                let eff_vel = if r.fixed_vel >= 0 { r.fixed_vel as u8 } else { v };
                let atten = r.attenuation_cb + velocity_atten_cb(eff_vel, r.amp_veltrack);
                let gain = cb_to_gain(atten) * cfg.master_volume;
                self.gain_table[ri * 128 + v as usize] = [gain * pan_l, gain * pan_r];
            }

            // Existence: `build_voice` gives up on a missing or empty sample and
            // on a pitch ratio that is not finite and positive. The sample is
            // fixed per region and the ratio depends on the key but never on
            // the velocity, so one bit per (region, key) covers every case.
            let Some(s) = self.samples.get(r.sample as usize) else {
                continue;
            };
            if s.len == 0 {
                continue;
            }
            for k in 0..128u8 {
                let eff_key = if r.fixed_key >= 0 { r.fixed_key as u8 } else { k };
                let root = if r.root_key_override >= 0 {
                    r.root_key_override as f64
                } else {
                    s.root_key as f64
                };
                let cents = (eff_key as f64 - root) * r.scale_tuning as f64
                    + r.coarse_tune as f64 * 100.0
                    + r.fine_tune as f64
                    + s.correction_cents as f64;
                let mut ratio = (cents / 1200.0).exp2();
                if self.pool_rate == 0 {
                    ratio *= s.rate as f64 / cfg.sample_rate as f64;
                } else {
                    ratio *= self.pool_rate as f64 / cfg.sample_rate as f64;
                }
                if ratio.is_finite() && ratio > 0.0 {
                    let bit = ri * 128 + k as usize;
                    self.key_ok[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
    }

    #[inline]
    fn region_key_ok(&self, region: u32, key: u8) -> bool {
        let bit = region as usize * 128 + key as usize;
        self.key_ok[bit / 64] >> (bit % 64) & 1 != 0
    }

    /// The layers `note_on` would produce, and each one's opening gain, without
    /// building any of them.
    ///
    /// Iterates exactly as `note_on` does -- same preset key index, same
    /// velocity window, same `max_layers` cut -- so the `i`-th preview entry
    /// corresponds to the `i`-th voice, which is what lets note ids be handed
    /// out here and honoured later.
    pub fn preview_note_on(
        &self,
        preset: u32,
        key: u8,
        vel: u8,
        max_layers: usize,
        out: &mut Vec<PreviewLayer>,
    ) {
        let Some(p) = self.presets.get(preset as usize) else {
            return;
        };
        let key = key.min(127);
        let lo = p.key_index[key as usize] as usize;
        let hi = p.key_index[key as usize + 1] as usize;
        let mut layers = 0usize;

        for &ri in &p.key_regions[lo..hi] {
            if layers >= max_layers {
                break;
            }
            let r = &self.regions[ri as usize];
            if vel < r.vel_lo || vel > r.vel_hi {
                continue;
            }
            if !self.region_key_ok(ri, key) {
                continue;
            }
            let g = self.gain_table[ri as usize * 128 + vel as usize];
            out.push(PreviewLayer {
                region: ri,
                gain_l: g[0],
                gain_r: g[1],
                delay_frames: self.delay_frames[ri as usize],
            });
            layers += 1;
        }
    }

    /// Build one copy of the params table with a channel's sound controllers
    /// applied. Identical in layout to variant zero, so the channel only has
    /// to supply which copy to read. See `ParamMod`.
    ///
    /// Call only after `build_params` has assigned the region offsets.
    pub fn build_variant(&self, cfg: &Config, m: &ParamMod) -> Vec<RegionParams> {
        let sr = cfg.sample_rate as f32;
        let mut params = Vec::with_capacity(self.params.len());
        for r in &self.regions {
            let per_key = r.params_stride != 0;
            for v in 0..r.params_vel_span {
                let vel = (r.vel_lo as u32 + v).min(127) as u8;
                if per_key {
                    for k in 0..128u32 {
                        params.push(make_params(r, k as u8, vel, sr, cfg, m));
                    }
                } else {
                    params.push(make_params(r, 60, vel, sr, cfg, m));
                }
            }
        }
        params
    }

    pub fn pool_bytes(&self) -> u64 {
        self.pool.len() as u64 * 2
    }

    pub fn describe(&self) -> String {
        format!(
            "{}: {} presets, {} regions, {} samples, {:.1} MiB pool @ {} Hz",
            self.name,
            self.presets.len(),
            self.regions.len(),
            self.samples.len(),
            self.pool_bytes() as f64 / (1024.0 * 1024.0),
            if self.pool_rate == 0 {
                "mixed".to_string()
            } else {
                self.pool_rate.to_string()
            }
        )
    }
}

fn make_params(r: &Region, key: u8, vel: u8, sr: f32, cfg: &Config, m: &ParamMod) -> RegionParams {
    // SF2 key scaling is expressed in timecents per key, relative to key 60.
    let key_delta = 60.0 - key as f32;
    let hold = r.hold * (2.0f32).powf(r.keynum_to_hold as f32 * key_delta / 1200.0);
    let decay = r.decay * (2.0f32).powf(r.keynum_to_decay as f32 * key_delta / 1200.0);

    // A controller may not fade faster than `CC_MIN_FADE_SECS`, but it may
    // never lengthen a region that already asked for less than that either --
    // so the floor is the smaller of the two. Attack has no such floor:
    // BASSMIDI takes a scaled-down attack all the way to instant.
    let floor = |t: f32, base: f32| t.max(CC_MIN_FADE_SECS.min(base));

    let attack_frames = (m.attack.apply(r.attack) * sr).max(0.0);
    let hold_frames = (hold * sr).max(0.0);
    let decay_frames = (floor(m.decay.apply(decay), decay) * sr).max(1.0);
    let release_frames = (floor(m.release.apply(r.release), r.release) * sr).max(1.0);

    let attack_rate = if attack_frames < 1.0 { 1.0 } else { 1.0 / attack_frames };
    let attack_end = 1.0 + hold_frames * attack_rate;

    let sustain = r.sustain.clamp(0.0, 1.0);
    let floor = cfg.env_floor;

    // Both curves are defined by the SF2 convention that the quoted time is
    // how long a full-scale envelope takes to fall to -100 dB.
    let (decay_coef, release_coef) = match cfg.decay_curve {
        EnvelopeCurve::Exponential => (
            (10.0f32).powf(-100.0 / (20.0 * decay_frames)),
            match cfg.release_curve {
                EnvelopeCurve::Exponential => (10.0f32).powf(-100.0 / (20.0 * release_frames)),
                EnvelopeCurve::Linear => 1.0 / release_frames,
            },
        ),
        EnvelopeCurve::Linear => (
            1.0 / decay_frames,
            match cfg.release_curve {
                EnvelopeCurve::Exponential => (10.0f32).powf(-100.0 / (20.0 * release_frames)),
                EnvelopeCurve::Linear => 1.0 / release_frames,
            },
        ),
    };

    // Velocity opens the cutoff linearly in the normalised velocity, which is
    // the SFZ 1.0 reading of `fil_veltrack`. Implementations differ here --
    // some square the velocity first -- and the difference is audible, so it
    // is worth saying which one this is.
    let fc_cents = r.filter_fc_cents + r.filter_veltrack_cents * (vel as f32 / 127.0)
        + m.cutoff_cents;
    let fc = cents_to_hz(fc_cents);
    let nyq_guard = sr * 0.49;
    // 13500 cents is the SF2 default and means "this preset did not ask for a
    // filter". Running a biquad there would colour every voice for nothing and
    // cost a multiply-add per voice per frame on the device. A veltracked
    // cutoff can climb past it, which is the same statement: at full velocity
    // a sampled piano library here asks for 22 kHz, and the right answer is no filter
    // rather than one sitting on Nyquist.
    let use_filter = cfg.filter_enabled && fc_cents < 13500.0 && fc < nyq_guard && fc > 20.0;

    let (b0, b1, a1, a2) = if use_filter {
        biquad_lowpass(fc, r.filter_q_cb + m.q_cb, sr)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };

    RegionParams {
        attack_rate,
        attack_end,
        decay_coef,
        decay_target: sustain.max(floor),
        sustain,
        release_coef,
        b0,
        b1,
        a1,
        a2,
        flags: if use_filter { RP_FILTER } else { 0 },
        _pad: 0,
    }
}

/// RBJ low-pass, normalised so resonance does not raise the passband level.
/// `b2 == b0`, so it is not returned.
///
/// SF2 quotes resonance in centibels above the DC gain. FluidSynth compensates
/// with `1/sqrt(q)`, which leaves a +1.5 dB passband boost at the default Q;
/// normalising against the default Q instead keeps an unresonant filter at
/// unity, so enabling the filter never shifts the mix level on its own.
pub fn biquad_lowpass(fc: f32, q_cb: f32, sr: f32) -> (f32, f32, f32, f32) {
    const Q_DEFAULT: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let q_db = q_cb / 10.0 - 3.01;
    let q = (10.0f32).powf(q_db / 20.0).max(0.001);
    let gain = (Q_DEFAULT / q).sqrt();

    let w0 = 2.0 * std::f32::consts::PI * (fc / sr).clamp(1.0e-5, 0.49);
    let (sin_w0, cos_w0) = w0.sin_cos();
    let alpha = sin_w0 / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = gain * (1.0 - cos_w0) * 0.5 / a0;
    let b1 = gain * (1.0 - cos_w0) / a0;
    let a1 = -2.0 * cos_w0 / a0;
    let a2 = (1.0 - alpha) / a0;
    (b0, b1, a1, a2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_curve_is_sane() {
        assert!(velocity_atten_cb(127, 100.0) < 0.01);
        // Half velocity should be around -12 dB, not -48 and not -3.
        let cb = velocity_atten_cb(64, 100.0);
        assert!((100.0..140.0).contains(&cb), "vel 64 gave {cb} cB");
        assert!(velocity_atten_cb(1, 100.0) >= 800.0);
    }

    #[test]
    fn amp_veltrack_scales_the_velocity_curve() {
        // 0 means velocity does not touch the amplitude at all.
        for vel in [1u8, 40, 64, 100, 127] {
            assert_eq!(velocity_atten_cb(vel, 0.0), 0.0, "vel {vel} at veltrack 0");
        }
        // Half tracking sits between no tracking and full, at every velocity.
        for vel in [1u8, 40, 64, 100] {
            let half = velocity_atten_cb(vel, 50.0);
            assert!(
                half > 0.0 && half < velocity_atten_cb(vel, 100.0),
                "vel {vel} gave {half} cB at veltrack 50"
            );
        }
        // Full velocity is unattenuated whatever the tracking.
        assert!(velocity_atten_cb(127, 50.0) < 0.01);
        // Negative inverts: loud notes are the quiet ones, and nothing is
        // ever amplified above unity.
        assert!(velocity_atten_cb(127, -100.0) >= 960.0);
        assert!(velocity_atten_cb(1, -100.0) < 0.01);
    }

    #[test]
    fn lowpass_is_unity_at_dc() {
        let (b0, b1, a1, a2) = biquad_lowpass(1000.0, 0.0, 48000.0);
        // H(1) = (b0 + b1 + b2) / (1 + a1 + a2), with b2 == b0.
        let h = (2.0 * b0 + b1) / (1.0 + a1 + a2);
        assert!((h - 1.0).abs() < 0.02, "dc gain was {h}");
        // Resonance must not raise the passband: at +12 dB of Q the DC gain
        // should come down, not up.
        let (b0, b1, a1, a2) = biquad_lowpass(1000.0, 120.0, 48000.0);
        let h_res = (2.0 * b0 + b1) / (1.0 + a1 + a2);
        assert!(h_res < h, "resonant filter raised dc gain to {h_res}");
    }

    #[test]
    fn timecents_round_trip() {
        assert_eq!(timecents_to_secs(-12000.0), 0.0);
        assert!((timecents_to_secs(0.0) - 1.0).abs() < 1e-6);
        assert!((timecents_to_secs(1200.0) - 2.0).abs() < 1e-5);
    }
}
