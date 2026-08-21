//! CPU reference synthesizer.
//!
//! Single-threaded, scalar, no clever tricks. This is the ground truth every
//! GPU phase is measured against, so it is written to be obviously correct
//! rather than fast. The arithmetic is deliberately f32 in exactly the places
//! the shader uses f32, and the order of operations matches
//! `shaders/render.wgsl` statement for statement.
//!
//! The one deliberate difference: the mixdown accumulates in f64 here. That
//! makes the reference more accurate than the device, so a null test measures
//! the device's error rather than the sum of both.

use crate::backend::{Backend, BlockStats};
use crate::bank::{Bank, RegionParams, RP_FILTER, VF_LOOP, VF_LOOP_UNTIL_RELEASE};
use crate::config::{Config, EnvelopeCurve, Interpolation, StealRule};
use crate::fixed::{Fixed, FRAC_SCALE_F32};
use crate::voice::*;
use anyhow::Result;
use std::sync::Arc;

/// Structure of arrays, one Vec per field. `phase` and `step` are kept as u64
/// here rather than split into hi/lo lanes; that split only exists on the
/// device because WGSL has no portable u64.
#[derive(Default)]
struct Pool {
    phase: Vec<Fixed>,
    step: Vec<Fixed>,
    smp_base: Vec<u32>,
    smp_len: Vec<u32>,
    loop_start: Vec<u32>,
    loop_end: Vec<u32>,
    flags: Vec<u32>,
    env_stage: Vec<u32>,
    env_level: Vec<f32>,
    gain_l: Vec<f32>,
    gain_r: Vec<f32>,
    filt_z1: Vec<f32>,
    filt_z2: Vec<f32>,
    params: Vec<u32>,
    region: Vec<u32>,
    gate_slot: Vec<u32>,
    ordinal: Vec<u32>,
    start_rel: Vec<u32>,
    /// Variant the voice was born under, plus one, or zero once it has
    /// outlived the block it was born in. Mirrors `F_BORN_VARIANT` on the
    /// device; see `SpawnCmd::variant`.
    born_variant: Vec<u32>,
    /// Frame in this block at which a stolen voice starts fading, plus one.
    /// Zero when the voice is not being stolen. Mirrors `F_STOP_REL`.
    stop_rel: Vec<u32>,
    note_id: Vec<u64>,
}

impl Pool {
    fn len(&self) -> usize {
        self.phase.len()
    }

    fn push(&mut self, c: &SpawnCmd) {
        self.phase.push(Fixed::from_parts(c.phase_hi, c.phase_lo));
        self.step.push(Fixed::from_parts(c.step_hi, c.step_lo));
        self.smp_base.push(c.smp_base);
        self.smp_len.push(c.smp_len);
        self.loop_start.push(c.loop_start);
        self.loop_end.push(c.loop_end);
        self.flags.push(c.flags);
        self.env_stage.push(ENV_ATTACK);
        self.env_level.push(0.0);
        self.gain_l.push(c.gain_l);
        self.gain_r.push(c.gain_r);
        self.filt_z1.push(0.0);
        self.filt_z2.push(0.0);
        self.params.push(c.params);
        self.region.push(c.region);
        self.gate_slot.push(c.gate_slot);
        self.ordinal.push(c.ordinal);
        self.start_rel.push(c.start_rel);
        self.born_variant.push(c.variant + 1);
        self.stop_rel.push(0);
        self.note_id
            .push(((c.note_id_hi as u64) << 32) | c.note_id_lo as u64);
    }

    fn swap_remove_compact(&mut self, keep: &[bool]) {
        // Stable compaction: order is preserved so behaviour does not depend
        // on which voices happened to die.
        let mut w = 0usize;
        for (r, &alive) in keep.iter().enumerate().take(self.len()) {
            if alive {
                if w != r {
                    self.phase[w] = self.phase[r];
                    self.step[w] = self.step[r];
                    self.smp_base[w] = self.smp_base[r];
                    self.smp_len[w] = self.smp_len[r];
                    self.loop_start[w] = self.loop_start[r];
                    self.loop_end[w] = self.loop_end[r];
                    self.flags[w] = self.flags[r];
                    self.env_stage[w] = self.env_stage[r];
                    self.env_level[w] = self.env_level[r];
                    self.gain_l[w] = self.gain_l[r];
                    self.gain_r[w] = self.gain_r[r];
                    self.filt_z1[w] = self.filt_z1[r];
                    self.filt_z2[w] = self.filt_z2[r];
                    self.params[w] = self.params[r];
                    self.region[w] = self.region[r];
                    self.gate_slot[w] = self.gate_slot[r];
                    self.ordinal[w] = self.ordinal[r];
                    self.start_rel[w] = self.start_rel[r];
                    self.born_variant[w] = self.born_variant[r];
                    self.stop_rel[w] = self.stop_rel[r];
                    self.note_id[w] = self.note_id[r];
                }
                w += 1;
            }
        }
        self.truncate(w);
    }

    fn truncate(&mut self, n: usize) {
        self.phase.truncate(n);
        self.step.truncate(n);
        self.smp_base.truncate(n);
        self.smp_len.truncate(n);
        self.loop_start.truncate(n);
        self.loop_end.truncate(n);
        self.flags.truncate(n);
        self.env_stage.truncate(n);
        self.env_level.truncate(n);
        self.gain_l.truncate(n);
        self.gain_r.truncate(n);
        self.filt_z1.truncate(n);
        self.filt_z2.truncate(n);
        self.params.truncate(n);
        self.region.truncate(n);
        self.gate_slot.truncate(n);
        self.ordinal.truncate(n);
        self.start_rel.truncate(n);
        self.born_variant.truncate(n);
        self.stop_rel.truncate(n);
        self.note_id.truncate(n);
    }
}

pub struct CpuSynth {
    cfg: Config,
    bank: Arc<Bank>,
    pool: Pool,
    /// Interleaved f64 accumulator for one block.
    mix: Vec<f64>,
    gate_rows: Vec<u32>,
    chan_rows: Vec<u32>,
    /// Copies of the params table beyond the bank's own. Index 0 is the
    /// bank's, so it is never stored here; entry `i` holds variant `i + 1`.
    variants: Vec<Vec<RegionParams>>,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
    tiles: usize,
    tile_frames: usize,
    stolen: u64,
    dropped: u64,
    peak: f32,
}

impl CpuSynth {
    pub fn new(cfg: &Config, bank: Arc<Bank>) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        CpuSynth {
            cfg: cfg.clone(),
            bank,
            pool: Pool::default(),
            mix: vec![0.0; cfg.block_samples()],
            gate_rows: vec![0; tiles * GATE_SLOTS],
            chan_rows: vec![0; tiles * BEND_CHANNELS * CHAN_FIELDS],
            variants: Vec::new(),
            bend_active: false,
            gain_active: false,
            variant_active: false,
            tiles,
            tile_frames: cfg.gate_frames as usize,
            stolen: 0,
            dropped: 0,
            peak: 0.0,
        }
    }

    /// Read one pool sample, normalised the same way `unpack2x16snorm` does on
    /// the device: divide by 32767 and clamp, not divide by 32768.
    #[inline(always)]
    fn fetch(&self, base: u32, idx: u32) -> f32 {
        let i = (base + idx) as usize;
        let v = *self.bank.pool.get(i).unwrap_or(&0);
        (v as f32 * (1.0 / 32767.0)).max(-1.0)
    }

    /// The 64-bit key voice stealing selects the k smallest of. Mirrors
    /// `steal_key` in `shaders/common.wgsl` bit for bit; the two backends
    /// choosing different victims would not show up as an error, only as two
    /// renders that quietly disagree.
    #[inline(always)]
    fn steal_key(&self, i: usize) -> u64 {
        let id = self.pool.note_id[i];
        if self.cfg.steal_rule != StealRule::Quietest {
            return id;
        }
        let level = self.pool.env_level[i].clamp(0.0, 1.0);
        let q = (level * 65535.0) as u32 as u64;
        (q << 48) | (id & 0x0000_FFFF_FFFF_FFFF)
    }

    /// One region's DSP constants out of one copy of the params table.
    /// Variant zero is the bank's own. The fallback is for a variant that was
    /// never uploaded, which the driver does not produce -- it builds every
    /// variant it hands out before the spawn that references it.
    #[inline(always)]
    fn params_of(&self, variant: u32, params_base: usize) -> RegionParams {
        match variant.checked_sub(1) {
            None => self.bank.params[params_base],
            Some(i) => self
                .variants
                .get(i as usize)
                .and_then(|t| t.get(params_base))
                .copied()
                .unwrap_or(self.bank.params[params_base]),
        }
    }

    /// Index of the sample `off` frames after `idx`, honouring the loop.
    /// Mirrors `neighbour_index` in `shaders/common.wgsl` exactly.
    #[inline(always)]
    fn advance_index(idx: u32, off: i32, looping: bool, loop_start: u32, loop_end: u32, len: u32) -> u32 {
        let raw = idx as i64 + off as i64;
        if looping {
            let ls = loop_start as i64;
            let le = loop_end as i64;
            if raw >= le {
                let span = (le - ls).max(1);
                (ls + (raw - ls) % span) as u32
            } else if raw < 0 {
                0
            } else {
                raw as u32
            }
        } else {
            raw.clamp(0, len as i64 - 1) as u32
        }
    }

    #[allow(clippy::too_many_arguments)] // mirrors the shader's signature
    #[inline(always)]
    fn interpolate(
        &self,
        base: u32,
        idx: u32,
        frac: f32,
        looping: bool,
        loop_start: u32,
        loop_end: u32,
        len: u32,
    ) -> f32 {
        match self.cfg.interpolation {
            Interpolation::Nearest => self.fetch(base, idx),
            Interpolation::Linear => {
                let i1 = Self::advance_index(idx, 1, looping, loop_start, loop_end, len);
                let s0 = self.fetch(base, idx);
                let s1 = self.fetch(base, i1);
                s0 + (s1 - s0) * frac
            }
            Interpolation::Cubic => {
                let im1 = Self::advance_index(idx, -1, looping, loop_start, loop_end, len);
                let i1 = Self::advance_index(idx, 1, looping, loop_start, loop_end, len);
                let i2 = Self::advance_index(idx, 2, looping, loop_start, loop_end, len);
                let sm1 = self.fetch(base, im1);
                let s0 = self.fetch(base, idx);
                let s1 = self.fetch(base, i1);
                let s2 = self.fetch(base, i2);
                catmull_rom(sm1, s0, s1, s2, frac)
            }
        }
    }
}

/// Catmull-Rom, written in the same Horner form as the shader.
#[inline(always)]
pub fn catmull_rom(sm1: f32, s0: f32, s1: f32, s2: f32, t: f32) -> f32 {
    let a = -0.5 * sm1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * sm1 + 0.5 * s1;
    ((a * t + b) * t + c) * t + s0
}

impl Backend for CpuSynth {
    fn set_channels(&mut self, rows: &[u32], bend: bool, gain: bool, variant: bool) -> Result<()> {
        self.chan_rows.copy_from_slice(rows);
        self.bend_active = bend;
        self.gain_active = gain;
        self.variant_active = variant;
        Ok(())
    }

    fn set_params_variant(&mut self, index: u32, data: &[RegionParams]) -> Result<()> {
        if index == 0 {
            return Ok(());
        }
        let i = index as usize - 1;
        if self.variants.len() <= i {
            self.variants.resize(i + 1, Vec::new());
        }
        self.variants[i] = data.to_vec();
        Ok(())
    }

    fn set_gates(&mut self, rows: &[u32]) -> Result<()> {
        self.gate_rows.copy_from_slice(rows);
        Ok(())
    }

    fn spawn(&mut self, cmds: &[SpawnCmd]) -> Result<()> {
        let cap = self.cfg.max_voices as usize;
        let live = self.pool.len();
        let want = cmds.len();

        if live + want > cap {
            match self.cfg.steal_rule {
                StealRule::DropNew => {
                    let take = want.min(cap.saturating_sub(live));
                    self.dropped += (want - take) as u64;
                    for i in 0..take {
                        self.pool.push(&cmds[spawn_pick(i, want, take)]);
                    }
                    return Ok(());
                }
                StealRule::Oldest | StealRule::Quietest => {
                    // Pick the voices with the smallest steal key, which is the
                    // note id by age or the envelope level with the id beneath
                    // it by level. Either way a total order with no ties, so the
                    // victim set is a pure function of the input and never of
                    // scheduling.
                    let need = (live + want).saturating_sub(cap);
                    // Bounded so a block cannot replace the whole pool. See
                    // `Config::max_steal_percent`.
                    let need = need.min(live).min(self.cfg.max_steal() as usize);
                    if need > 0 {
                        let mut order: Vec<u32> = (0..live as u32).collect();
                        order.select_nth_unstable_by_key(need - 1, |&i| {
                            self.steal_key(i as usize)
                        });
                        // Scheduled, not removed: each victim goes on sounding
                        // until its own stop frame and fades out there, so the
                        // steal is spread across the block instead of landing
                        // on the boundary. The low word of the note id spreads
                        // them, victims being a contiguous range of ids.
                        let span = self.cfg.steal_span();
                        for &i in &order[..need] {
                            let id = self.pool.note_id[i as usize] as u32;
                            self.pool.stop_rel[i as usize] = id % span + 1;
                        }
                        self.stolen += need as u64;
                    }
                    // Room is what will be free once the victims have gone,
                    // not what is free now; the pool runs over `max_voices`
                    // until the end-of-block compaction, exactly as the device
                    // pool does inside its headroom.
                    let take = want.min(cap - (live - need));
                    self.dropped += (want - take) as u64;
                    for i in 0..take {
                        self.pool.push(&cmds[spawn_pick(i, want, take)]);
                    }
                    return Ok(());
                }
            }
        }

        for c in cmds {
            self.pool.push(c);
        }
        Ok(())
    }

    fn render(&mut self, out: &mut [f32]) -> Result<()> {
        let block = self.cfg.block_frames as usize;
        debug_assert_eq!(out.len(), block * 2);
        self.mix.iter_mut().for_each(|v| *v = 0.0);

        let exp_decay = self.cfg.decay_curve == EnvelopeCurve::Exponential;
        let exp_release = self.cfg.release_curve == EnvelopeCurve::Exponential;
        let floor = self.cfg.env_floor;
        let fade = self.cfg.steal_fade_frames as usize;
        let n = self.pool.len();

        for v in 0..n {
            let stage0 = self.pool.env_stage[v];
            if stage0 == ENV_DEAD {
                continue;
            }
            let params_base = self.pool.params[v] as usize;
            // Non-zero only for a voice born in this block, and then it is the
            // variant current at its own note-on plus one.
            let born_variant = self.pool.born_variant[v];
            let mut variant = born_variant.saturating_sub(1);
            let mut p: RegionParams = self.params_of(variant, params_base);
            let mut use_filter = p.flags & RP_FILTER != 0;

            let base = self.pool.smp_base[v];
            let len = self.pool.smp_len[v];
            let loop_start = self.pool.loop_start[v];
            let loop_end = self.pool.loop_end[v];
            let vflags = self.pool.flags[v];
            let loop_enabled = vflags & VF_LOOP != 0;
            let loop_until_release = vflags & VF_LOOP_UNTIL_RELEASE != 0;
            let gate_slot = self.pool.gate_slot[v] as usize;
            let ordinal = self.pool.ordinal[v];
            // The pool holds the note's own gains and its unbent step. The
            // effective ones fold in the channel's volume, pan and bend, all
            // refreshed once per gate tile.
            let base_gain_l = self.pool.gain_l[v];
            let base_gain_r = self.pool.gain_r[v];
            let mut gain_l = base_gain_l;
            let mut gain_r = base_gain_r;
            let base_step = self.pool.step[v];
            let channel = ChannelTable::channel_of(gate_slot as u32);
            let mut step = base_step;

            let mut phase = self.pool.phase[v];
            let mut stage = stage0;
            let mut level = self.pool.env_level[v];
            let mut z1 = self.pool.filt_z1[v];
            let mut z2 = self.pool.filt_z2[v];
            let start_rel = self.pool.start_rel[v] as usize;
            let stop_rel = self.pool.stop_rel[v] as usize;
            // The gate tile this voice starts in. Only meaningful while
            // `born_variant` says the voice was born in this block; after that
            // `start_rel` has been cleared and every tile is one it was alive
            // for.
            let born_tile = start_rel / self.tile_frames;

            'voice: for tile in 0..self.tiles {
                // Note-off gate, sampled once per gate tile.
                if stage < ENV_RELEASE
                    && self.gate_rows[tile * GATE_SLOTS + gate_slot] >= ordinal
                {
                    stage = ENV_RELEASE;
                    level = level.min(1.0);
                }

                if self.bend_active || self.gain_active || self.variant_active {
                    let ci = (tile * BEND_CHANNELS + channel) * CHAN_FIELDS;
                    if self.bend_active {
                        step = base_step.scale(self.chan_rows[ci + CHAN_BEND]);
                    }
                    if self.gain_active {
                        gain_l = base_gain_l * f32::from_bits(self.chan_rows[ci + CHAN_GAIN_L]);
                        gain_r = base_gain_r * f32::from_bits(self.chan_rows[ci + CHAN_GAIN_R]);
                    }
                    // Only on a change: re-reading a voice's DSP constants
                    // every gate tile would be a load per voice per tile, and
                    // CC71-CC75 move a few hundred times in a whole file.
                    if self.variant_active {
                        let want = self.chan_rows[ci + CHAN_VARIANT];
                        // A row holds the state at the start of its tile,
                        // which is older than a voice born inside that tile:
                        // the voice already carries the variant that was
                        // current at its own frame. So rows govern it only
                        // from the tile after the one it was born in.
                        let born_here = born_variant != 0 && tile <= born_tile;
                        if want != variant && !born_here {
                            variant = want;
                            p = self.params_of(variant, params_base);
                            // CC74 can move a cutoff past the point where the
                            // filter is worth running, so this is part of the
                            // reload rather than fixed at spawn.
                            use_filter = p.flags & RP_FILTER != 0;
                        }
                    }
                }

                let f0 = tile * self.tile_frames;
                for i in 0..self.tile_frames {
                    let f = f0 + i;
                    if f < start_rel {
                        continue;
                    }
                    if stage == ENV_DEAD {
                        break 'voice;
                    }

                    let looping =
                        loop_enabled && !(loop_until_release && stage >= ENV_RELEASE);

                    let idx = phase.hi();
                    if !looping && idx >= len {
                        stage = ENV_DEAD;
                        break 'voice;
                    }

                    // The envelope advances before the sample is scaled, not
                    // after. With an instant attack that puts the voice at
                    // full level on its very first frame, so a note with no
                    // envelope renders as exactly the sample.
                    match stage {
                        ENV_ATTACK => {
                            level += p.attack_rate;
                            if level >= p.attack_end {
                                stage = ENV_DECAY;
                                level = 1.0;
                            }
                        }
                        ENV_DECAY => {
                            level = if exp_decay {
                                level * p.decay_coef
                            } else {
                                level - p.decay_coef
                            };
                            if level <= p.decay_target {
                                if p.sustain <= floor {
                                    stage = ENV_DEAD;
                                    level = 0.0;
                                } else {
                                    stage = ENV_SUSTAIN;
                                    level = p.sustain;
                                }
                            }
                        }
                        ENV_SUSTAIN => {}
                        ENV_RELEASE => {
                            level = if exp_release {
                                level * p.release_coef
                            } else {
                                level - p.release_coef
                            };
                            if level <= floor {
                                stage = ENV_DEAD;
                                level = 0.0;
                            }
                        }
                        _ => {}
                    }

                    let s = self.interpolate(
                        base,
                        idx,
                        phase.lo() as f32 * FRAC_SCALE_F32,
                        looping,
                        loop_start,
                        loop_end,
                        len,
                    );

                    let g = level.min(1.0);
                    let x = s * g;

                    // Transposed direct form II. b2 == b0.
                    let y = if use_filter {
                        let y = p.b0 * x + z1;
                        z1 = p.b1 * x - p.a1 * y + z2;
                        z2 = p.b0 * x - p.a2 * y;
                        y
                    } else {
                        x
                    };

                    // A stolen voice fades to silence over `steal_fade_frames`
                    // from its own stop frame. After the filter, so the biquad
                    // keeps seeing the untapered signal.
                    let y = if stop_rel != 0 && f + 1 >= stop_rel {
                        let d = f + 1 - stop_rel;
                        if d >= fade {
                            stage = ENV_DEAD;
                            0.0
                        } else {
                            y * (1.0 - d as f32 / fade as f32)
                        }
                    } else {
                        y
                    };

                    self.mix[f * 2] += (y * gain_l) as f64;
                    self.mix[f * 2 + 1] += (y * gain_r) as f64;

                    // ---- advance the phase ----
                    phase = phase.wrapping_add(step);
                    if looping {
                        let hi = phase.hi();
                        if hi >= loop_end {
                            let span = (loop_end - loop_start).max(1);
                            let wrapped = loop_start + (hi - loop_start) % span;
                            phase = Fixed::from_parts(wrapped, phase.lo());
                        }
                    }
                }
            }

            self.pool.phase[v] = phase;
            self.pool.env_stage[v] = stage;
            self.pool.env_level[v] = level;
            self.pool.filt_z1[v] = z1;
            self.pool.filt_z2[v] = z2;
            self.pool.start_rel[v] = 0;
            self.pool.born_variant[v] = 0;
            self.pool.stop_rel[v] = 0;
        }

        // Reduce to the output block.
        let mut peak = 0.0f32;
        for (o, m) in out.iter_mut().zip(&self.mix).take(block * 2) {
            let v = *m as f32;
            *o = v;
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        self.peak = peak;

        // Compaction. Dead voices leave, order is preserved.
        let keep: Vec<bool> = self
            .pool
            .env_stage
            .iter()
            .map(|&s| s != ENV_DEAD)
            .collect();
        if keep.iter().any(|k| !k) {
            self.pool.swap_remove_compact(&keep);
        }

        Ok(())
    }

    fn stats(&self) -> BlockStats {
        BlockStats {
            active_voices: self.pool.len() as u64,
            stolen: self.stolen,
            dropped: self.dropped,
            peak: self.peak,
        }
    }

    fn name(&self) -> &'static str {
        "cpu-reference"
    }
}
