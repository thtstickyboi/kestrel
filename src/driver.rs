//! Block-at-a-time render driver.
//!
//! Owns the MIDI stream, the tempo clock, channel state and the gate table,
//! and feeds a `Backend` one block of spawn commands at a time. Both the CPU
//! reference and the GPU path go through this, which is what makes the null
//! test meaningful: the two backends see byte-identical input.

use crate::backend::Backend;
use crate::bank::{Bank, ParamMod, VoiceSpawn};
use crate::config::{AdmitRule, Config};
use crate::limiter::{clamp_block, Brickwall, Limiter, LimiterMode};
use crate::midi::{Event, MidiStream, TempoClock};
use crate::fixed::bend_factor;
use crate::voice::{ChannelTable, GateTable, SpawnCmd};
use anyhow::{bail, Result};
use std::path::Path;
use std::sync::Arc;

/// A selected RPN that is not one this synth acts on, including the null RPN
/// and anything reached through NRPN. Parked outside the 14-bit range so it
/// can never compare equal to RPN 0.
pub const RPN_NONE: u16 = 0x4000;

/// How far CC71-CC75 are quantised before they become a params variant.
///
/// Two steps of a controller, so 64 levels. CC74 spans +/-2400 cents across
/// its 128 values, which makes a quantised step about 76 cents of cutoff --
/// well under a semitone, and inaudible as a step. At the eight steps this
/// started with, a step was 300 cents and a move from CC74=64 to CC74=70
/// landed in the same bucket and did nothing at all.
pub const SOUND_CC_SHIFT: u32 = 1;

/// What this synth does with a controller number.
///
/// Every number from 0 to 127 has an entry, because the alternative -- a match
/// arm with a `_ => {}` at the end -- is how a soundfont ends up sounding
/// wrong for a session with nothing to show for it. `kestrel info` prints
/// this for the controllers a file actually sends, so "is this doing
/// anything" is answerable without reading source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcRole {
    /// Acted on.
    Applied,
    /// Recognised, and correctly does nothing to an offline render.
    Inert(&'static str),
    /// Would change what is heard, and is not implemented. The string says
    /// what it would take.
    Missing(&'static str),
}

/// Name and role for one controller number.
pub fn cc_role(num: u8) -> (&'static str, CcRole) {
    use CcRole::*;
    const NO_DEFAULT: CcRole =
        Inert("no default mapping in GM; a soundfont would have to bind it");
    const UNDEFINED: CcRole = Inert("undefined by the MIDI spec");
    const GENERAL: CcRole = Inert("general purpose, no defined destination");
    const LSB_UNUSED: CcRole = Inert("fine resolution for a controller that is not implemented");
    match num {
        0 => ("bank select msb", Applied),
        1 => ("modulation", Applied),
        2 => ("breath", NO_DEFAULT),
        3 => ("", UNDEFINED),
        4 => ("foot controller", NO_DEFAULT),
        5 => ("portamento time", Missing("portamento needs per-voice glide state")),
        6 => ("data entry msb", Applied),
        7 => ("channel volume", Applied),
        8 => ("balance", Inert("for two-source channels; a sampler has one")),
        9 => ("", UNDEFINED),
        10 => ("pan", Applied),
        11 => ("expression", Applied),
        12 | 13 => ("effect control", Missing("effects are not implemented")),
        14 | 15 => ("", UNDEFINED),
        16..=19 => ("general purpose", GENERAL),
        20..=31 => ("", UNDEFINED),
        32 => ("bank select lsb", Applied),
        33 => ("modulation lsb", Applied),
        38 => ("data entry lsb", Applied),
        39 => ("channel volume lsb", Applied),
        42 => ("pan lsb", Applied),
        43 => ("expression lsb", Applied),
        34..=63 => ("", LSB_UNUSED),
        64 => ("sustain pedal", Applied),
        65 => ("portamento", Missing("portamento needs per-voice glide state")),
        66 => ("sostenuto", Applied),
        67 => ("soft pedal", Applied),
        68 => ("legato footswitch", Missing("legato needs mono voice allocation")),
        69 => ("hold 2", Missing("a second hold that lengthens release rather than damping")),
        70 => ("sound variation", NO_DEFAULT),
        71 => ("resonance", Applied),
        72 => ("release time", Applied),
        73 => ("attack time", Applied),
        74 => ("brightness", Applied),
        75 => ("decay time", Applied),
        76 => ("vibrato rate", Applied),
        77 => ("vibrato depth", Applied),
        78 => ("vibrato delay", Missing("the LFO starts immediately")),
        79 => ("sound controller 10", NO_DEFAULT),
        80..=83 => ("general purpose", GENERAL),
        84 => ("portamento control", Missing("portamento needs per-voice glide state")),
        85..=87 => ("", UNDEFINED),
        88 => ("high resolution velocity", Missing("velocity is read as 7 bits")),
        89 | 90 => ("", UNDEFINED),
        91 => ("reverb send", Missing("effects are not implemented")),
        92 => ("tremolo depth", Applied),
        93 => ("chorus send", Missing("effects are not implemented")),
        94 => ("celeste depth", Missing("effects are not implemented")),
        95 => ("phaser depth", Missing("effects are not implemented")),
        96 => ("data increment", Applied),
        97 => ("data decrement", Applied),
        98 | 99 => ("nrpn select", Applied),
        100 | 101 => ("rpn select", Applied),
        102..=119 => ("", UNDEFINED),
        120 => ("all sound off", Applied),
        121 => ("reset controllers", Applied),
        122 => ("local control", Inert("there is no keyboard attached to an offline render")),
        123 => ("all notes off", Applied),
        124 => ("omni off", Applied),
        125 => ("omni on", Applied),
        126 => ("mono mode on", Applied),
        127 => ("poly mode on", Applied),
        _ => ("", UNDEFINED),
    }
}

/// Whether a controller changes anything about the render.
pub fn handles_cc(num: u8) -> bool {
    cc_role(num).1 == CcRole::Applied
}

/// How long to keep rendering after the last MIDI event before giving up on
/// voices that will not die, in seconds.
const MAX_TAIL_SECONDS: f64 = 60.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct DriverStats {
    /// Copies of the params table CC71-CC75 asked for and got.
    pub param_variants: u32,
    /// Times a sound-controller change had to reuse an approximate copy
    /// because the budget was spent. Any of these is audible as the
    /// controller appearing to do nothing.
    pub variant_fallbacks: u64,
    /// Distinct quantised states the file asked for, whether or not they
    /// were granted.
    pub variant_states: u32,
    /// Times a slot was reused for a different state. Correct, just work.
    pub variant_rebuilds: u64,
    pub frames: u64,
    pub blocks: u64,
    pub events: u64,
    pub notes: u64,
    pub voices_spawned: u64,
    pub peak: f32,
    /// Samples the final clamp had to pull back to full scale. Every run of
    /// these is a squared-off waveform top, which is what a click sounds like.
    pub clipped: u64,
}

pub struct Driver {
    cfg: Config,
    bank: Arc<Bank>,
    stream: MidiStream,
    clock: TempoClock,
    gates: GateTable,
    chan: ChannelTable,
    limiter: Limiter,
    brickwall: Brickwall,

    bank_msb: [u8; 16],
    bank_lsb: [u8; 16],
    preset: [u32; 16],
    /// Raw pitch bend, -8192..8191 relative to centre.
    bend_val: [i16; 16],
    /// Bend range in semitones, RPN 0. Two is the GM default.
    bend_range: [f64; 16],
    /// Currently selected RPN, from CC101 (MSB) and CC100 (LSB).
    rpn_sel: [u16; 16],
    /// CC7 channel volume, CC11 expression, CC10 pan.
    cc_volume: [u8; 16],
    cc_expression: [u8; 16],
    cc_pan: [u8; 16],
    /// CC71 resonance, CC72 release, CC73 attack, CC74 brightness, CC75 decay.
    /// Quantised to sixteen steps each before they become a params variant, so
    /// a slow sweep does not demand a new copy of the table per event.
    cc_sound: [[u8; 5]; 16],
    /// CC1 modulation depth, CC76 vibrato rate, CC77 vibrato depth, CC92
    /// tremolo depth, CC67 soft pedal.
    cc_mod: [u8; 16],
    cc_vib_rate: [u8; 16],
    cc_vib_depth: [u8; 16],
    cc_tremolo: [u8; 16],
    cc_soft: [u8; 16],
    /// Fine halves for the four controllers that have one worth reading. A
    /// file almost never sends them, but declaring a controller "applied" and
    /// then dropping half its resolution is the same silence this whole table
    /// exists to stop.
    lsb_mod: [u8; 16],
    lsb_volume: [u8; 16],
    lsb_pan: [u8; 16],
    lsb_expression: [u8; 16],
    /// LFO phase per channel, in cycles, carried across blocks so vibrato does
    /// not restart every 85 ms.
    lfo_phase: [f64; 16],
    /// Quantised sound-controller states, indexed by variant number.
    variants: Vec<[u8; 5]>,
    /// Every quantised state the file has asked for, for diagnostics.
    seen_states: Vec<[u8; 5]>,
    /// Which variant each channel is currently on.
    cur_variant: [u32; 16],
    /// Bitmask of variants any channel has pointed at during this block. A
    /// variant referenced by a row already written cannot be rebuilt under it.
    variant_used: u64,
    /// Monotonic tick per variant, for choosing the stalest one to reuse.
    variant_seen: Vec<u64>,
    variant_clock: u64,
    /// Variants asked for this block but not yet built and uploaded.
    pending_variants: Vec<(u32, ParamMod)>,
    /// Set when a voice queued this block was born under a non-zero variant.
    /// Its copy of the params table has to stay reachable even if no published
    /// row names it, which happens when a channel moves onto a variant and
    /// back off inside a single gate tile.
    spawn_variants: bool,

    next_note_id: u64,
    block_index: u64,
    /// Event pulled from the stream that belongs to a later block.
    pending: Option<(u64, Event)>,
    stream_done: bool,
    end_sent: bool,
    tail_blocks_left: u64,

    spawn_buf: Vec<SpawnCmd>,
    voice_buf: Vec<VoiceSpawn>,
    /// Voices whose SF2 delay pushes their start past this block.
    deferred: Vec<(u64, SpawnCmd)>,

    pub stats: DriverStats,
}

impl Driver {
    pub fn open(cfg: &Config, bank: Arc<Bank>, midi: impl AsRef<Path>) -> Result<Self> {
        cfg.validate()?;
        let stream = MidiStream::open(midi)?;
        let clock = TempoClock::new(stream.division, cfg.sample_rate);

        let mut d = Driver {
            cfg: cfg.clone(),
            bank,
            stream,
            clock,
            gates: GateTable::new(cfg),
            chan: ChannelTable::new(cfg),
            limiter: Limiter::new(cfg.sample_rate),
            brickwall: Brickwall::new(
                cfg.sample_rate,
                cfg.limiter_ceiling(),
                cfg.limiter_lookahead_ms,
                cfg.limiter_release_ms,
                cfg.limiter_sustain_ms,
                cfg.limiter_true_peak,
            ),
            bank_msb: [0; 16],
            bank_lsb: [0; 16],
            preset: [0; 16],
            bend_val: [0; 16],
            bend_range: [2.0; 16],
            rpn_sel: [0; 16],
            cc_volume: [cfg.default_channel_volume.min(127); 16],
            cc_expression: [127; 16],
            cc_pan: [64; 16],
            cc_sound: [[64; 5]; 16],
            cc_mod: [0; 16],
            cc_vib_rate: [64; 16],
            cc_vib_depth: [64; 16],
            cc_tremolo: [0; 16],
            cc_soft: [0; 16],
            lsb_mod: [0; 16],
            lsb_volume: [0; 16],
            lsb_pan: [0; 16],
            lsb_expression: [0; 16],
            lfo_phase: [0.0; 16],
            variants: vec![[64 >> SOUND_CC_SHIFT; 5]],
            seen_states: vec![[64 >> SOUND_CC_SHIFT; 5]],
            cur_variant: [0; 16],
            variant_used: 1,
            variant_seen: vec![0],
            variant_clock: 0,
            pending_variants: Vec::new(),
            spawn_variants: false,
            next_note_id: 1,
            block_index: 0,
            pending: None,
            stream_done: false,
            end_sent: false,
            tail_blocks_left: (MAX_TAIL_SECONDS * cfg.sample_rate as f64
                / cfg.block_frames as f64) as u64,
            spawn_buf: Vec::new(),
            voice_buf: Vec::new(),
            deferred: Vec::new(),
            stats: DriverStats::default(),
        };
        for ch in 0..16 {
            d.refresh_preset(ch, 0);
        }
        Ok(d)
    }

    pub fn track_count(&self) -> u16 {
        self.stream.track_count
    }

    fn refresh_preset(&mut self, ch: usize, program: u8) {
        // Channel 10 is percussion by GM convention, and SF2 files put their
        // drum kits in bank 128.
        let bank_num = if ch == 9 {
            128u16
        } else {
            self.bank_msb[ch] as u16
        };
        self.preset[ch] = self
            .bank
            .find_preset(bank_num, program as u16)
            .unwrap_or(0);
    }

    /// Render one block. Returns false once the file and its tail are done.
    pub fn next_block(&mut self, backend: &mut dyn Backend, out: &mut [f32]) -> Result<bool> {
        if out.len() != self.cfg.block_samples() {
            bail!(
                "output block is {} samples, expected {}",
                out.len(),
                self.cfg.block_samples()
            );
        }

        let block_frames = self.cfg.block_frames as u64;
        let block_start = self.block_index * block_frames;
        let block_end = block_start + block_frames;

        self.gates.begin_block();
        self.chan.begin_block();
        // Whatever each channel is already on stays referenced for the whole
        // block, because `end_block` fills every unwritten row with it.
        self.variant_used = 1;
        for v in self.cur_variant {
            self.variant_used |= 1u64 << v;
        }
        self.spawn_buf.clear();
        self.spawn_variants = false;

        // Voices whose delay landed them in this block.
        let mut i = 0;
        while i < self.deferred.len() {
            if self.deferred[i].0 < block_end {
                let (frame, mut cmd) = self.deferred.swap_remove(i);
                cmd.start_rel = frame.saturating_sub(block_start) as u32;
                self.queue_spawn(cmd);
            } else {
                i += 1;
            }
        }

        // Drain events belonging to this block.
        loop {
            let (tick, ev) = match self.pending.take() {
                Some(p) => p,
                None => match self.stream.next() {
                    Some(p) => p,
                    None => {
                        self.stream_done = true;
                        break;
                    }
                },
            };

            if let Event::Tempo(us) = ev {
                // Tempo applies from its own tick, so it must be consumed even
                // if it lands past the block boundary check below.
                self.clock.set_tempo(tick, us);
                self.stats.events += 1;
                continue;
            }

            let frame = self.clock.frame_at(tick);
            let frame = if frame < 0.0 { 0 } else { frame as u64 };
            if frame >= block_end {
                self.pending = Some((tick, ev));
                break;
            }
            let rel = frame.saturating_sub(block_start) as u32;
            let rel = rel.min(self.cfg.block_frames - 1);
            self.stats.events += 1;
            self.handle_event(ev, rel, block_start);
        }

        // Once the file runs out, release everything so looping voices stop.
        if self.stream_done && !self.end_sent {
            for ch in 0..16u8 {
                // Not `all_notes_off`: a file that ends with the pedal still
                // down would otherwise leave every held note un-released and
                // ringing into the tail until it timed out.
                self.gates.all_sound_off(ch, 0);
            }
            self.end_sent = true;
        }

        for (i, m) in std::mem::take(&mut self.pending_variants) {
            let data = self.bank.build_variant(&self.cfg, &m);
            backend.set_params_variant(i, &data)?;
        }

        self.gates.end_block();
        self.chan.end_block();
        self.apply_modulation();
        self.chan.refresh_active();
        backend.set_gates(&self.gates.rows)?;
        backend.set_channels(
            &self.chan.rows,
            self.chan.bend_active(),
            self.chan.gain_active(),
            // A voice born under a variant no published row names still has to
            // be handed back to the rows on its second gate tile, so the
            // controller path stays on for it.
            self.chan.variant_active() || self.spawn_variants,
        )?;
        // Rank the block before the backend thins it. Sorted here rather than
        // in each backend so the two cannot disagree, and after the whole
        // block's events are in so `released` sees every note-off that landed
        // inside it. A voice's position in time rides on `start_rel`, so
        // reordering the queue changes only which notes survive, never when
        // the survivors sound.
        if self.cfg.admit_rule == AdmitRule::Loudest && !self.spawn_buf.is_empty() {
            let live = backend.stats().active_voices as u32;
            let queued = self.spawn_buf.len();
            let take = self.cfg.admit_take(live, queued as u32) as usize;
            // A block that fits needs no ranking at all: nothing is dropped,
            // so the order it goes in is the order it came in.
            if take < queued {
                let gates = &self.gates;
                let key = |c: &SpawnCmd| {
                    std::cmp::Reverse(crate::voice::admit_key(
                        c,
                        !gates.released(c.gate_slot, c.ordinal),
                    ))
                };
                // Partition first, then sort only the part that survives.
                // Sorting the whole block is O(n log n) over every note-on in
                // it; a saturated block queues several times what it can
                // admit, so this is the difference between sorting 58k and
                // sorting 8k. The prefix is still fully sorted afterwards
                // because `select_nth_unstable` leaves it in an unspecified
                // order, and an unspecified order would decide which voice
                // lands in which pool slot -- and therefore the summation
                // order of the mixdown, and therefore the last bit of the
                // output.
                self.spawn_buf.select_nth_unstable_by_key(take, key);
                self.spawn_buf[..take].sort_unstable_by_key(key);
            }
        }
        backend.spawn(&self.spawn_buf)?;
        self.stats.voices_spawned += self.spawn_buf.len() as u64;
        backend.render(out)?;

        if self.cfg.limiter {
            match self.cfg.limiter_mode {
                LimiterMode::Off => {}
                LimiterMode::Omni => self.limiter.process(out),
                LimiterMode::Brickwall => self.brickwall.process(out),
            }
        }
        // Always last, so nothing leaves the renderer out of range whatever
        // ran before it. In brickwall mode it should never fire; the count is
        // reported so that "should" is checkable rather than assumed.
        self.stats.clipped += clamp_block(out);

        if self.cfg.nan_guard {
            if let Some(i) = out.iter().position(|v| !v.is_finite()) {
                bail!(
                    "block {} sample {} is {}; the synth produced a non-finite value",
                    self.block_index,
                    i,
                    out[i]
                );
            }
        }

        let st = backend.stats();
        self.stats.peak = self.stats.peak.max(st.peak);
        self.stats.frames += block_frames;
        self.stats.blocks += 1;
        self.block_index += 1;

        let more = if !self.stream_done || !self.deferred.is_empty() {
            true
        } else if st.active_voices > 0 {
            self.tail_blocks_left = self.tail_blocks_left.saturating_sub(1);
            if self.tail_blocks_left == 0 {
                log::warn!(
                    "tail exceeded {MAX_TAIL_SECONDS} s with {} voices still alive; stopping",
                    st.active_voices
                );
                false
            } else {
                true
            }
        } else {
            false
        };
        Ok(more)
    }

    /// Queue one voice for this block, pinning the copy of the params table it
    /// was born under. A voice carries its variant rather than reading it from
    /// the tile row, so that reference has to be honoured by `set_sound_cc`
    /// the same way a row's is -- a delayed voice can even be carrying one
    /// from an earlier block.
    fn queue_spawn(&mut self, cmd: SpawnCmd) {
        if cmd.variant != 0 {
            self.variant_used |= 1u64 << cmd.variant;
            self.spawn_variants = true;
        }
        self.spawn_buf.push(cmd);
    }

    fn handle_event(&mut self, ev: Event, rel: u32, block_start: u64) {
        match ev {
            Event::NoteOn { ch, key, vel } => {
                let ordinal = self.gates.note_on(ch, key, rel);
                let slot = GateTable::slot(ch, key) as u32;
                self.voice_buf.clear();
                self.bank.note_on(
                    self.preset[ch as usize & 15],
                    key,
                    vel,
                    &self.cfg,
                    self.cfg.max_layers as usize,
                    &mut self.voice_buf,
                );
                self.stats.notes += 1;
                for v in std::mem::take(&mut self.voice_buf) {
                    let id = self.next_note_id;
                    self.next_note_id += 1;
                    let cmd = SpawnCmd {
                        phase_lo: v.phase.lo(),
                        phase_hi: v.phase.hi(),
                        step_lo: v.step.lo(),
                        step_hi: v.step.hi(),
                        smp_base: v.smp_base,
                        smp_len: v.smp_len,
                        loop_start: v.loop_start,
                        loop_end: v.loop_end,
                        flags: v.flags,
                        params: v.params,
                        // Events are handled in stream order, so this is the
                        // variant in force at the note-on's own frame -- which
                        // the tile row does not yet carry if the controller
                        // arrived on this very tick.
                        variant: self.cur_variant[ch as usize & 15],
                        region: v.region,
                        gate_slot: slot,
                        ordinal,
                        start_rel: rel,
                        note_id_lo: id as u32,
                        note_id_hi: (id >> 32) as u32,
                        gain_l: v.gain_l,
                        gain_r: v.gain_r,
                    };
                    if v.delay_frames == 0 {
                        self.queue_spawn(cmd);
                    } else {
                        let start = block_start + rel as u64 + v.delay_frames as u64;
                        if start < block_start + self.cfg.block_frames as u64 {
                            let mut c = cmd;
                            c.start_rel = (start - block_start) as u32;
                            self.queue_spawn(c);
                        } else {
                            self.deferred.push((start, cmd));
                        }
                    }
                }
                // Put the buffer back so its allocation is reused.
                self.voice_buf = Vec::new();
            }
            Event::NoteOff { ch, key } => {
                self.gates.note_off(ch, key, rel);
            }
            Event::Cc { ch, num, val } => {
                let c = ch as usize & 15;
                match num {
                    0 => self.bank_msb[c] = val,
                    1 => {
                        self.cc_mod[c] = val;
                        self.lsb_mod[c] = 0;
                    }
                    33 => self.lsb_mod[c] = val,
                    39 => {
                        self.lsb_volume[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    42 => {
                        self.lsb_pan[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    43 => {
                        self.lsb_expression[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    32 => self.bank_lsb[c] = val,
                    // Data entry, and the increment/decrement pair that walks
                    // the same value. Only RPN 0, pitch bend sensitivity, is
                    // acted on; selecting an NRPN parks `rpn_sel` outside the
                    // 14-bit range so a file's NRPN data entry cannot be
                    // mistaken for a bend range.
                    6 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = val as f64;
                        self.refresh_bend(c, rel);
                    }
                    38 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = self.bend_range[c].trunc() + val as f64 / 100.0;
                        self.refresh_bend(c, rel);
                    }
                    96 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] + 1.0).min(127.0);
                        self.refresh_bend(c, rel);
                    }
                    97 if self.rpn_sel[c] == 0 => {
                        self.bend_range[c] = (self.bend_range[c] - 1.0).max(0.0);
                        self.refresh_bend(c, rel);
                    }
                    98 | 99 => self.rpn_sel[c] = RPN_NONE,
                    100 => {
                        let keep = if self.rpn_sel[c] == RPN_NONE { 0 } else { self.rpn_sel[c] };
                        self.rpn_sel[c] = (keep & 0x3F80) | val as u16;
                    }
                    101 => {
                        let keep = if self.rpn_sel[c] == RPN_NONE { 0 } else { self.rpn_sel[c] };
                        self.rpn_sel[c] = (keep & 0x7F) | ((val as u16) << 7);
                    }
                    7 => {
                        self.cc_volume[c] = val;
                        self.lsb_volume[c] = 0;
                        self.refresh_gain(c, rel);
                    }
                    10 => {
                        self.cc_pan[c] = val;
                        self.lsb_pan[c] = 0;
                        self.refresh_gain(c, rel);
                    }
                    11 => {
                        self.cc_expression[c] = val;
                        self.lsb_expression[c] = 0;
                        self.refresh_gain(c, rel);
                    }
                    64 => self.gates.set_sustain(ch, val >= 64, rel),
                    66 => self.gates.set_sostenuto(ch, val >= 64, rel),
                    67 => {
                        self.cc_soft[c] = val;
                        self.refresh_gain(c, rel);
                    }
                    71 => self.set_sound_cc(ch, 0, val, rel),
                    72 => self.set_sound_cc(ch, 1, val, rel),
                    73 => self.set_sound_cc(ch, 2, val, rel),
                    74 => self.set_sound_cc(ch, 3, val, rel),
                    75 => self.set_sound_cc(ch, 4, val, rel),
                    76 => self.cc_vib_rate[c] = val,
                    77 => self.cc_vib_depth[c] = val,
                    92 => self.cc_tremolo[c] = val,
                    120 => self.gates.all_sound_off(ch, rel),
                    121 => self.reset_controllers(ch, rel),
                    // Every channel mode message below carries an all-notes-off
                    // with it, which is the part that changes what is heard.
                    // The mode itself -- omni, and mono against poly voice
                    // allocation -- is not implemented; see the README.
                    123..=127 => self.gates.all_notes_off(ch, rel),
                    _ => {}
                }
            }
            Event::Program { ch, val } => {
                self.refresh_preset(ch as usize & 15, val);
            }
            Event::PitchBend { ch, val } => {
                self.bend_val[ch as usize & 15] = val;
                self.refresh_bend(ch as usize & 15, rel);
            }
            Event::Tempo(_) | Event::Other => {}
        }
    }

    /// Take one of CC71-CC75 and move the channel onto whichever copy of the
    /// params table matches its new combination, building that copy if this is
    /// the first time the combination has come up.
    ///
    /// Values are quantised to sixteen steps first. A file that sweeps CC74
    /// would otherwise ask for a new full rebuild of the table on every event,
    /// and the table can be tens of thousands of entries; sixteen steps across
    /// the +/-2400 cent range is 300 cents a step, which is coarse for a
    /// continuous sweep and exact for the discrete settings files actually
    /// use. Past `max_param_variants` a channel takes the nearest copy that
    /// already exists rather than failing or thrashing.
    fn set_sound_cc(&mut self, ch: u8, which: usize, val: u8, rel: u32) {
        let c = ch as usize & 15;
        if self.cc_sound[c][which] == val {
            return;
        }
        self.cc_sound[c][which] = val;

        let mut want = [0u8; 5];
        for (i, w) in want.iter_mut().enumerate() {
            *w = self.cc_sound[c][i] >> SOUND_CC_SHIFT;
        }
        if !self.seen_states.contains(&want) {
            self.seen_states.push(want);
            self.stats.variant_states = self.seen_states.len() as u32;
        }
        let cap = self.cfg.max_param_variants.clamp(1, 63);
        let idx = match self.variants.iter().position(|v| *v == want) {
            Some(i) => i as u32,
            None if (self.variants.len() as u32) < cap => {
                let i = self.variants.len() as u32;
                self.variants.push(want);
                self.variant_seen.push(0);
                self.stats.param_variants = i + 1;
                self.build_variant_at(i, c);
                i
            }
            // Out of slots. Rebuilding a stale one is much better than
            // approximating, because an approximation is silent: this file
            // asks for 67 distinct states, and approximating meant most of it
            // rendered with whatever the first sixteen happened to be.
            //
            // A slot is only safe to rebuild if no channel has pointed at it
            // during this block, since rows already written for earlier tiles
            // still name it and the device would read the new contents under
            // the old reference. Voices always read their channel's current
            // variant, so nothing older than this block can be holding one.
            None => {
                let victim = (1..self.variants.len())
                    .filter(|i| self.variant_used & (1u64 << i) == 0)
                    .min_by_key(|i| self.variant_seen[*i]);
                match victim {
                    Some(i) => {
                        self.variants[i] = want;
                        self.build_variant_at(i as u32, c);
                        self.stats.variant_rebuilds += 1;
                        i as u32
                    }
                    // Every slot is spoken for within this block already.
                    None => {
                        self.stats.variant_fallbacks += 1;
                        let mut best = 0u32;
                        let mut best_d = u32::MAX;
                        for (i, v) in self.variants.iter().enumerate() {
                            let d: u32 = v
                                .iter()
                                .zip(&want)
                                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
                                .sum();
                            if d < best_d {
                                best_d = d;
                                best = i as u32;
                            }
                        }
                        best
                    }
                }
            }
        };
        self.variant_clock += 1;
        self.variant_seen[idx as usize] = self.variant_clock;
        self.variant_used |= 1u64 << idx;
        self.cur_variant[c] = idx;
        self.chan.set_variant(ch, idx, rel);
    }

    /// Queue the build of variant `i` from channel `c`'s current controllers.
    /// Built at the end of the block rather than here: a table can be tens of
    /// thousands of entries and the event loop is not the place for it.
    fn build_variant_at(&mut self, i: u32, c: usize) {
        let m = ParamMod::from_controllers(
            self.cc_sound[c][0],
            self.cc_sound[c][1],
            self.cc_sound[c][2],
            self.cc_sound[c][3],
            self.cc_sound[c][4],
        );
        // A slot rebuilt twice in one block only needs its last state.
        self.pending_variants.retain(|(j, _)| *j != i);
        self.pending_variants.push((i, m));
    }

    /// CC121, and the start-of-file state. Everything a channel carries goes
    /// back to where it powers on.
    fn reset_controllers(&mut self, ch: u8, rel: u32) {
        let c = ch as usize & 15;
        self.bank_msb[c] = 0;
        self.bank_lsb[c] = 0;
        self.bend_range[c] = 2.0;
        self.bend_val[c] = 0;
        self.rpn_sel[c] = RPN_NONE;
        self.cc_volume[c] = self.cfg.default_channel_volume.min(127);
        self.cc_expression[c] = 127;
        self.cc_pan[c] = 64;
        self.cc_soft[c] = 0;
        self.cc_mod[c] = 0;
        self.lsb_mod[c] = 0;
        self.lsb_volume[c] = 0;
        self.lsb_pan[c] = 0;
        self.lsb_expression[c] = 0;
        self.cc_tremolo[c] = 0;
        self.cc_vib_rate[c] = 64;
        self.cc_vib_depth[c] = 64;
        for w in 0..5 {
            self.set_sound_cc(ch, w, 64, rel);
        }
        self.refresh_bend(c, rel);
        self.refresh_gain(c, rel);
        self.gates.reset_controllers(ch, rel);
    }

    /// Lay the vibrato and tremolo LFOs over the controller rows.
    ///
    /// Both are per channel, not per voice, so they can be evaluated on the
    /// host: every voice on a channel shares one curve. That is the whole
    /// reason modulation is cheap here. It is sampled once per gate tile, 32
    /// frames or 0.67 ms, which is a 1.5 kHz sample rate for a curve that
    /// moves at 5 Hz.
    ///
    /// Depth is the GM default of +/-50 cents at full wheel, scaled by CC77.
    /// Rate is 5 Hz at CC76's centre and spans roughly 0.5 to 14 Hz.
    fn apply_modulation(&mut self) {
        let tiles = self.chan.tiles;
        let tile_seconds = self.cfg.gate_frames as f64 / self.cfg.sample_rate as f64;
        for c in 0..16usize {
            let vib = (self.cc_mod[c] as f64 * 128.0 + self.lsb_mod[c] as f64) / 16383.0
                * (self.cc_vib_depth[c] as f64 / 64.0)
                * 50.0;
            let trem = self.cc_tremolo[c] as f64 / 127.0 * 0.25;
            let rate = 5.0 * (2.0f64).powf((self.cc_vib_rate[c] as f64 - 64.0) / 32.0);
            if vib <= 0.0 && trem <= 0.0 {
                // Keep the phase running anyway, so switching the wheel on
                // mid-note does not jump the curve to wherever it left off.
                self.lfo_phase[c] =
                    (self.lfo_phase[c] + rate * tile_seconds * tiles as f64).fract();
                continue;
            }
            let mut phase = self.lfo_phase[c];
            for t in 0..tiles {
                let s = (phase * std::f64::consts::TAU).sin();
                if vib > 0.0 {
                    self.chan
                        .modulate_bend(c as u8, t, bend_factor(vib * s / 100.0));
                }
                if trem > 0.0 {
                    self.chan
                        .modulate_gain(c as u8, t, (1.0 - trem + trem * s) as f32);
                }
                phase = (phase + rate * tile_seconds).fract();
            }
            self.lfo_phase[c] = phase;
        }
    }

    /// Freeze this channel's bend into the factor the backends multiply by.
    /// The power lives here, on the host, once per event rather than once per
    /// voice per tile.
    fn refresh_bend(&mut self, c: usize, rel: u32) {
        let semitones = (self.bend_val[c] as f64 / 8192.0) * self.bend_range[c];
        self.chan.set_bend(c as u8, bend_factor(semitones), rel);
    }

    /// Fold CC7, CC11 and CC10 into the pair of gains a voice multiplies by.
    ///
    /// Volume and expression are squared, which is the General MIDI
    /// definition: the controller is 40*log10(v/127) dB of attenuation, and
    /// that is the same statement as an amplitude of (v/127)^2. Getting this
    /// wrong is not subtle -- a linear reading makes every mid-level fader
    /// setting several dB too loud.
    ///
    /// Pan is constant power, and multiplies whatever pan the region already
    /// had rather than replacing it.
    fn refresh_gain(&mut self, c: usize, rel: u32) {
        // 14-bit where a file bothers to send the fine half, 7-bit otherwise:
        // with the LSB at zero these are exactly msb/127.
        let fine = |msb: u8, lsb: u8| {
            if lsb == 0 {
                msb as f32 / 127.0
            } else {
                (msb as f32 * 128.0 + lsb as f32) / 16383.0
            }
        };
        let v = fine(self.cc_volume[c], self.lsb_volume[c]);
        let e = fine(self.cc_expression[c], self.lsb_expression[c]);
        // The soft pedal is continuous rather than a switch, and takes off up
        // to 6 dB at the bottom of its travel. A real una corda also darkens
        // the tone; that part is not modelled.
        let soft = 1.0 - 0.5 * (self.cc_soft[c] as f32 / 127.0);
        let amp = (v * v) * (e * e) * soft;
        let theta = fine(self.cc_pan[c], self.lsb_pan[c]) * std::f32::consts::FRAC_PI_2;
        // Centre has to come out at exactly one on both sides, or a file that
        // sends a redundant "pan centre" would quietly drop 3 dB and stop the
        // unity fast path.
        let (l, r) = if self.cc_pan[c] == 64 && self.lsb_pan[c] == 0 {
            (1.0, 1.0)
        } else {
            (theta.cos() * std::f32::consts::SQRT_2, theta.sin() * std::f32::consts::SQRT_2)
        };
        self.chan.set_gain(c as u8, amp * l, amp * r, rel);
    }

    pub fn seconds_rendered(&self) -> f64 {
        self.stats.frames as f64 / self.cfg.sample_rate as f64
    }

}
