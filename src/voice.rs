//! Voice pool layout, spawn commands, and the note-off gate table.
//!
//! Every field here has a counterpart in `shaders/common.wgsl`. The comment
//! block at the top of that file lists the binding order; if you add a field,
//! change both.

use crate::config::Config;
use crate::fixed::BEND_ONE;

// Envelope stages. Part of the host/device contract, so do not renumber them.
pub const ENV_ATTACK: u32 = 0;
pub const ENV_DECAY: u32 = 1;
pub const ENV_SUSTAIN: u32 = 2;
pub const ENV_RELEASE: u32 = 3;
pub const ENV_DEAD: u32 = 4;

/// Slots in the gate table: 16 MIDI channels by 128 keys.
pub const GATE_SLOTS: usize = 16 * 128;

/// Everything needed to start one voice, in the exact order the spawn shader
/// reads it. `#[repr(C)]` plus Pod means this uploads as a straight memcpy.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpawnCmd {
    pub phase_lo: u32,
    pub phase_hi: u32,
    pub step_lo: u32,
    pub step_hi: u32,
    pub smp_base: u32,
    pub smp_len: u32,
    pub loop_start: u32,
    pub loop_end: u32,
    pub flags: u32,
    pub params: u32,
    /// The params variant the channel was on at this voice's own note-on.
    ///
    /// Voices otherwise take their variant from the per-tile channel row, and
    /// that row carries the state at the *start* of the tile, so a controller
    /// arriving on the note's own tick would not reach it for a whole gate
    /// tile. This carries it across that gap: the voice is born on the right
    /// copy of the table, and the rows govern it from the next tile on, which
    /// leaves bend and channel gain timing alone.
    pub variant: u32,
    pub region: u32,
    /// `channel * 128 + key`, the note-off gate this voice listens to.
    pub gate_slot: u32,
    /// Which note-on of that slot this voice belongs to, counting from 1.
    /// The voice releases once the slot's off count reaches this.
    pub ordinal: u32,
    /// Frames into the block before the voice starts. Gives sample-accurate
    /// note-on without shrinking the block.
    pub start_rel: u32,
    pub note_id_lo: u32,
    pub note_id_hi: u32,
    pub gain_l: f32,
    pub gain_r: f32,
}

/// Which of `want` queued spawns the `i`-th accepted one should be, when only
/// `take` of them fit in the pool.
///
/// Taking the first `take` in event order is the obvious thing and it is
/// wrong. A saturated block's note-ons span the whole block, so a prefix keeps
/// everything from the opening of the block and drops everything after it --
/// the block is heard at its start and silent at its end, which is a rhythmic
/// artifact at exactly the block rate, built in by construction. Spreading the
/// choice evenly keeps the block's timing intact and merely thins it.
///
/// Both backends call this. Two backends thinning a block differently would
/// not show up as an error, only as two renders that quietly disagree.
#[inline]
pub fn spawn_pick(i: usize, want: usize, take: usize) -> usize {
    debug_assert!(take > 0 && take <= want && i < take);
    (i as u64 * want as u64 / take as u64) as usize
}

/// The 64-bit key admission ranks queued spawns by, highest kept first.
///
/// `spawn_pick` thins a saturated block evenly, which fixes the rhythmic
/// artifact a prefix causes but is blind to what it is dropping: a fortissimo
/// whole note and a 10 ms grace note have identical odds. On a file that
/// oversubscribes the pool fifteen to one that is most of what you hear, and
/// what you hear is the loud sustained material being punched out at random.
///
/// The key is two fields, most significant first:
///
/// * **loudness** -- `gain_l + gain_r` quantised to 15 bits. These gains
///   already carry the velocity, the region's own attenuation and its pan, so
///   this is the voice's actual opening amplitude rather than a raw velocity
///   byte.
/// * **position** -- the candidate's index in the block, bit-reversed, as a
///   tiebreak that carries no information about when the note arrived.
///
/// The scrambled position makes it a total order with no ties, so the admitted set
/// is a pure function of the input and never of scheduling order. Both
/// backends call this. Two backends admitting different notes would not show
/// up as an error, only as two renders that quietly disagree.
///
/// # The field that used to sit above loudness
///
/// There was a third field, and it was the top one: **sustained**, 1 if the
/// note was still sounding at the end of *this block*. It was removed on
/// 2026-08-22 because it was the largest single source of block-rate pumping
/// in the renderer, and because its stated justification is false.
///
/// The justification was that a note whose note-off already arrived is the
/// cheapest thing in the block to give up, "because nothing is lost past the
/// block boundary". That holds only if release is instant. The EastWest
/// sampled piano releases over seconds, and black MIDI notes are a tick long,
/// so the release tail *is* the sound -- what the bit called disposable is
/// nearly the whole voice.
///
/// Worse, "still sounding at the end of this block" is a fact about where in
/// the block a note falls, not about the note. For tick-length notes it is
/// very nearly a function of position alone, and it sat *above* loudness, so
/// it decided the ranking. A rank that is a function of time, applied inside
/// time strata whose entire purpose is to keep rank and time independent.
///
/// Measured over six seconds of a saturated section, as AM depth
/// at the block rate: 26.31% before, 16.49% with the tiebreak scrambled, and
/// **1.23%** once this bit went -- level with the 1.33% that `--admit even`
/// reaches by not ranking at all. See `PUMPING.md`.
///
/// If the distinction is ever wanted back, it has to be expressed without
/// reference to the block: "how much sound has this note left to make" is
/// `gain * release`, both known per region, and neither mentions a boundary.
///
/// Reordering the queue is safe because a voice's position in time is carried
/// by `start_rel`, not by its index: thinning by rank still leaves every
/// admitted note sounding at exactly the frame it should.
#[inline]
pub fn admit_key(cmd: &SpawnCmd, index: u64) -> u64 {
    (rank_gain_q(cmd.gain_l, cmd.gain_r) << 48) | mix48(index)
}

/// The tiebreak field of the ranking keys: a scrambled function of a
/// candidate's position in the block.
///
/// Two wrong answers came before this one and both are worth keeping written
/// down, because each looked correct and one of them shipped a defect.
///
/// **The raw note id.** Ids are handed out in time order, so every tie
/// resolved toward the later note -- and ties are not the rare case. Measured
/// on saturated material, admitted mean gain equals queued
/// mean gain to three decimals (selectivity 1.000, and 1.024 on a second file),
/// so the loudness field ties for almost every pair and the tiebreak decides
/// nearly every comparison. That made the rank a function of *when* a note
/// arrives, inside time strata whose whole purpose is to keep rank and time
/// independent.
///
/// **The bit-reversed index**, a van der Corput spread, on the theory that
/// selecting an evenly spread subset would beat a random one. It measured
/// 8.56% against the hash's 8.52% on a second file -- no difference -- and it
/// **broke the stereo image**: 3.77 dB of channel imbalance against 0.68 dB
/// for `AdmitRule::Even`. A stereo library becomes two hard-panned mono
/// regions, so every note queues two adjacent candidates, left at an even
/// index and right at an odd one. Bit-reversing puts bit 0 of the index in the
/// *most significant* bit of the tiebreak, so every right channel outranked
/// every left channel and admission threw away the left. Any tiebreak built
/// from the index must destroy its low bits, not promote them.
///
/// An avalanche mix does that: consecutive indices scatter, so channel parity
/// carries no rank. It is a bijection on 48 bits -- xor-shift-right and odd
/// multiplies, both invertible mod 2^48 -- so distinct candidates still get
/// distinct keys and the order stays total with no ties, which is what makes
/// the admitted set a pure function of the input rather than of scheduling.
#[inline]
pub(crate) fn mix48(index: u64) -> u64 {
    const M: u64 = 0x0000_FFFF_FFFF_FFFF;
    let mut z = index & M;
    z = (z ^ (z >> 24)) & M;
    z = z.wrapping_mul(0x9E37_79B9_7F4B) & M;
    z = (z ^ (z >> 23)) & M;
    z = z.wrapping_mul(0xBF58_476D_1CE5) & M;
    z = (z ^ (z >> 25)) & M;
    z
}

/// Quantise a voice's opening gain to the 15 bits `admit_key` has for it.
///
/// This was `(gain_l + gain_r).clamp(0.0, 1.0) * 32767`, and the clamp was the
/// whole problem. A loud soundfont at unity volume puts almost every voice
/// above 1.0, so the field **saturated**: on a loud sampled piano the queued
/// mean was 32,470 of a possible 32,767 and the
/// admitted mean was 32,767.00 exactly, a selectivity of 1.009. With the field
/// carrying no information the key fell through to its tiebreak. A small
/// `--volume` degenerates the same expression at the other end, collapsing
/// every voice onto one or two steps, which is how this hid for so long: the
/// shipped setting and the diagnostic setting `PUMPING.md` prescribes both
/// look ranked and neither is.
///
/// The fix costs one shift. An IEEE-754 bit pattern of a positive float is
/// monotonically increasing in the value, and its exponent field is literally
/// log2, so the top 15 bits below the sign are already a logarithmic
/// quantisation covering the entire positive range. No clamp to saturate, no
/// window to fall outside, and no `log10` in a path that runs about five
/// billion times on this file -- the honest arithmetic version measured 1.88x
/// slower end to end.
///
/// Monotonic and exact, so it is a pure function of the gain and both
/// backends and both runs agree.
#[inline]
pub fn rank_gain_q(gain_l: f32, gain_r: f32) -> u64 {
    let g = gain_l.abs() + gain_r.abs();
    // NaN as well as zero and negatives: a NaN gain must not become a rank.
    if g.is_nan() || g <= 0.0 {
        return 0;
    }
    ((g.to_bits() >> 16) & 0x7FFF) as u64
}

/// Per-block note-off state, sampled once per reduce tile.
///
/// A voice cannot be found by searching the pool without either a sort or an
/// index that survives compaction, so the lookup is inverted: the host
/// publishes a small table of "how many note-offs has this key seen", and each
/// voice compares its own ordinal against it. The table is 8 KiB per tile, so
/// it stays in L1 and costs one cached read per voice per tile.
pub struct GateTable {
    /// `tiles * GATE_SLOTS` entries, tile-major.
    pub rows: Vec<u32>,
    pub tiles: usize,
    /// Live counters, carried across blocks.
    on_count: Vec<u32>,
    off_count: Vec<u32>,
    /// Note-offs that arrived while the channel's sustain pedal was down and
    /// so have not been published yet. See `set_sustain`.
    pending_off: Vec<u32>,
    /// One bit per channel, set while CC64 is down.
    sustain: u16,
    /// One bit per channel, set while CC66 is down.
    sostenuto: u16,
    /// How many notes at each slot the sostenuto pedal caught. Sostenuto only
    /// holds what was already sounding when it went down, which is the whole
    /// difference between it and the sustain pedal.
    sost_held: Vec<u32>,
    /// Next tile that still needs its row written.
    cursor: usize,
    tile_frames: u32,
}

impl GateTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        GateTable {
            rows: vec![0; tiles * GATE_SLOTS],
            tiles,
            on_count: vec![0; GATE_SLOTS],
            off_count: vec![0; GATE_SLOTS],
            pending_off: vec![0; GATE_SLOTS],
            sustain: 0,
            sostenuto: 0,
            sost_held: vec![0; GATE_SLOTS],
            cursor: 0,
            tile_frames: cfg.gate_frames,
        }
    }

    #[inline]
    pub fn slot(ch: u8, key: u8) -> usize {
        (ch as usize & 15) * 128 + (key as usize & 127)
    }

    /// Start a new block. Rows are written lazily as events arrive.
    pub fn begin_block(&mut self) {
        self.cursor = 0;
    }

    /// Write rows up to and including the tile containing `frame`, so they
    /// reflect the state before any event at that frame.
    #[inline]
    fn advance_to(&mut self, frame: u32) {
        let tile = (frame / self.tile_frames) as usize;
        while self.cursor <= tile && self.cursor < self.tiles {
            let base = self.cursor * GATE_SLOTS;
            self.rows[base..base + GATE_SLOTS].copy_from_slice(&self.off_count);
            self.cursor += 1;
        }
    }

    /// Whether the note-off for `ordinal` at `slot` has already been
    /// published. Used by admission to tell a note that outlives the block
    /// from one that is over before the block ends.
    #[inline]
    pub fn released(&self, slot: u32, ordinal: u32) -> bool {
        self.off_count[slot as usize % GATE_SLOTS] >= ordinal
    }

    /// Register a note-on. Returns the ordinal the voice should carry.
    pub fn note_on(&mut self, ch: u8, key: u8, frame: u32) -> u32 {
        self.advance_to(frame);
        let s = Self::slot(ch, key);
        self.on_count[s] = self.on_count[s].wrapping_add(1);
        self.on_count[s]
    }

    /// Register a note-off. A note-off with nothing sounding is ignored, which
    /// is what makes "release the oldest un-released note" fall out for free.
    pub fn note_off(&mut self, ch: u8, key: u8, frame: u32) {
        self.advance_to(frame);
        let s = Self::slot(ch, key);
        if self.sostenuto & (1u16 << (ch & 15)) != 0 && self.sost_held[s] > 0 {
            // Caught by the sostenuto pedal when it went down, so this off is
            // held even if the sustain pedal is up.
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
                self.sost_held[s] -= 1;
            }
            return;
        }
        if self.sustained(ch) {
            // Held, not released. Counting it here rather than in `off_count`
            // is the whole of the sustain pedal: the device only ever sees
            // published note-offs, so a deferred one simply has not happened
            // yet as far as any voice is concerned.
            if self.off_count[s].wrapping_add(self.pending_off[s]) != self.on_count[s] {
                self.pending_off[s] += 1;
            }
        } else if self.off_count[s] != self.on_count[s] {
            self.off_count[s] = self.off_count[s].wrapping_add(1);
        }
    }

    #[inline]
    pub fn sustained(&self, ch: u8) -> bool {
        self.sustain & (1u16 << (ch & 15)) != 0
    }

    /// CC64. Pressing holds every later note-off on the channel; releasing
    /// publishes all of them at this frame, which is what makes a pedalled
    /// chord ring on and then damp together.
    pub fn set_sustain(&mut self, ch: u8, down: bool, frame: u32) {
        if down == self.sustained(ch) {
            return;
        }
        self.advance_to(frame);
        let bit = 1u16 << (ch & 15);
        if down {
            self.sustain |= bit;
            return;
        }
        self.sustain &= !bit;
        if self.sostenuto & bit != 0 {
            // Sostenuto is still down and still holding what it caught.
            return;
        }
        self.flush_pending(ch);
    }

    /// CC66. Holds only the notes already sounding when it goes down; notes
    /// struck afterwards damp normally, which is the whole point of it.
    pub fn set_sostenuto(&mut self, ch: u8, down: bool, frame: u32) {
        let bit = 1u16 << (ch & 15);
        if down == (self.sostenuto & bit != 0) {
            return;
        }
        self.advance_to(frame);
        let base = (ch as usize & 15) * 128;
        if down {
            self.sostenuto |= bit;
            for k in 0..128 {
                let s = base + k;
                self.sost_held[s] = self.on_count[s]
                    .wrapping_sub(self.off_count[s])
                    .wrapping_sub(self.pending_off[s]);
            }
            return;
        }
        self.sostenuto &= !bit;
        for k in 0..128 {
            self.sost_held[base + k] = 0;
        }
        if self.sustained(ch) {
            // The other pedal is still down, so nothing damps yet.
            return;
        }
        self.flush_pending(ch);
    }

    /// Publish everything a pedal was holding on this channel.
    fn flush_pending(&mut self, ch: u8) {
        let base = (ch as usize & 15) * 128;
        for k in 0..128 {
            let s = base + k;
            if self.pending_off[s] != 0 {
                self.off_count[s] = self.off_count[s].wrapping_add(self.pending_off[s]);
                self.pending_off[s] = 0;
            }
        }
    }

    /// CC123. Releases what is sounding, but a held pedal still holds: the
    /// notes damp when the pedal comes up, not here.
    pub fn all_notes_off(&mut self, ch: u8, frame: u32) {
        self.advance_to(frame);
        let base = (ch as usize & 15) * 128;
        if self.sustained(ch) {
            for k in 0..128 {
                let s = base + k;
                self.pending_off[s] = self.on_count[s].wrapping_sub(self.off_count[s]);
            }
            return;
        }
        for k in 0..128 {
            self.off_count[base + k] = self.on_count[base + k];
        }
    }

    /// CC120. Stops everything on the channel now, pedal or not, and drops
    /// what the pedal was holding.
    pub fn all_sound_off(&mut self, ch: u8, frame: u32) {
        self.advance_to(frame);
        let base = (ch as usize & 15) * 128;
        for k in 0..128 {
            self.off_count[base + k] = self.on_count[base + k];
            self.pending_off[base + k] = 0;
            self.sost_held[base + k] = 0;
        }
    }

    /// CC121. Lifting the pedal is part of resetting a channel's controllers,
    /// so anything it was holding damps here.
    pub fn reset_controllers(&mut self, ch: u8, frame: u32) {
        self.set_sostenuto(ch, false, frame);
        self.set_sustain(ch, false, frame);
    }

    /// Finish the block by filling any tiles no event reached.
    pub fn end_block(&mut self) {
        while self.cursor < self.tiles {
            let base = self.cursor * GATE_SLOTS;
            self.rows[base..base + GATE_SLOTS].copy_from_slice(&self.off_count);
            self.cursor += 1;
        }
    }

    #[inline]
    pub fn row(&self, tile: usize) -> &[u32] {
        let base = tile * GATE_SLOTS;
        &self.rows[base..base + GATE_SLOTS]
    }

    /// Number of notes started but not yet released, across all slots. Notes
    /// the pedal is holding count as sounding, because they are.
    pub fn sounding(&self) -> u64 {
        self.on_count
            .iter()
            .zip(&self.off_count)
            .map(|(a, b)| a.wrapping_sub(*b) as u64)
            .sum()
    }
}

pub const BEND_CHANNELS: usize = 16;
/// Words per channel in a `ChannelTable` row: bend factor, left gain, right
/// gain, and one spare to keep the stride a power of two so a channel's whole
/// entry lands in one 16-byte chunk of the same cache line.
pub const CHAN_FIELDS: usize = 4;
pub const CHAN_BEND: usize = 0;
pub const CHAN_GAIN_L: usize = 1;
pub const CHAN_GAIN_R: usize = 2;
/// Which copy of the params table this channel's voices read, see `ParamMod`.
pub const CHAN_VARIANT: usize = 3;

/// Per-channel controller state, published the same way the note-off gate is:
/// one row per gate tile, which a voice reads for its own channel.
///
/// Neither of these can be resolved on the host the way sustain can. Sustain
/// only changes *when* a note-off is published, and note-offs are already a
/// host table. Bend changes the step of every voice already in the pool, and
/// channel volume changes the gain of every voice already in the pool, and
/// there may be fourteen million of them. So this is the part of the
/// controller set that costs device work.
///
/// It is arranged to cost as little as possible. The two are tracked
/// separately -- a file that bends but never touches volume pays only for the
/// bend -- and if a control sat at unity for the whole block, the render pass
/// is told and skips that work entirely. Both live in one table so that a
/// channel's bend and gains are adjacent rather than in two rows a kilobyte
/// apart.
pub struct ChannelTable {
    /// `tiles * BEND_CHANNELS * CHAN_FIELDS` entries, tile-major. Gains are
    /// f32 bit patterns; the bend factor is 8.24 fixed point.
    pub rows: Vec<u32>,
    pub tiles: usize,
    /// Current state per channel, carried across blocks.
    now: [u32; BEND_CHANNELS * CHAN_FIELDS],
    cursor: usize,
    tile_frames: u32,
    bend_active: bool,
    gain_active: bool,
    variant_active: bool,
}

impl ChannelTable {
    pub fn new(cfg: &Config) -> Self {
        let tiles = (cfg.block_frames / cfg.gate_frames) as usize;
        let mut now = [0u32; BEND_CHANNELS * CHAN_FIELDS];
        for c in 0..BEND_CHANNELS {
            now[c * CHAN_FIELDS + CHAN_BEND] = BEND_ONE;
            now[c * CHAN_FIELDS + CHAN_GAIN_L] = 1.0f32.to_bits();
            now[c * CHAN_FIELDS + CHAN_GAIN_R] = 1.0f32.to_bits();
            now[c * CHAN_FIELDS + CHAN_VARIANT] = 0;
        }
        let mut rows = vec![0u32; tiles * BEND_CHANNELS * CHAN_FIELDS];
        for t in 0..tiles {
            let base = t * BEND_CHANNELS * CHAN_FIELDS;
            rows[base..base + BEND_CHANNELS * CHAN_FIELDS].copy_from_slice(&now);
        }
        ChannelTable {
            rows,
            tiles,
            now,
            cursor: 0,
            tile_frames: cfg.gate_frames,
            bend_active: false,
            gain_active: false,
            variant_active: false,
        }
    }

    pub fn begin_block(&mut self) {
        self.cursor = 0;
    }

    #[inline]
    fn advance_to(&mut self, frame: u32) {
        let tile = (frame / self.tile_frames) as usize;
        let w = BEND_CHANNELS * CHAN_FIELDS;
        while self.cursor <= tile && self.cursor < self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Set a channel's bend factor from this frame on.
    pub fn set_bend(&mut self, ch: u8, factor: u32, frame: u32) {
        let i = (ch as usize & 15) * CHAN_FIELDS + CHAN_BEND;
        if self.now[i] == factor {
            return;
        }
        self.advance_to(frame);
        self.now[i] = factor;
    }

    /// Set a channel's output gains from this frame on. These multiply the
    /// per-voice gains, which carry the region's own pan and the velocity.
    pub fn set_gain(&mut self, ch: u8, l: f32, r: f32, frame: u32) {
        let base = (ch as usize & 15) * CHAN_FIELDS;
        let (lb, rb) = (l.to_bits(), r.to_bits());
        if self.now[base + CHAN_GAIN_L] == lb && self.now[base + CHAN_GAIN_R] == rb {
            return;
        }
        self.advance_to(frame);
        self.now[base + CHAN_GAIN_L] = lb;
        self.now[base + CHAN_GAIN_R] = rb;
    }

    /// Select which copy of the params table this channel reads from.
    pub fn set_variant(&mut self, ch: u8, variant: u32, frame: u32) {
        let i = (ch as usize & 15) * CHAN_FIELDS + CHAN_VARIANT;
        if self.now[i] == variant {
            return;
        }
        self.advance_to(frame);
        self.now[i] = variant;
    }

    /// Fill any tiles no event reached. Call before `modulate` and
    /// `refresh_active`.
    pub fn end_block(&mut self) {
        let w = BEND_CHANNELS * CHAN_FIELDS;
        while self.cursor < self.tiles {
            let base = self.cursor * w;
            self.rows[base..base + w].copy_from_slice(&self.now);
            self.cursor += 1;
        }
    }

    /// Multiply a tile's already-published bend factor, for an LFO the host
    /// evaluates per tile rather than per event. Kept separate from `set_bend`
    /// because modulation is a continuous curve laid over whatever the wheel
    /// and the bend lever last asked for, not a replacement for it.
    pub fn modulate_bend(&mut self, ch: u8, tile: usize, factor: u32) {
        let i = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS + CHAN_BEND;
        let scaled = ((self.rows[i] as u64 * factor as u64) >> 24) as u32;
        self.rows[i] = scaled.max(1);
    }

    /// Scale a tile's already-published gains, for a tremolo LFO.
    pub fn modulate_gain(&mut self, ch: u8, tile: usize, factor: f32) {
        let base = (tile * BEND_CHANNELS + (ch as usize & 15)) * CHAN_FIELDS;
        for f in [CHAN_GAIN_L, CHAN_GAIN_R] {
            let v = f32::from_bits(self.rows[base + f]) * factor;
            self.rows[base + f] = v.to_bits();
        }
    }

    pub fn refresh_active(&mut self) {
        let one = 1.0f32.to_bits();
        self.bend_active = false;
        self.gain_active = false;
        self.variant_active = false;
        for e in self.rows.chunks_exact(CHAN_FIELDS) {
            if e[CHAN_BEND] != BEND_ONE {
                self.bend_active = true;
            }
            if e[CHAN_GAIN_L] != one || e[CHAN_GAIN_R] != one {
                self.gain_active = true;
            }
            if e[CHAN_VARIANT] != 0 {
                self.variant_active = true;
            }
        }
    }

    /// False when every channel read the untouched params table, which lets
    /// both backends skip re-reading a voice's DSP constants entirely.
    #[inline]
    pub fn variant_active(&self) -> bool {
        self.variant_active
    }

    /// True when any of the three needs the controller path at all.
    #[inline]
    pub fn any_active(&self) -> bool {
        self.bend_active || self.gain_active || self.variant_active
    }

    /// False when nothing in this block is bent, which lets both backends skip
    /// the per-voice step scale.
    #[inline]
    pub fn bend_active(&self) -> bool {
        self.bend_active
    }

    /// False when every channel sat at unity gain, which lets both backends
    /// use the voice's own gains untouched.
    #[inline]
    pub fn gain_active(&self) -> bool {
        self.gain_active
    }

    #[inline]
    pub fn row(&self, tile: usize) -> &[u32] {
        let w = BEND_CHANNELS * CHAN_FIELDS;
        &self.rows[tile * w..tile * w + w]
    }

    /// The channel a voice belongs to, recovered from its gate slot. Voices
    /// already carry the slot for the note-off gate, so neither control needs
    /// an extra per-voice field.
    #[inline]
    pub fn channel_of(gate_slot: u32) -> usize {
        (gate_slot >> 7) as usize & 15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            block_frames: 64,
            reduce_tile: 8,
            gate_frames: 16,
            ..Default::default()
        }
    }

    #[test]
    fn gate_rows_reflect_state_at_tile_start() {
        let mut g = GateTable::new(&cfg());
        assert_eq!(g.tiles, 4);
        g.begin_block();
        let ord = g.note_on(0, 60, 0);
        assert_eq!(ord, 1);
        g.note_off(0, 60, 40); // tile 2
        g.end_block();

        let s = GateTable::slot(0, 60);
        assert_eq!(g.row(0)[s], 0);
        assert_eq!(g.row(1)[s], 0);
        assert_eq!(g.row(2)[s], 0, "the off happens inside tile 2, not before it");
        assert_eq!(g.row(3)[s], 1);
    }

    #[test]
    fn the_pedal_defers_note_offs_and_releases_them_together() {
        let mut g = GateTable::new(&cfg());
        let s = GateTable::slot(0, 60);
        g.begin_block();
        g.note_on(0, 60, 0);
        g.set_sustain(0, true, 0);
        g.note_off(0, 60, 16); // held
        g.end_block();
        assert_eq!(g.row(3)[s], 0, "a pedalled note-off must not be published");
        assert_eq!(g.sounding(), 1, "the held note is still sounding");

        g.begin_block();
        g.set_sustain(0, false, 16);
        g.end_block();
        // Rows carry the state at the *start* of their tile, so the lift at
        // frame 16 shows up in tile 2, not in the tile it happened in.
        assert_eq!(g.row(1)[s], 0, "the release lands at the lift, not before");
        assert_eq!(g.row(2)[s], 1);
        assert_eq!(g.sounding(), 0);
    }

    /// Restriking a key while the pedal is down leaves two notes sounding and
    /// one deferred note-off. Lifting must release exactly the first: the
    /// deferral is a count, not a flag, and it has to stay clamped to what is
    /// actually sounding or the second note is released by the first's off.
    #[test]
    fn a_restrike_under_the_pedal_releases_only_what_was_lifted() {
        let mut g = GateTable::new(&cfg());
        let s = GateTable::slot(0, 60);
        g.begin_block();
        g.set_sustain(0, true, 0);
        g.note_on(0, 60, 0);
        g.note_off(0, 60, 0);
        g.note_on(0, 60, 16);
        // A second off with only one note left un-lifted is still legal.
        g.note_off(0, 60, 16);
        // A third has nothing behind it and must be dropped, exactly as an
        // unmatched note-off is when there is no pedal.
        g.note_off(0, 60, 16);
        g.end_block();
        assert_eq!(g.sounding(), 2, "both strikes are held by the pedal");

        g.begin_block();
        g.set_sustain(0, false, 0);
        g.end_block();
        assert_eq!(g.row(3)[s], 2, "both held offs publish, the third does not");
        assert_eq!(g.sounding(), 0);
    }

    #[test]
    fn unmatched_note_off_does_not_pre_release() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        g.note_off(0, 60, 0); // nothing sounding
        let ord = g.note_on(0, 60, 16);
        g.end_block();
        let s = GateTable::slot(0, 60);
        assert_eq!(ord, 1);
        assert_eq!(g.row(3)[s], 0, "stray note-off must not release the next note");
    }

    #[test]
    fn retrigger_releases_in_order() {
        let mut g = GateTable::new(&cfg());
        g.begin_block();
        let a = g.note_on(0, 60, 0);
        let b = g.note_on(0, 60, 0);
        g.note_off(0, 60, 16);
        g.end_block();
        assert_eq!((a, b), (1, 2));
        let s = GateTable::slot(0, 60);
        // Only the first note is released.
        assert!(g.row(3)[s] >= a);
        assert!(g.row(3)[s] < b);
    }
}
