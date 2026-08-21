// Shared declarations, prepended to every pass.
//
// WGSL has no include mechanism, so `gpu::shader_source` concatenates this file
// with each pass and substitutes the {{...}} placeholders from the Config.
// Keep the field offsets below in step with `src/voice.rs`.

// ---------------------------------------------------------------------------
// voice pool layout
// ---------------------------------------------------------------------------
// One flat u32 buffer holding structure of arrays: field f of voice i lives at
// `f * capacity + i`. Every field's stride stays 1 across neighbouring
// invocations, which is what makes the region sort worth doing, and it costs
// one binding instead of twenty-two.

const F_PHASE_LO: u32   = 0u;
const F_PHASE_HI: u32   = 1u;
const F_STEP_LO: u32    = 2u;
const F_STEP_HI: u32    = 3u;
const F_SMP_BASE: u32   = 4u;
const F_SMP_LEN: u32    = 5u;
const F_LOOP_START: u32 = 6u;
const F_LOOP_END: u32   = 7u;
const F_FLAGS: u32      = 8u;
const F_ENV_STAGE: u32  = 9u;
const F_ENV_LEVEL: u32  = 10u;
const F_GAIN_L: u32     = 11u;
const F_GAIN_R: u32     = 12u;
const F_FILT_Z1: u32    = 13u;
const F_FILT_Z2: u32    = 14u;
const F_PARAMS: u32     = 15u;
const F_REGION: u32     = 16u;
const F_GATE_SLOT: u32  = 17u;
const F_ORDINAL: u32    = 18u;
const F_START_REL: u32  = 19u;
const F_NOTE_LO: u32    = 20u;
const F_NOTE_HI: u32    = 21u;
// The params variant this voice was born under, plus one, or zero once it has
// outlived the block it was born in. See `SpawnCmd::variant` in src/voice.rs.
// Like F_START_REL it is cleared at the end of every block.
const F_BORN_VARIANT: u32 = 22u;
// Frame within this block at which a stolen voice starts fading, plus one.
// Zero means the voice is not being stolen, so a zeroed slot is inert.
const F_STOP_REL: u32 = 23u;
const VOICE_FIELDS: u32 = 24u;

const ENV_ATTACK: u32  = 0u;
const ENV_DECAY: u32   = 1u;
const ENV_SUSTAIN: u32 = 2u;
const ENV_RELEASE: u32 = 3u;
const ENV_DEAD: u32    = 4u;

const VF_LOOP: u32 = 1u;
const VF_LOOP_UNTIL_RELEASE: u32 = 2u;

const RP_FILTER: u32 = 1u;

const GATE_SLOTS: u32 = 2048u;

const INTERP_NEAREST: u32 = 0u;
const INTERP_LINEAR: u32 = 1u;
const INTERP_CUBIC: u32 = 2u;

// ---------------------------------------------------------------------------
// device-side state
// ---------------------------------------------------------------------------
// The live voice count lives on the device, not in the uniform block, so a
// whole block of passes can be recorded into one command buffer without the
// host having to rewrite a uniform between dispatches.

const S_LIVE: u32       = 0u;
const S_LIVE_NEW: u32   = 1u;
const S_PREFIX_HI: u32  = 2u;
const S_PREFIX_LO: u32  = 3u;
const S_K: u32          = 4u;
const S_BYTE: u32       = 5u;
const S_THRESH_HI: u32  = 6u;
const S_THRESH_LO: u32  = 7u;
const S_STOLEN: u32     = 8u;
const S_DROPPED: u32    = 9u;
const S_PEAK_BITS: u32  = 10u;
const S_SORT_BIT: u32   = 11u;
const S_TOTAL0: u32     = 12u;
const STATE_SLOTS: u32  = 16u;

// Substituted from Config so the compiler sees literals.
const WG: u32 = {{WG}}u;
// Frames a stolen voice fades over. See Config::steal_fade_frames.
const STEAL_FADE: u32 = {{STEAL_FADE}}u;
const TILE: u32 = {{TILE}}u;
// Frames between note-off gate checks. A multiple of TILE.
const GATE_TILE: u32 = {{GATE_TILE}}u;
const TILES_PER_GATE: u32 = GATE_TILE / TILE;

struct RegionParams {
    attack_rate: f32,
    attack_end: f32,
    decay_coef: f32,
    decay_target: f32,
    sustain: f32,
    release_coef: f32,
    b0: f32,
    b1: f32,
    a1: f32,
    a2: f32,
    flags: u32,
    pad: u32,
};

struct Uniforms {
    block_frames: u32,
    tiles: u32,
    capacity: u32,
    spawn_count: u32,

    render_workgroups: u32,
    interp: u32,
    exp_decay: u32,
    exp_release: u32,

    env_floor: f32,
    pool_words: u32,
    steal_k: u32,
    sort_bits: u32,

    sort_region_shift: u32,
    sort_stage_shift: u32,
    sort_phase_shift: u32,
    sort_phase_mask: u32,

    sort_dead_region: u32,
    chan_active: u32,
    params_per_variant: u32,
    /// Non-zero to steal by envelope level rather than by age.
    steal_by_level: u32,
};

// Bits of Uniforms::chan_active.
const CHAN_ACTIVE_BEND: u32 = 1u;
const CHAN_ACTIVE_GAIN: u32 = 2u;
const CHAN_ACTIVE_VARIANT: u32 = 4u;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// 32.32 fixed point, held as two u32 lanes because WGSL has no portable u64.
// Channels in a controller row. Sixteen, like the MIDI spec.
const BEND_CHANNELS: u32 = 16u;
// Words per channel: bend factor, left gain, right gain, one spare.
const CHAN_FIELDS: u32 = 4u;
const CHAN_BEND: u32 = 0u;
const CHAN_GAIN_L: u32 = 1u;
const CHAN_GAIN_R: u32 = 2u;
const CHAN_VARIANT: u32 = 3u;
// Fractional bits in a bend factor. Matches BEND_FRAC_BITS on the host.
const BEND_SHIFT: u32 = 24u;

// The 64-bit key voice stealing selects the k smallest of, as (hi, lo). The
// caller loads the words; this only decides how they are combined, so that the
// selection and the marking cannot disagree about the ordering.
//
// By age the key is simply the note id, unique and handed out in event order.
// By level it is the envelope level in the top 16 bits and the note id in the
// low 48, so quiet voices sort first and the id breaks ties. The radix select
// needs a total order with no ties or "the k smallest" is not a well-defined
// set, which is why the id is carried rather than the level being used alone:
// levels collide constantly, with a million voices sitting at exactly 1.0 or
// exactly 0.0. A note id needs 48 bits to stay unique for any file this will
// see -- 2.8e14 against the 5.5e9 of the largest known.
fn steal_key(hi: u32, lo: u32, level_bits: u32) -> vec2<u32> {
    if (u.steal_by_level == 0u) {
        return vec2<u32>(hi, lo);
    }
    let level = max(bitcast<f32>(level_bits), 0.0);
    let q = u32(min(level, 1.0) * 65535.0);
    return vec2<u32>((q << 16u) | (hi & 0xFFFFu), lo);
}

// 32x32 -> 64 unsigned multiply, returned as (lo, hi). WGSL has no portable
// 64-bit integer, so this is the schoolbook version in 16-bit limbs, with the
// middle carry handled explicitly because `p01 + p10` overflows 32 bits.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid1 = p10 + (p00 >> 16u);
    let mid2 = p01 + (mid1 & 0xFFFFu);
    let lo = (mid2 << 16u) | (p00 & 0xFFFFu);
    let hi = p11 + (mid1 >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// Scale a 32.32 value by an 8.24 fixed-point factor, truncating, saturating
// rather than wrapping. Returns (hi, lo) like `add64`.
//
// This has to agree bit for bit with `Fixed::scale` on the host: it is how a
// bent voice gets its step, and a step that differs in its low bits does not
// null as an error, it nulls as two renders drifting apart over a block.
fn scale64(hi: u32, lo: u32, factor: u32) -> vec2<u32> {
    let pl = mul32(lo, factor);   // product bits 0..63
    let ph = mul32(hi, factor);   // product bits 32..95
    let b = pl.y + ph.x;          // bits 32..63
    var c = ph.y;                 // bits 64..95
    if (b < ph.x) { c = c + 1u; } // carry out of b
    if ((c >> BEND_SHIFT) != 0u) {
        return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    let r_lo = (pl.x >> BEND_SHIFT) | (b << (32u - BEND_SHIFT));
    let r_hi = (b >> BEND_SHIFT) | (c << (32u - BEND_SHIFT));
    return vec2<u32>(r_hi, r_lo);
}

fn add64(hi: u32, lo: u32, add_hi: u32, add_lo: u32) -> vec2<u32> {
    let nlo = lo + add_lo;
    var carry = 0u;
    if (nlo < lo) { carry = 1u; }
    return vec2<u32>(hi + add_hi + carry, nlo);
}

// True when a < b, treating each pair as one 64-bit value.
fn less64(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> bool {
    if (a_hi != b_hi) { return a_hi < b_hi; }
    return a_lo < b_lo;
}

fn frac_of(lo: u32) -> f32 {
    return f32(lo) * (1.0 / 4294967296.0);
}

// The index `off` frames away from `idx`, honouring the loop. Mirrors
// `CpuSynth::advance_index` statement for statement.
fn neighbour_index(idx: u32, off: i32, looping: bool, ls: u32, le: u32, len: u32) -> u32 {
    let raw = i32(idx) + off;
    if (looping) {
        if (raw >= i32(le)) {
            let span = max(i32(le) - i32(ls), 1);
            return u32(i32(ls) + (raw - i32(ls)) % span);
        }
        if (raw < 0) { return 0u; }
        return u32(raw);
    }
    return u32(clamp(raw, 0, i32(len) - 1));
}
