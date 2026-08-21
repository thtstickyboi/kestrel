// Voice stealing: find the k-th smallest steal key in the pool.
//
// The key is `steal_key` in common.wgsl -- the note id, or the envelope level
// with the id beneath it as a tiebreak. Either way it is a total order with no
// ties, so the victim set is a pure function of the MIDI file and never of
// scheduling. Getting there needs a k-th-smallest selection over it, done as an
// MSB-first radix select: eight rounds of a 256-bin histogram, each round
// pinning one more byte of the answer.
//
// Integer atomics are used for the histogram. That is not the float-atomic
// problem ruled out for the mixdown: integer addition is exact and
// order-independent, so the histogram is the same every run.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> voices: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;

fn key_byte(hi: u32, lo: u32, b: u32) -> u32 {
    if (b >= 4u) {
        return (hi >> ((b - 4u) * 8u)) & 255u;
    }
    return (lo >> (b * 8u)) & 255u;
}

// Do the bytes above `b` match the prefix pinned so far?
fn prefix_match(hi: u32, lo: u32, b: u32, phi: u32, plo: u32) -> bool {
    if (b >= 4u) {
        let sh = (b - 3u) * 8u; // 8..32
        if (sh >= 32u) { return true; }
        return (hi >> sh) == (phi >> sh);
    }
    if (hi != phi) { return false; }
    let sh = (b + 1u) * 8u; // 8..32
    if (sh >= 32u) { return true; }
    return (lo >> sh) == (plo >> sh);
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 256u) {
        atomicStore(&hist[gid.x], 0u);
    }
}

/// Set up a fresh selection for k = u.steal_k.
@compute @workgroup_size(1)
fn init() {
    state[S_PREFIX_HI] = 0u;
    state[S_PREFIX_LO] = 0u;
    state[S_K] = max(u.steal_k, 1u);
    state[S_BYTE] = 7u;
    state[S_THRESH_HI] = 0u;
    state[S_THRESH_LO] = 0u;
}

@compute @workgroup_size({{WG}})
fn histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let live = state[S_LIVE];
    let b = state[S_BYTE];
    let phi = state[S_PREFIX_HI];
    let plo = state[S_PREFIX_LO];

    var i = gid.x;
    loop {
        if (i >= live) { break; }
        var lvl = 0u;
        if (u.steal_by_level != 0u) { lvl = voices[F_ENV_LEVEL * c + i]; }
        let k = steal_key(voices[F_NOTE_HI * c + i], voices[F_NOTE_LO * c + i], lvl);
        if (prefix_match(k.x, k.y, b, phi, plo)) {
            atomicAdd(&hist[key_byte(k.x, k.y, b)], 1u);
        }
        i = i + stride;
    }
}

/// Walk the 256 bins, pin one byte of the answer, and move to the next byte.
@compute @workgroup_size(1)
fn refine() {
    let b = state[S_BYTE];
    let k = state[S_K];

    var cum = 0u;
    var chosen = 255u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let h = atomicLoad(&hist[i]);
        if (cum + h >= k) {
            chosen = i;
            break;
        }
        cum = cum + h;
    }

    if (b >= 4u) {
        state[S_PREFIX_HI] = state[S_PREFIX_HI] | (chosen << ((b - 4u) * 8u));
    } else {
        state[S_PREFIX_LO] = state[S_PREFIX_LO] | (chosen << (b * 8u));
    }
    state[S_K] = k - cum;

    if (b == 0u) {
        state[S_THRESH_HI] = state[S_PREFIX_HI];
        state[S_THRESH_LO] = state[S_PREFIX_LO];
    } else {
        state[S_BYTE] = b - 1u;
    }
}
