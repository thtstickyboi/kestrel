// Voice pool sort, run as part of compaction.
//
// Why this exists, measured rather than assumed. With a million voices on an
// RTX 5060 the render pass spends 86 ms of its 147 ms inside the two sample
// fetches, and the size of the sample pool barely moves that number: a 4 MiB
// pool costs 145 ms and a 512 MiB pool 147 ms. So the cost is not DRAM
// bandwidth, it is the rate at which the L1 can serve distinct 32-byte
// sectors. Thirty-two lanes of a warp reading thirty-two unrelated samples
// need thirty-two sectors; thirty-two lanes reading neighbouring positions in
// the same sample need two or three.
//
// So the key is (region, envelope stage, phase bucket): region first so a warp
// shares a sample, stage next to cut branch divergence, and a coarsened phase
// last so the lanes that share a sample also share cache lines.
//
// Least-significant-bit-first, one bit per pass, each pass a stable binary
// split built on the same prefix sum the compaction uses. One bit at a time is
// more passes than a byte at a time would be, but each pass only moves an
// 8-byte (key, index) pair, and the whole sort costs a couple of milliseconds
// against a render pass measured in hundreds.
//
// Determinism: a binary split is stable by construction and uses no atomics,
// so the permutation is a pure function of the pool contents.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> pairs_in: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> pairs_out: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> scan: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<u32>;
@group(0) @binding(6) var<storage, read_write> voices: array<u32>;
@group(0) @binding(7) var<storage, read_write> voices_out: array<u32>;

var<workgroup> sh_scan: array<u32, WG>;

@compute @workgroup_size(1)
fn init() {
    state[S_SORT_BIT] = 0u;
}

@compute @workgroup_size(1)
fn advance_bit() {
    state[S_SORT_BIT] = state[S_SORT_BIT] + 1u;
}

@compute @workgroup_size({{WG}})
fn build_keys(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let live = state[S_LIVE];

    var i = gid.x;
    loop {
        if (i >= live) { break; }
        let stage = voices[F_ENV_STAGE * c + i];
        // Dead voices get a region one past the last real one, so they sort
        // to the end and the gather simply stops before reaching them.
        var region = u.sort_dead_region;
        if (stage != ENV_DEAD) {
            region = min(voices[F_REGION * c + i], u.sort_dead_region - 1u);
        }
        let ph = (voices[F_PHASE_HI * c + i] >> u.sort_phase_shift) & u.sort_phase_mask;
        let key = (region << u.sort_region_shift)
            | (min(stage, 7u) << u.sort_stage_shift)
            | ph;
        pairs_in[i] = vec2<u32>(key, i);
        i = i + stride;
    }
}

@compute @workgroup_size({{WG}})
fn scan_local(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let bit = state[S_SORT_BIT];
    let live = state[S_LIVE];
    // Same grid-stride over blocks as `compact.wgsl::scan_local`, and for the
    // same reason: the pool may hold more WG-sized blocks than one dispatch
    // dimension can address.
    let blocks = (live + WG - 1u) / WG;
    var block = wgid.x;
    loop {
        if (block >= blocks) { break; }
        let i = block * WG + tid;

        var zero = 0u;
        if (i < live && ((pairs_in[i].x >> bit) & 1u) == 0u) { zero = 1u; }

        sh_scan[tid] = zero;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if (offset >= WG) { break; }
            var v = 0u;
            if (tid >= offset) { v = sh_scan[tid - offset]; }
            workgroupBarrier();
            sh_scan[tid] = sh_scan[tid] + v;
            workgroupBarrier();
            offset = offset << 1u;
        }

        if (i < u.capacity) {
            scan[i] = sh_scan[tid] - zero;
        }
        if (tid == WG - 1u) {
            block_sums[block] = sh_scan[tid];
        }
        workgroupBarrier();
        block = block + nwg.x;
    }
}

@compute @workgroup_size({{WG}})
fn scan_blocks(@builtin(local_invocation_index) tid: u32) {
    let blocks = (state[S_LIVE] + WG - 1u) / WG;
    var running = 0u;
    var chunk = 0u;
    loop {
        if (chunk * WG >= blocks) { break; }
        let i = chunk * WG + tid;
        var v = 0u;
        if (i < blocks) { v = block_sums[i]; }

        sh_scan[tid] = v;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if (offset >= WG) { break; }
            var t = 0u;
            if (tid >= offset) { t = sh_scan[tid - offset]; }
            workgroupBarrier();
            sh_scan[tid] = sh_scan[tid] + t;
            workgroupBarrier();
            offset = offset << 1u;
        }

        let total = sh_scan[WG - 1u];
        if (i < blocks) {
            block_sums[i] = running + sh_scan[tid] - v;
        }
        workgroupBarrier();
        running = running + total;
        chunk = chunk + 1u;
    }
    if (tid == 0u) {
        state[S_TOTAL0] = running;
    }
}

@compute @workgroup_size({{WG}})
fn split(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let live = state[S_LIVE];
    let bit = state[S_SORT_BIT];
    let total0 = state[S_TOTAL0];

    var i = gid.x;
    loop {
        if (i >= live) { break; }
        let p = pairs_in[i];
        let zeros_before = block_sums[i / WG] + scan[i];
        var dst = zeros_before;
        if (((p.x >> bit) & 1u) != 0u) {
            // Ones go after every zero, keeping their relative order.
            dst = total0 + (i - zeros_before);
        }
        if (dst < u.capacity) {
            pairs_out[dst] = p;
        }
        i = i + stride;
    }
}

/// Move the surviving voices into their sorted slots. One copy of the pool per
/// block: this does the compaction and the reordering together, because the
/// permutation already has the dead voices pushed past `S_LIVE_NEW`.
@compute @workgroup_size({{WG}})
fn gather(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let n = state[S_LIVE_NEW];

    var j = gid.x;
    loop {
        if (j >= n) { break; }
        let src = pairs_in[j].y;
        if (src < c) {
            for (var f = 0u; f < VOICE_FIELDS; f = f + 1u) {
                voices_out[f * c + j] = voices[f * c + src];
            }
            // Start offsets and the born-under variant only apply to the
            // block a voice was born in.
            voices_out[F_START_REL * c + j] = 0u;
            voices_out[F_BORN_VARIANT * c + j] = 0u;
            voices_out[F_STOP_REL * c + j] = 0u;
        }
        j = j + stride;
    }
}
