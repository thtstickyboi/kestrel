// Pass 4 of 4: compact.
//
// Exclusive prefix sum over the liveness mask, then a stream compaction that
// copies survivors into the back buffer. Order is preserved, so which voices
// died never changes the order of the ones that lived.
//
// Three entry points, run in sequence:
//   scan_local   one workgroup per WG voices, scans the mask locally
//   scan_blocks  one workgroup total, scans the per-workgroup totals
//   scatter      moves survivors, and writes the sort key when sorting is on
//
// When voice sorting is on, `scatter` is not used at all: the counting half
// still runs to produce the live count, and `sort.wgsl` does the compaction
// and the reordering together in a single copy of the pool.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> voices: array<u32>;
@group(0) @binding(2) var<storage, read_write> voices_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> scan: array<u32>;
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<u32>;
@group(0) @binding(6) var<storage, read_write> sort_keys: array<u32>;

var<workgroup> sh_scan: array<u32, WG>;

fn is_alive(i: u32) -> bool {
    return voices[F_ENV_STAGE * u.capacity + i] != ENV_DEAD;
}

@compute @workgroup_size({{WG}})
fn scan_local(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let live = state[S_LIVE];
    let i = wgid.x * WG + tid;
    var alive = 0u;
    if (i < live && is_alive(i)) { alive = 1u; }

    sh_scan[tid] = alive;
    workgroupBarrier();

    // Hillis-Steele inclusive scan. WG is a power of two.
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
        scan[i] = sh_scan[tid] - alive; // exclusive
    }
    if (tid == WG - 1u) {
        block_sums[wgid.x] = sh_scan[tid];
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
            block_sums[i] = running + sh_scan[tid] - v; // exclusive, global
        }
        workgroupBarrier();
        running = running + total;
        chunk = chunk + 1u;
    }
    if (tid == 0u) {
        state[S_LIVE_NEW] = running;
    }
}

@compute @workgroup_size({{WG}})
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let live = state[S_LIVE];
    var i = gid.x;
    loop {
        if (i >= live) { break; }
        if (is_alive(i)) {
            let dst = block_sums[i / WG] + scan[i];
            if (dst < c) {
                for (var f = 0u; f < VOICE_FIELDS; f = f + 1u) {
                    voices_out[f * c + dst] = voices[f * c + i];
                }
                // Start offsets and the born-under variant only apply to
                // the block a voice was born in.
                voices_out[F_START_REL * c + dst] = 0u;
                voices_out[F_BORN_VARIANT * c + dst] = 0u;
                voices_out[F_STOP_REL * c + dst] = 0u;
                // Sort key: region first so neighbouring invocations read the
                // same sample data, envelope stage second to cut divergence.
                sort_keys[dst] = (min(voices[F_REGION * c + i], 0x1FFFFFFFu) << 3u)
                    | min(voices[F_ENV_STAGE * c + i], 7u);
            }
        }
        i = i + stride;
    }
}

// Marks the k oldest voices dead, where the threshold came out of the radix
// select in select.wgsl. Note ids are unique, so "note_id <= threshold" names
// exactly k voices.
@compute @workgroup_size({{WG}})
fn mark_stolen(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let t_hi = state[S_THRESH_HI];
    let t_lo = state[S_THRESH_LO];
    let live = state[S_LIVE];
    var i = gid.x;
    loop {
        if (i >= live) { break; }
        let lo = voices[F_NOTE_LO * c + i];
        var lvl = 0u;
        if (u.steal_by_level != 0u) { lvl = voices[F_ENV_LEVEL * c + i]; }
        let k = steal_key(voices[F_NOTE_HI * c + i], lo, lvl);
        if (!less64(t_hi, t_lo, k.x, k.y)) { // key <= threshold
            // Not killed here. Cutting every victim at frame zero is what put
            // a step at the block boundary; instead each one stops at a frame
            // of its own and fades out there. Victims are a contiguous range
            // of note ids, so the low word spreads them evenly across the
            // block, and it needs no ordering or coordination to compute.
            let span = u.block_frames - STEAL_FADE;
            voices[F_STOP_REL * c + i] = (lo % span) + 1u;
        }
        i = i + stride;
    }
}

// Publish the compacted count. Separate dispatch so the scatter above is done
// reading the old count before it changes.
@compute @workgroup_size(1)
fn commit() {
    state[S_LIVE] = state[S_LIVE_NEW];
}

@compute @workgroup_size(1)
fn note_stolen() {
    state[S_STOLEN] = state[S_STOLEN] + u.steal_k;
}
