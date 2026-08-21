// Pass 2 of 4: render.
//
// One invocation per voice. Each voice's state stays in registers for the whole
// block, so the sequential parts of the DSP (phase accumulation, envelope,
// filter memory) are never round-tripped through memory. Frames are processed
// in tiles of TILE; at the end of each tile the workgroup reduces its
// TILE * 2 channel lanes across all WG voices and adds the result into this
// workgroup's own slice of the partial buffer.
//
// Per-frame results go straight into workgroup storage rather than into a
// per-invocation array. An `array<f32, TILE>` in the function address space
// looks free but is not: the shader compiler spills it to local memory as soon
// as the frame loop stops being trivially unrollable, and every write then
// costs a memory round trip. Measured on an RTX 5060 with a million voices, a
// private array cost 232 ms per block at TILE=16 and 596 ms at TILE=32, where
// writing straight to workgroup storage does not grow with TILE that way.
//
// No atomics anywhere. Each workgroup owns its partial slice outright, and the
// in-workgroup reduction is a fixed-order tree, so two runs produce bitwise
// identical partials. Two renders of the same input must be byte-identical;
// that requirement is why there is no atomicAdd mixdown here.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> pool: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<RegionParams>;
@group(0) @binding(3) var<storage, read> gates: array<u32>;
@group(0) @binding(4) var<storage, read_write> voices: array<u32>;
// Frame-major: the partial for output sample j from workgroup w sits at
// j * NWG + w, so the reduce pass reads contiguously across workgroups.
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read> state: array<u32>;
@group(0) @binding(7) var<storage, read> chan: array<u32>;

// Lanes reduced per tile: TILE frames times two channels, interleaved so a
// lane index is already an offset into the interleaved output block.
// Whether this build of the pass carries the channel controller path at all.
// Two pipelines are compiled, and the host picks per block. The branch below
// is uniform across the whole dispatch, but a runtime-uniform branch still
// costs the registers that hold the unbent step and the unscaled gains for
// every voice, whether or not the block uses them: measured 1.8 ms per block
// at a million voices, or 1.5% of the budget, charged to files that never
// send a controller. A `const` lets the compiler delete them instead.
const CHAN_ENABLED: bool = {{CHAN}};

const M: u32 = TILE * 2u;
// Threads cooperating on one lane during the first reduction level.
const PER_LANE: u32 = WG / M;

var<workgroup> sh: array<f32, WG * M>;
var<workgroup> sh2: array<f32, WG>;

// One 32-bit word of the pool holds two consecutive frames.
fn word_pair(w: u32) -> vec2<f32> {
    if (w >= u.pool_words) { return vec2<f32>(0.0, 0.0); }
    return unpack2x16snorm(pool[w]);
}

fn fetch(base: u32, idx: u32) -> f32 {
    let i = base + idx;
    let p = word_pair(i >> 1u);
    if ((i & 1u) == 1u) { return p.y; }
    return p.x;
}

// The frame right after `idx`. Specialised out of `neighbour_index` because
// this is the hot one: idx is always below `le`, so the wrap can only ever
// land exactly on the loop start and the integer modulo is unnecessary. The
// result is identical to the general path the CPU reference uses.
fn next_index(idx: u32, looping: bool, ls: u32, le: u32, len: u32) -> u32 {
    let n = idx + 1u;
    if (looping) {
        if (n >= le) { return ls; }
        return n;
    }
    if (n >= len) { return len - 1u; }
    return n;
}

fn interpolate(
    base: u32, idx: u32, frac: f32,
    looping: bool, ls: u32, le: u32, len: u32
) -> f32 {
    if (u.interp == INTERP_NEAREST) {
        return fetch(base, idx);
    }
    if (u.interp == INTERP_LINEAR) {
        // Both taps come out of one word whenever the first one is even, which
        // is exactly half the time and, because the pool is phase-sorted,
        // usually the same half for every lane in a warp. So address the pair
        // from a single word index and only issue a second load on the odd
        // case, rather than calling `fetch` twice and recomputing the address,
        // the bounds check and the unpack for a word that is often the one
        // already in hand.
        let i = base + idx;
        let w = i >> 1u;
        let p0 = word_pair(w);
        let p1 = word_pair(w + 1u);
        let odd = (i & 1u) == 1u;
        let s0 = select(p0.x, p0.y, odd);
        var s1 = select(p0.y, p1.x, odd);
        // The tap after `idx` is `idx + 1` except where it runs off a loop end
        // or the end of the sample. That is rare enough to be a fixup instead
        // of a term in the hot path.
        let n = next_index(idx, looping, ls, le, len);
        if (n != idx + 1u) { s1 = fetch(base, n); }
        return s0 + (s1 - s0) * frac;
    }
    let im1 = neighbour_index(idx, -1, looping, ls, le, len);
    let i1 = neighbour_index(idx, 1, looping, ls, le, len);
    let i2 = neighbour_index(idx, 2, looping, ls, le, len);
    let sm1 = fetch(base, im1);
    let s0 = fetch(base, idx);
    let s1 = fetch(base, i1);
    let s2 = fetch(base, i2);
    let a = -0.5 * sm1 + 1.5 * s0 - 1.5 * s1 + 0.5 * s2;
    let b = sm1 - 2.5 * s0 + 2.0 * s1 - 0.5 * s2;
    let c = -0.5 * sm1 + 0.5 * s1;
    return ((a * frac + b) * frac + c) * frac + s0;
}

// Sum sh[] across the workgroup, M lanes at a time, and add the totals into
// this workgroup's partial slice. Two levels, so the barrier count per tile is
// three rather than log2(WG). The strided read in level one keeps a warp
// spread across shared-memory banks instead of piling onto two of them.
fn reduce_into_partials(tid: u32, wg: u32, nwg: u32, first_sample: u32) {
    let lane = tid / PER_LANE;
    let chunk = tid % PER_LANE;

    var acc = 0.0;
    for (var k = 0u; k < M; k = k + 1u) {
        acc = acc + sh[lane * WG + chunk + k * PER_LANE];
    }
    sh2[lane * PER_LANE + chunk] = acc;
    workgroupBarrier();

    if (tid < M) {
        var total = 0.0;
        for (var k = 0u; k < PER_LANE; k = k + 1u) {
            total = total + sh2[tid * PER_LANE + k];
        }
        let j = first_sample + tid;
        partials[j * nwg + wg] = partials[j * nwg + wg] + total;
    }
}

@compute @workgroup_size({{WG}})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let wg = wgid.x;
    let nwg = u.render_workgroups;
    let live = state[S_LIVE];
    let samples = u.block_frames * 2u;

    // Clear this workgroup's partial slice. Workgroups with no voices still
    // have to do this, or the reduce pass folds in the previous block's audio.
    for (var j = tid; j < samples; j = j + WG) {
        partials[j * nwg + wg] = 0.0;
    }
    workgroupBarrier();

    var batch = wg;
    loop {
        if (batch * WG >= live) { break; }
        let v = batch * WG + tid;
        let is_live = v < live;

        // ---- load voice state into registers ----
        var phase_hi = 0u;
        var phase_lo = 0u;
        var step_hi = 0u;
        var step_lo = 0u;
        var smp_base = 0u;
        var smp_len = 1u;
        var loop_start = 0u;
        var loop_end = 1u;
        var vflags = 0u;
        var stage = ENV_DEAD;
        var level = 0.0;
        var gain_l = 0.0;
        var gain_r = 0.0;
        var z1 = 0.0;
        var z2 = 0.0;
        var gate_slot = 0u;
        var ordinal = 0u;
        var start_rel = 0u;
        // Frame this voice was stolen at, plus one; zero if it was not.
        var stop_rel = 0u;
        var p: RegionParams;
        var use_filter = false;
        var params_base = 0u;
        var variant = 0u;
        // Non-zero only for a voice born in this block, and then it is the
        // variant current at its own note-on plus one.
        var born_variant = 0u;

        var base_step_hi = 0u;
        var base_step_lo = 0u;
        var base_gain_l = 0.0;
        var base_gain_r = 0.0;

        if (is_live) {
            let c = u.capacity;
            phase_lo = voices[F_PHASE_LO * c + v];
            phase_hi = voices[F_PHASE_HI * c + v];
            // The pool holds the note's unbent step. When any channel is bent
            // the effective step is this scaled by the channel's factor,
            // refreshed once per gate tile below; when nothing is bent the two
            // are the same value and the multiply never happens.
            base_step_lo = voices[F_STEP_LO * c + v];
            base_step_hi = voices[F_STEP_HI * c + v];
            step_lo = base_step_lo;
            step_hi = base_step_hi;
            smp_base = voices[F_SMP_BASE * c + v];
            smp_len = voices[F_SMP_LEN * c + v];
            loop_start = voices[F_LOOP_START * c + v];
            loop_end = voices[F_LOOP_END * c + v];
            vflags = voices[F_FLAGS * c + v];
            stage = voices[F_ENV_STAGE * c + v];
            level = bitcast<f32>(voices[F_ENV_LEVEL * c + v]);
            base_gain_l = bitcast<f32>(voices[F_GAIN_L * c + v]);
            base_gain_r = bitcast<f32>(voices[F_GAIN_R * c + v]);
            gain_l = base_gain_l;
            gain_r = base_gain_r;
            z1 = bitcast<f32>(voices[F_FILT_Z1 * c + v]);
            z2 = bitcast<f32>(voices[F_FILT_Z2 * c + v]);
            gate_slot = voices[F_GATE_SLOT * c + v];
            ordinal = voices[F_ORDINAL * c + v];
            start_rel = voices[F_START_REL * c + v];
            stop_rel = voices[F_STOP_REL * c + v];
            params_base = voices[F_PARAMS * c + v];
            born_variant = voices[F_BORN_VARIANT * c + v];
            if (born_variant != 0u) { variant = born_variant - 1u; }
            p = params[params_base + variant * u.params_per_variant];
            use_filter = (p.flags & RP_FILTER) != 0u;
        }

        let loop_enabled = (vflags & VF_LOOP) != 0u;
        let loop_until_release = (vflags & VF_LOOP_UNTIL_RELEASE) != 0u;
        // Voices already carry their gate slot for the note-off lookup, and
        // the channel is the top of it, so bend needs no extra pool field.
        let channel = gate_slot >> 7u;
        // The gate tile this voice starts in. Only meaningful while
        // `born_variant` says the voice was born in this block; after that
        // `start_rel` has been cleared and every tile is one it was alive for.
        let born_gate = start_rel / GATE_TILE;

        for (var tile = 0u; tile < u.tiles; tile = tile + 1u) {
            // Note-off gate, sampled once per GATE_TILE frames. The table row
            // is 8 KiB so it stays in L1, but at a million voices even a
            // cached read per voice per reduce tile costs several milliseconds
            // a block, hence the coarser cadence.
            if (is_live && (tile % TILES_PER_GATE) == 0u) {
                let gt = tile / TILES_PER_GATE;
                if (stage < ENV_RELEASE
                    && gates[gt * GATE_SLOTS + gate_slot] >= ordinal) {
                    stage = ENV_RELEASE;
                    level = min(level, 1.0);
                }
                // Releasing voices bend and fade too, so this is not folded
                // into the gate check above. One channel's entry is four
                // adjacent words, so both reads come off the same cache line.
                if (CHAN_ENABLED && u.chan_active != 0u) {
                    let ci = (gt * BEND_CHANNELS + channel) * CHAN_FIELDS;
                    if ((u.chan_active & CHAN_ACTIVE_BEND) != 0u) {
                        let st = scale64(
                            base_step_hi, base_step_lo, chan[ci + CHAN_BEND]
                        );
                        step_hi = st.x;
                        step_lo = st.y;
                    }
                    if ((u.chan_active & CHAN_ACTIVE_GAIN) != 0u) {
                        gain_l = base_gain_l * bitcast<f32>(chan[ci + CHAN_GAIN_L]);
                        gain_r = base_gain_r * bitcast<f32>(chan[ci + CHAN_GAIN_R]);
                    }
                    // CC71-CC75 change things that live in RegionParams, so
                    // the voice re-reads its constants from a different copy
                    // of the table. Only when the copy actually changes: an
                    // unconditional reload would be a scattered load per voice
                    // per gate tile, and these controllers move a few hundred
                    // times in a whole file.
                    let want = select(0u, chan[ci + CHAN_VARIANT],
                                      (u.chan_active & CHAN_ACTIVE_VARIANT) != 0u);
                    // A row holds the state at the start of its tile, which is
                    // older than a voice born inside that tile: the voice
                    // already carries the variant that was current at its own
                    // frame. So rows govern it only from the tile after the one
                    // it was born in.
                    let born_here = born_variant != 0u && gt <= born_gate;
                    if (want != variant && !born_here) {
                        variant = want;
                        p = params[params_base + variant * u.params_per_variant];
                        use_filter = (p.flags & RP_FILTER) != 0u;
                    }
                }
            }

            let f0 = tile * TILE;
            for (var i = 0u; i < TILE; i = i + 1u) {
                var y = 0.0;
                let f = f0 + i;

                if (is_live && stage != ENV_DEAD && f >= start_rel) {
                    let looping = loop_enabled
                        && !(loop_until_release && stage >= ENV_RELEASE);

                    if (!looping && phase_hi >= smp_len) {
                        stage = ENV_DEAD;
                    } else {
                        // Envelope first, so an instant attack is at full
                        // level on the voice's very first frame.
                        if (stage == ENV_ATTACK) {
                            level = level + p.attack_rate;
                            if (level >= p.attack_end) {
                                stage = ENV_DECAY;
                                level = 1.0;
                            }
                        } else if (stage == ENV_DECAY) {
                            if (u.exp_decay != 0u) {
                                level = level * p.decay_coef;
                            } else {
                                level = level - p.decay_coef;
                            }
                            if (level <= p.decay_target) {
                                if (p.sustain <= u.env_floor) {
                                    stage = ENV_DEAD;
                                    level = 0.0;
                                } else {
                                    stage = ENV_SUSTAIN;
                                    level = p.sustain;
                                }
                            }
                        } else if (stage == ENV_RELEASE) {
                            if (u.exp_release != 0u) {
                                level = level * p.release_coef;
                            } else {
                                level = level - p.release_coef;
                            }
                            if (level <= u.env_floor) {
                                stage = ENV_DEAD;
                                level = 0.0;
                            }
                        }

                        let s = interpolate(
                            smp_base, phase_hi, frac_of(phase_lo),
                            looping, loop_start, loop_end, smp_len
                        );
                        let g = min(level, 1.0);
                        let x = s * g;

                        // Transposed direct form II. b2 == b0.
                        y = x;
                        if (use_filter) {
                            y = p.b0 * x + z1;
                            z1 = p.b1 * x - p.a1 * y + z2;
                            z2 = p.b0 * x - p.a2 * y;
                        }

                        // A stolen voice fades to silence over STEAL_FADE
                        // frames from its own stop frame, rather than every
                        // victim being cut together at the top of the block.
                        // After the filter, so the biquad keeps seeing the
                        // untapered signal and does not ring on the taper.
                        if (stop_rel != 0u && f + 1u >= stop_rel) {
                            let d = f + 1u - stop_rel;
                            if (d >= STEAL_FADE) {
                                y = 0.0;
                                stage = ENV_DEAD;
                            } else {
                                y = y * (1.0 - f32(d) / f32(STEAL_FADE));
                            }
                        }

                        let np = add64(phase_hi, phase_lo, step_hi, step_lo);
                        phase_hi = np.x;
                        phase_lo = np.y;
                        if (looping && phase_hi >= loop_end) {
                            // One subtraction covers any step shorter than the
                            // loop; the modulo is only there for extreme
                            // pitches, and gives the same answer either way.
                            let span = max(loop_end - loop_start, 1u);
                            phase_hi = phase_hi - span;
                            if (phase_hi >= loop_end) {
                                phase_hi = loop_start + (phase_hi - loop_start) % span;
                            }
                        }
                    }
                }

                sh[(i * 2u) * WG + tid] = y * gain_l;
                sh[(i * 2u + 1u) * WG + tid] = y * gain_r;
            }

            // Two barriers per tile, not three. The barrier inside the reduce
            // already separates level one's reads of `sh` from the next tile's
            // writes to it, and the barrier at the top of the next tile
            // separates level two's reads of `sh2` from the next level one's
            // writes. A third barrier here would guard nothing.
            workgroupBarrier();
            reduce_into_partials(tid, wg, nwg, f0 * 2u);
        }

        // ---- write voice state back ----
        if (is_live) {
            let c = u.capacity;
            voices[F_PHASE_LO * c + v] = phase_lo;
            voices[F_PHASE_HI * c + v] = phase_hi;
            voices[F_ENV_STAGE * c + v] = stage;
            voices[F_ENV_LEVEL * c + v] = bitcast<u32>(level);
            voices[F_FILT_Z1 * c + v] = bitcast<u32>(z1);
            voices[F_FILT_Z2 * c + v] = bitcast<u32>(z2);
            voices[F_START_REL * c + v] = 0u;
            voices[F_BORN_VARIANT * c + v] = 0u;
            voices[F_STOP_REL * c + v] = 0u;
        }

        batch = batch + nwg;
    }
}
