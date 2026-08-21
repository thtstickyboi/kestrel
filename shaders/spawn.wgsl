// Pass 1 of 4: spawn.
//
// Appends this block's note-ons to the end of the voice pool. The destination
// slot is `spawn_base + i` where i is the command's index in the host's event
// order, so allocation never depends on which invocation runs first. No
// atomics, and the pool contents are a pure function of the MIDI file.

struct SpawnCmd {
    phase_lo: u32,
    phase_hi: u32,
    step_lo: u32,
    step_hi: u32,
    smp_base: u32,
    smp_len: u32,
    loop_start: u32,
    loop_end: u32,
    flags: u32,
    params: u32,
    variant: u32,
    region: u32,
    gate_slot: u32,
    ordinal: u32,
    start_rel: u32,
    note_id_lo: u32,
    note_id_hi: u32,
    gain_l: f32,
    gain_r: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cmds: array<SpawnCmd>;
@group(0) @binding(2) var<storage, read_write> voices: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: array<u32>;

@compute @workgroup_size({{WG}})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WG;
    let c = u.capacity;
    let base = state[S_LIVE];
    var i = gid.x;
    loop {
        if (i >= u.spawn_count) { break; }
        let dst = base + i;
        if (dst < c) {
            let s = cmds[i];
            voices[F_PHASE_LO * c + dst] = s.phase_lo;
            voices[F_PHASE_HI * c + dst] = s.phase_hi;
            voices[F_STEP_LO * c + dst] = s.step_lo;
            voices[F_STEP_HI * c + dst] = s.step_hi;
            voices[F_SMP_BASE * c + dst] = s.smp_base;
            voices[F_SMP_LEN * c + dst] = s.smp_len;
            voices[F_LOOP_START * c + dst] = s.loop_start;
            voices[F_LOOP_END * c + dst] = s.loop_end;
            voices[F_FLAGS * c + dst] = s.flags;
            voices[F_ENV_STAGE * c + dst] = ENV_ATTACK;
            voices[F_ENV_LEVEL * c + dst] = bitcast<u32>(0.0);
            voices[F_GAIN_L * c + dst] = bitcast<u32>(s.gain_l);
            voices[F_GAIN_R * c + dst] = bitcast<u32>(s.gain_r);
            voices[F_FILT_Z1 * c + dst] = bitcast<u32>(0.0);
            voices[F_FILT_Z2 * c + dst] = bitcast<u32>(0.0);
            voices[F_PARAMS * c + dst] = s.params;
            voices[F_REGION * c + dst] = s.region;
            voices[F_GATE_SLOT * c + dst] = s.gate_slot;
            voices[F_ORDINAL * c + dst] = s.ordinal;
            voices[F_START_REL * c + dst] = s.start_rel;
            voices[F_NOTE_LO * c + dst] = s.note_id_lo;
            voices[F_NOTE_HI * c + dst] = s.note_id_hi;
            // Plus one, so that zero can mean "not born in this block" once
            // the render pass clears it.
            voices[F_BORN_VARIANT * c + dst] = s.variant + 1u;
            voices[F_STOP_REL * c + dst] = 0u;
        }
        i = i + stride;
    }
}

// Advance the live count once the appends are done. One invocation, so there
// is nothing to race.
@compute @workgroup_size(1)
fn commit() {
    let live = state[S_LIVE];
    let room = u.capacity - min(live, u.capacity);
    let taken = min(u.spawn_count, room);
    state[S_LIVE] = live + taken;
    state[S_DROPPED] = state[S_DROPPED] + (u.spawn_count - taken);
}
