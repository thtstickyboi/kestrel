// Pass 3 of 4: reduce.
//
// Sums the per-workgroup partials into the stereo output block. One workgroup
// per output sample; each thread takes a strided share of the partials and the
// workgroup finishes with a fixed-order binary tree.
//
// The tree matters for more than speed. Summing a million voice contributions
// sequentially in f32 accumulates error proportional to the term count; a tree
// makes it proportional to log2 of the term count. Kahan compensation on the
// per-thread partial sums is available for the cases where even that is not
// enough.

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_block: array<f32>;

var<workgroup> red: array<f32, WG>;

const KAHAN: bool = {{KAHAN}};

@compute @workgroup_size({{WG}})
fn main(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let j = wgid.x;
    if (j >= u.block_frames * 2u) { return; }

    let nwg = u.render_workgroups;
    let base = j * nwg;

    var acc = 0.0;
    if (KAHAN) {
        var comp = 0.0;
        for (var w = tid; w < nwg; w = w + WG) {
            let y = partials[base + w] - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
    } else {
        for (var w = tid; w < nwg; w = w + WG) {
            acc = acc + partials[base + w];
        }
    }
    red[tid] = acc;
    workgroupBarrier();

    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out_block[j] = red[0];
    }
}
