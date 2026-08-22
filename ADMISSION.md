# Admission, and the 76 GiB allocation

A user rendering `DYHTM Community Merge.mid` on an RTX 3060 hit

```
memory allocation of 81604378624 bytes failed
```

partway through. This is what that was, and what changed.

## Reading the number

81,604,378,624 bytes is exactly 76 GiB. `size_of::<SpawnCmd>()` is 76, and
81,604,378,624 / 76 is exactly 2^30 — so the failed allocation was a
`Vec<SpawnCmd>` doubling from 2^29. Only three such vectors exist:
`Driver::spawn_buf`, `GpuSynth::pending_spawns` (a straight copy of it), and
`spawn_scratch` (bounded by `max_voices`, so not this one).

One 85 ms block had queued more than **536,870,912** spawn commands.

## The file

| | |
|---|---|
| size | 6.6 GB |
| tracks | 4,393 |
| note-ons | **824,167,030** |
| pitch bends | 2,675,662 |
| duration | ~378 s |

Reproduced on an RTX 5060 against a one-region soundfont — the *least*
favourable case, one voice per note-on, and it still reproduced. Rendering at
`--block 512` and reconstructing wall time from the cumulative realtime figure
gives the shape of it:

| audio reached | +wall | +audio | +note-ons |
|---|---|---|---|
| 286.63 s | 1.4 s | 30 ms | 6,102,991 |
| 286.66 s | 1.5 s | 30 ms | 4,237,610 |
| **286.67 s** | **18.9 s** | **10 ms** | **94,289,755** |
| 286.70 s | 7.2 s | 10 ms | 2,122,269 |

One 512-frame block took 94 million note-ons and 18.9 seconds. At the
4096-frame default the same spike gathers roughly eight times as many, which
lands on the half-billion that failed.

The spike is spread across ticks rather than piled on one. That is what makes
block width a usable lever at all — half a billion note-ons on a *single* tick
would land in one block no matter how narrow it is.

## Why the renderer did that

Admission ranks note-ons by `voice::admit_key`, whose middle field is the
voice's real opening gain — velocity, the region's own attenuation, and its
pan — rather than a raw velocity byte. That gain came out of `build_voice`. So
ranking a block meant building every voice in it first, including the 98.9% a
saturated block cannot admit, at 76 bytes each and held twice: once in the
driver and once in the backend.

Peak host memory, sampled across a `--block 512` run: **14.73 GiB.**

## What changed

The gain turns out not to need the build. `gain_l`/`gain_r` are a function of
`(region, velocity)` alone, and whether a layer exists at all is a function of
`(region, key)` alone. Both are precomputed at bank load, and
`Bank::preview_note_on` answers admission's question in two table lookups per
layer. It is exact, not approximate — the same layers in the same order with
gains equal bit for bit — so moving admission earlier changed nothing about
which notes are admitted.

The block now records candidates instead of voices: 8 bytes per note-on packed
plus 4 of ordinal, and 24 bytes per layer. `Config::admit_take` decides how many
survive, the stratified ranking reorders candidates, and `build_layer` runs for
the winners alone. The backends receive exactly what they will keep.

## What it bought

| | before | after | |
|---|---|---|---|
| peak host memory, DYHTM | 14.73 GiB | **4.79 GiB** | 3.07x |
| the spike block | 18.94 s | **10.61 s** | 1.79x |
| DYHTM, whole file | 244.3 s | 243.3 s | noise |
| Hypernova, whole file | 91.10 s | 87.68 s | 3.8% |

Both renders are byte-identical to the versions before the change, with every
count matching: Hypernova `91708ae5…`, DYHTM `699f0037…`.

**This is a memory fix that happens to be about 4% faster, not a throughput
fix.** The distinction is worth stating because the profile invites the opposite
conclusion: on the spike block the GPU did 42.8 ms of work against 18,941 ms
of wall, which is 99.77% host — but "host work" is not the same as "work
admission can skip". Culling removes *materialisation*, not *enumeration*:
every note-on in the block still has to be previewed, packed and ranked, and
only the 76-byte build and the copying go away. Hence 1.79x on the worst block
rather than the order of magnitude the 99.77% suggests.

## If you are hitting this

Host memory scales with **note-ons per block**, so the lever is `--block`:

- `--block 512` renders DYHTM in full at ~4.8 GiB.
- `--block 256` and `--block 128` halve it again each. 128 is the floor, since
  `steal_fade_frames` defaults to 96 and the block must exceed it while staying
  a multiple of `--gate-frames`.
- `--layers 1` cuts it by up to 16x on a layered soundfont, but it changes what
  you hear. A fallback, not a fix.
- **Lowering `--max-voices` does not help.** Every note-on in the block is
  considered before admission thins it; that is the whole shape of the problem.

Culling widened the margin by about 3x. It did not remove the need for the
lever: at `--block 4096` this file still wants ~28 GB of candidates.
