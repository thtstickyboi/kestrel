# Kestrel

'*experimental*'

A GPU-accelerated SoundFont/SFZ synthesizer for Black MIDI.

Kestrel renders a MIDI file to a WAV offline, doing the synthesis in compute
shaders instead of on the CPU. It is built for files where the note count is
absurd -- hundreds of millions of notes, with a hundred thousand to several
million voices sounding at once -- which is the point where conventional synths
stop being fast enough.

On an RTX 5060 Laptop it sustains **1,048,576 concurrent voices at 1.02x
realtime**.

It is a library and a command-line tool. There is no GUI, no realtime playback,
and no plugin build.

---

## Read this first: Kestrel is 100% vibecoded

**Every line of code in this repository was written by an AI.** Not
AI-assisted, not AI-reviewed -- AI-written, start to finish, from the shaders to
the SF2 parser to the limiter. I directed the work, made the design calls, ran
the renders and decided what was acceptable, but I did not hand-write this code
and I have not read every line of it.

What that means for you, concretely:

- **No human has line-by-line reviewed this code.** Not me, not anyone. If you
  are the sort of person who reads a diff before running it, read this one.
- **It was developed against a test suite**, which is *not* included in this
  repository: null tests against a separate single-threaded CPU reference
  implementation, byte-for-byte determinism checks, phase-accumulator precision
  tests, and envelope curves checked against analytically computed ones. The
  GPU path matched the CPU reference to better than -100 dB. That is real
  verification and it caught real bugs -- but you are taking my word for it,
  because the evidence is not in this repo.
- **The corners are where it will break.** The paths that got exercised
  constantly -- SF2 loading, the render loop, voice stealing under saturation --
  are in decent shape. Unusual soundfonts, exotic SFZ opcodes and malformed
  MIDI are much less certain.
- **Check the output yourself.** Do not put this in a pipeline you care about
  and assume the WAV is fine. Scan it. `kestrel null` exists precisely so you
  can diff a render against a reference you trust.

I am not claiming this is production software. I am claiming it renders black
MIDI fast and the output sounded right to me. Use it accordingly, and please
open an issue when it breaks, because it will.

---

## Requirements

| | |
|---|---|
| **Rust** | 1.80 or newer, from [rustup.rs](https://rustup.rs) |
| **GPU** | Anything with a working Vulkan, DX12 or Metal driver |
| **VRAM** | Depends on the soundfont and `--max-voices`; see below. Tuned against 8 GB |
| **OS** | Windows, Linux, macOS. Only Windows has been *run*; see below |

There are no system libraries to install and nothing to download separately.
Every dependency comes from crates.io, and the shaders are compiled into the
binary.

**Tested on NVIDIA and Intel, both on Windows via Vulkan.** The two agree:
rendering the same file on an RTX 5060 and on an Intel iGPU nulls at
**-111 dB peak, -127 dB RMS**, and each device is byte-identical with itself
across runs. That is worth more than it sounds -- WGSL is compiled by your
driver, not by cargo, so Intel's shader compiler is an entirely separate path
from NVIDIA's, and the two producing the same audio is real evidence the
shaders are portable.

Linux and macOS **compile** clean -- `cargo check --all-targets` passes for
`x86_64-unknown-linux-gnu` and `aarch64-apple-darwin`, which type-checks wgpu's
Vulkan and Metal backends along with everything else. Neither has been linked
or run, so that is one step short of a guarantee. There is no platform-specific
code in the crate and no C dependencies, which is the reason to expect it to
work rather than a promise that it does.

AMD, Apple silicon and Linux are still unrun. If you are first on one of
those, please open an issue either way; a report that it simply worked is as
useful as a crash.

Integrated GPUs are slower but genuinely usable -- the Intel part held 2.95x
realtime against the 5060's 32.5x on the same 100k-note file, which is about
what shared DDR5 against dedicated VRAM predicts for a bandwidth-bound
workload. Kestrel prefers a discrete GPU when both are present, so you will not
land on the slow one by accident.

Kestrel requests whatever limits your adapter reports rather than demanding a
fixed set, and it checks the ones it can before allocating: if the render pass
needs more workgroup storage than your card offers, or `--max-voices` implies
more compaction workgroups than your card allows, it stops and tells you which
flag to lower. It does not degrade quietly.

The GPU is the point, but there is a **complete CPU backend** -- it is the
reference implementation the GPU path was built against, so it produces correct
output on any machine. It is far slower and it is single-threaded. Use
`--backend cpu` if you have no usable GPU, or to check a suspicious render
against something simpler.

**VRAM is usually the binding constraint, not compute.** You do not have to
guess at it -- Kestrel prints the exact breakdown on startup:

```
gpu: NVIDIA GeForce RTX 5060 Laptop GPU (Vulkan) | 339.6 MiB of device buffers
(0.0 MiB sample pool, 240.0 MiB voice pool for 1310720 voices, 64.0 MiB partials)
```

Three things allocate:

- **The voice pool**, sized by `--max-voices`. The default of 1,048,576 costs
  240 MiB. It scales linearly, so 4M voices costs roughly 960 MiB.
- **The partial buffers** used by the mixdown, sized by `--render-workgroups`
  and `--block`. 64 MiB at the defaults.
- **The sample pool**, which is your entire soundfont resampled to one rate. A
  large multi-sampled piano runs to hundreds of megabytes.

If the sample pool will not fit within `--pool-budget` (2 GB by default),
Kestrel halves its sample rate until it does rather than failing, on the
grounds that a downsampled render beats no render. Lower that budget on a
smaller card, and lower `--max-voices` with it.

## Building

```bash
git clone https://github.com/thtstickyboi/kestrel-midi.git
```

```bash
cd kestrel-midi && cargo build --release
```

Build it in **release mode**. The debug build is not merely slower, it is
unusable for real files.

The binary lands at `target/release/kestrel`, or `target\release\kestrel.exe`
on Windows. It is self-contained, so you can copy it anywhere.

Check that your GPU was found before anything else:

```bash
kestrel gpu-info
```

If that lists no adapter, your driver is the problem, and `--backend cpu` is
the fallback.

## Running

The basic form:

```bash
kestrel render input.mid -s soundfont.sfz -o output.wav
```

`-s` takes a `.sf2` or a `.sfz`. Output is 32-bit float WAV by default; pass
`--format pcm16` for 16-bit.

A more realistic invocation for a large file:

```bash
kestrel render huge.mid -s piano.sfz -o out.wav --max-voices 2000000 --profile
```
```bash
kestrel render huge.mid -s piano.sfz -o out.wav --limiter brickwall --profile
```

### The flags that actually matter

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `gpu` | `cpu` is the reference implementation: correct, slow, single-threaded. |
| `--max-voices N` | `1048576` | Ceiling on simultaneous voices. The single biggest lever on both VRAM and speed. |
| `--interp` | `linear` | `nearest`, `linear` or `cubic`. Cubic costs roughly double the bandwidth. |
| `--seconds N` | off | Stop after N seconds of output. Use this constantly while experimenting. |
| `--profile` | off | Per-pass GPU timings, once per wall-clock second. |
| `--limiter` | `brickwall` | `brickwall`, `omni` or `off`. See *Limiting*. |
| `--steal` | `quietest` | `quietest`, `oldest` or `drop-new`. See *Voice stealing*. |
| `--admit` | `loudest` | `loudest` or `even`. Which note-ons survive when a block oversubscribes the pool. See *Admission*. |
| `--volume X` | `1.0` | Pre-limiter gain. |
| `--rate N` | `48000` | Output sample rate. |
| `--format` | `float32` | `float32` or `pcm16`. |
| `--nan-guard` | off | Check every block for NaN and Inf. **Off in release builds**, so a clean exit is not by itself proof the WAV is finite. |
| `--gpu-backend`, `--gpu-adapter` | off | Force a specific API or card on multi-GPU machines. |

`kestrel render --help` lists everything, including the tuning knobs
(`--block`, `--workgroup`, `--reduce-tile`, `--pool-budget`) that are best left
alone unless you are measuring.

### The other commands

```bash
kestrel gpu-info                  # list the GPU adapters wgpu can see
kestrel info file.sf2             # dump what the loader made of a soundfont
kestrel info file.mid             # ...or of a MIDI: note counts, CC usage, tempo
kestrel null a.wav b.wav          # peak difference between two renders, in dB
kestrel gen-assets dir/           # write synthetic soundfonts and MIDI
```

`kestrel info` is the first thing to reach for when a render sounds wrong. It
tells you what Kestrel *thinks* your file contains, which is often not what you
think it contains.

`gen-assets` writes reproducible synthetic material for benchmarking. Add
`--big-mb 512` for a sample pool too large to sit in cache, and
`--sustained 1500000` for a MIDI that holds the voice pool permanently full:

```bash
kestrel gen-assets bench --big-mb 512 --sustained 1500000
```

```bash
kestrel render bench/sustained1500000.mid -s bench/big512.sf2 -o bench.wav --profile --seconds 10
```

## Limiting

Black MIDI mixes clip constantly -- a saturated section can peak at hundreds of
times full scale -- so what happens at the ceiling matters more than usual.

**`brickwall`** (default) is a lookahead true-peak limiter. It sees peaks
before they arrive and cannot exceed its ceiling, so nothing downstream ever
has to hard-clip. `--ceiling-db`, `--lookahead-ms` and `--limiter-release-ms`
control it. The lookahead is also the render latency, which for an offline
renderer costs nothing.

**`omni`** is a port of the realtime limiter OmniConverter ships, originally
from Kiva. It is a feedback follower: it only sees a peak once the peak has
already passed, which is why it needs a third of a second of release, and that
release audibly drags the level down after every loud moment. It also does not
bound its own output, so samples still get clipped behind it. **Keep it only
for level-matching a render against BASS or XSynth**, which is the one thing it
does better.

**`off`** disables limiting. Your mix will clip. Combine with `--volume` if you
want to handle headroom yourself.

## Admission

Stealing decides who dies. **Admission decides who never starts**, and on a
saturated file that is the bigger number by an order of magnitude -- on a
44.7M-note file against a 32,767-voice pool, 83.7M note-ons were dropped
against 5.7M voices stolen. Whatever rule governs admission is deciding most of
what you hear.

**`loudest`** (default) ranks each oversubscribed block and keeps the top of
it. The rank is a 64-bit key: whether the note is still sounding at the end of
the block, then its opening amplitude, then its note id as a tiebreak. The
amplitude term is the voice's real opening gain -- it already carries the
velocity, the region's own attenuation and its pan -- rather than a raw
velocity byte. The id makes the key a total order with no ties, so the admitted
set is a pure function of the input.

**`even`** thins the block by position in time and ignores what each note is,
which is what the renderer did before ranking existed. It is blind in a
specific way: a fortissimo whole note and a 10 ms grace note have identical
odds of survival.

Measured on the same file and pool, `loudest` against `even`:

| | `even` | `loudest` |
|---|---|---|
| RMS | -11.225 dB | **-10.065 dB** |
| crest factor | 11.225 dB | **10.065 dB** |
| samples above 0.01 | 88.7% | **93.7%** |

A higher RMS at an identical peak, with a lower crest factor, is what "the long
loud notes survived" looks like as a number. Ranking costs about 5% of render
time.

One limit worth knowing: "still sounding at the end of the block" is resolved
at block granularity, not by true duration. A note that ends 5 ms into the next
block counts as sustained. It works because genuinely short notes almost always
end inside their own block, but it is an approximation rather than a
measurement.

## Voice stealing

When more notes arrive than the pool can hold, something has to give. The rule
is fixed and deterministic, never dependent on scheduling order, because that
would make renders non-reproducible.

**`quietest`** (default) kills the voices with the lowest envelope level, ties
broken by note id. These contribute least to the mix.

**`oldest`** kills the earliest-started voices. This sounds backwards under
saturation and it is: the oldest voices are the mature, sounding ones, while
the survivors are whichever were struck most recently and are therefore still
silent.

**`drop-new`** refuses the incoming note instead of killing an existing one.

`--steal-percent` (25 by default) caps how much of the pool a single block may
replace. This matters more than it sounds. Unbounded, a block whose note-ons
outnumber the pool replaces *every* voice, so no voice outlives the block it
was born in and a saturated passage renders as a stream of 85 ms attack
fragments instead of notes. Setting it to 100 restores that behaviour, which is
useful only for hearing the difference.

## How it works

One command buffer per audio block, five compute passes: **steal**, **spawn**,
**render**, **reduce**, **compact**. The host parses MIDI and resolves presets,
which is branchy table-lookup work that stays on the CPU permanently, and the
device does everything per-voice and per-sample.

A few decisions worth knowing about if you plan to read the code:

**Renders are byte-identical.** Two runs of the same file with the same
settings produce bitwise identical WAVs. Nothing in the render path uses a
floating-point atomic, because float atomics are non-deterministic in ordering.
The mixdown is a fixed-order tree reduction into per-workgroup partial buffers
that no other workgroup touches; voice slots are assigned by index rather than
from an atomic counter; the pool sort is a stable radix sort. The only atomics
in the codebase are integer histogram bins, where addition is exact and
order-independent.

**Phase is 32.32 fixed point**, not float. f32 loses fractional resolution past
2^24 samples, which is audible as detuning on long samples, and f64 costs real
throughput on consumer NVIDIA parts. WGSL has no portable u64, so the device
holds it as two u32 lanes.

**The voice pool is structure-of-arrays** inside a single storage buffer: field
`f` of voice `i` lives at `f * capacity + i`. It is re-sorted by sample region
every block during compaction, so neighbouring invocations hit the same cache
lines. This workload is memory-bandwidth bound -- roughly 20 flops per voice
per frame against one scattered global read -- so the access pattern is worth
far more than the arithmetic.

**Note-offs do not search the pool.** The pool is reordered every block, so a
voice cannot be found by index. Instead the lookup is inverted: the host
publishes a small table of how many note-offs each (channel, key) has seen, and
each voice compares its own ordinal against it.

## Known limitations

- **No effects.** No reverb, no chorus. The controllers that would need them
  (CC91 reverb, CC93 chorus, CC94 celeste, CC95 phaser) are recognised but do
  nothing. Run `kestrel info` on a MIDI first: it lists every controller the
  file uses, marks the unimplemented ones `MISSING`, and tells you what
  percentage of the file's controller events fall in that bucket.
- **Notes can keep sounding after the music stops, on some soundfonts.**
  Reported on a 44.7M-note file: from roughly two thirds of the way in, notes
  sound in a region with no note-ons behind them. It reproduces at every pool
  size tried and on one soundfont but not another, which is the clue --
  suspicion is that note-offs arriving while the sustain pedal (CC64) is down
  are deferred and never released if the pedal does not come back up. Those
  voices then sound until their sample runs out, so a library with long
  reverb-tail samples makes it obvious while a drier one hides it. Not yet
  root-caused; if you hear it, `--limiter off` and a shorter-sampled soundfont
  will tell you quickly whether you are looking at the same thing.
- **No realtime playback.** Offline rendering to WAV only.
- **No host integration.** There is no C ABI and no plugin build, so nothing
  else can currently drive this as a backend.
- **Pitch bend and channel volume/pan reach a voice up to 32 frames late**
  (0.67 ms) when the controller lands on the note's own tick. Ordinary
  controllers do not have this problem; these two were left on the older
  timing deliberately, because the consequence is bounded and the fix changes
  timing elsewhere.
- **SFZ `lorand`/`hirand` are ignored, so random region selection does not
  work.** *(Very likely fixed in the next release -- see below.)* A soundfont
  that uses random slices to choose between alternative regions gets **all of
  them at once** rather than one. Ten slices per velocity layer is a common
  arrangement, and that is then ten times the intended voice count on every
  note. Worse, when the alternatives differ only by a small sample offset --
  the usual trick for keeping repeated notes from phase-locking -- summing them
  comb-filters the tone rather than merely making it louder. Kestrel warns when
  it sees these opcodes, so you will not hit this silently. Until it is fixed,
  use a preset that does not use `lorand` (most packs ship one), or cap
  `--layers`.
- **SFZ support is otherwise partial.** The common opcodes work, including
  `#include`, velocity layers and `fil_veltrack`. Per-voice LFOs (`amplfo_*`,
  `fillfo_*`, `pitchlfo_*`) are recognised and reported, not applied -- though
  note that an LFO with a rate and no `*_depth` opcode would do nothing anyway,
  which is the common case in piano libraries.
- **NaN checking is off in release builds** unless you pass `--nan-guard`.

## Planned for the next release

**`lorand`/`hirand` support** is the priority and is the next thing being
worked on. It is one of the cheaper missing opcodes to add and it does not threaten
the determinism guarantee: the roll becomes a hash of the note id, so it varies
from note to note while two renders of the same file stay byte-identical.

Nothing else is committed. Effects, a host C ABI and realtime playback remain
out of scope for now.

## License

MIT -- see [LICENSE](LICENSE).

`src/limiter.rs` is a port of the realtime limiter OmniConverter ships, which
came originally from Kiva; the `--limiter omni` mode is named after it. The SF2
and SFZ loaders were written against the published format specifications.
Everything else was written for this project.
