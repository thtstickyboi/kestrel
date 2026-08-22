//! Thin CLI over the library. No GUI, on purpose.

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use kestrel::backend::Backend;
use kestrel::limiter::LimiterMode;
use kestrel::config::{
    AdmitRule, BackendKind, Config, EnvelopeCurve, Interpolation, StealRule,
};
use kestrel::{cpu::CpuSynth, driver::Driver, gpu, load_bank, testkit, wav};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "kestrel", version, about = "GPU-accelerated SoundFont/SFZ renderer for black MIDI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // RenderArgs is big and there is one of it
enum Cmd {
    /// Render a MIDI file to WAV.
    Render(RenderArgs),
    /// Print what the loader made of a soundfont or MIDI file.
    Info {
        path: PathBuf,
        /// Frames per render block, for the per-block density report. Match
        /// what you intend to render with.
        #[arg(long, default_value_t = 4096)]
        block: u32,
        #[arg(long, default_value_t = 48000)]
        rate: u32,
        /// Voices each note-on spawns, for the memory projection. One is right
        /// for most black MIDI soundfonts; a layered bank is higher and the
        /// figures scale with it.
        #[arg(long, default_value_t = 1)]
        layers: u32,
        /// The soundfont's release time, in seconds. Sets the window the voice
        /// estimate counts over; a voice outlives its note-off by roughly this
        /// long, and in black MIDI that is nearly its whole life.
        #[arg(long, default_value_t = 1.0)]
        release: f64,
    },
    /// Compare two WAV files and report the null-test difference.
    Null {
        a: PathBuf,
        b: PathBuf,
        /// Fail if the peak difference is above this many dB.
        #[arg(long, default_value_t = -80.0)]
        threshold: f64,
    },
    /// List the GPU adapters wgpu can see.
    GpuInfo,
    /// Write synthetic soundfonts and MIDI files, for benchmarking against
    /// material you can reproduce exactly.
    GenAssets {
        dir: PathBuf,
        /// Also write a soundfont with a sample pool of this many MiB, for
        /// benchmarking against a pool that does not fit in cache.
        #[arg(long = "big-mb")]
        big_mb: Option<usize>,
        /// Also write a MIDI with this many sustained notes, for benchmarking
        /// a pool that stays full.
        #[arg(long = "sustained")]
        sustained: Option<usize>,
    },
}

#[derive(Args)]
struct RenderArgs {
    /// Input MIDI file.
    midi: PathBuf,
    /// Soundfont, .sf2 or .sfz.
    #[arg(short = 's', long = "soundfont")]
    soundfont: PathBuf,
    /// Output WAV.
    #[arg(short = 'o', long = "out")]
    out: PathBuf,

    #[arg(long, default_value = "gpu", value_parser = ["cpu", "gpu"])]
    backend: String,
    #[arg(long, default_value_t = 48000)]
    rate: u32,
    #[arg(long, default_value_t = 4096)]
    block: u32,
    /// Frames per workgroup reduction round, and the note-off gate resolution.
    #[arg(long = "reduce-tile", default_value_t = 4)]
    reduce_tile: u32,
    /// Frames between note-off gate checks. A multiple of --reduce-tile.
    #[arg(long = "gate-frames", default_value_t = 32)]
    gate_frames: u32,
    /// Invocations per render workgroup.
    #[arg(long = "workgroup", default_value_t = 256)]
    workgroup: u32,
    /// Upper bound on render workgroups; sizes the partial buffer.
    #[arg(long = "render-workgroups", default_value_t = 2048)]
    render_workgroups: u32,
    #[arg(long = "max-voices", default_value_t = 1 << 20)]
    max_voices: u32,
    #[arg(long, default_value_t = 16)]
    layers: u32,
    #[arg(long, default_value = "linear")]
    interp: String,
    #[arg(long = "decay-curve", default_value = "exponential")]
    decay_curve: String,
    #[arg(long = "release-curve", default_value = "exponential")]
    release_curve: String,
    #[arg(long, default_value = "quietest")]
    steal: String,
    /// Which note-ons survive when one block has more of them than the pool
    /// has room for: loudest ranks by whether the note outlives the block and
    /// then by opening amplitude; even thins by position, ignoring both.
    #[arg(long = "admit", default_value = "loudest")]
    admit: String,
    /// Ceiling on how much of the voice pool one block may steal, in percent.
    /// 100 lets a saturated block replace the entire pool, which pumps.
    #[arg(long = "steal-percent", default_value_t = 25)]
    steal_percent: u32,
    #[arg(long, default_value = "float32", value_parser = ["float32", "pcm16"])]
    format: String,
    #[arg(long, default_value_t = 1.0)]
    volume: f32,
    /// Turn the soft limiter off.
    #[arg(long = "no-limiter")]
    no_limiter: bool,
    /// Which limiter: brickwall (lookahead true-peak, the default), off, or
    /// omni (the OmniConverter port, deprecated for rendering).
    #[arg(long, default_value = "brickwall", value_parser = ["brickwall", "omni", "off"])]
    limiter: String,
    /// Brickwall ceiling in dBFS. 0 is flat full scale.
    #[arg(long = "ceiling-db", default_value_t = 0.0)]
    ceiling_db: f64,
    /// Brickwall lookahead in ms. Also the render latency.
    #[arg(long = "lookahead-ms", default_value_t = 2.0)]
    lookahead_ms: f64,
    /// Brickwall release in ms.
    #[arg(long = "limiter-release-ms", default_value_t = 60.0)]
    limiter_release_ms: f64,
    /// Time constant of the brickwall's sustained stage, in ms. 0, the default,
    /// disables it: it measured worse on saturated black MIDI.
    #[arg(long = "limiter-sustain-ms", default_value_t = 0.0)]
    limiter_sustain_ms: f64,
    /// Detect only sample peaks, not inter-sample ones. Cheaper, lets
    /// inter-sample overshoot through.
    #[arg(long = "no-true-peak")]
    no_true_peak: bool,
    /// Turn the per-voice low-pass filter off.
    #[arg(long = "no-filter")]
    no_filter: bool,
    /// Do not re-sort the voice pool during compaction. Much slower at high
    /// voice counts; only useful for measuring what the sort buys.
    #[arg(long = "no-sort")]
    no_sort: bool,
    /// Keep the sample pool at its source rates instead of converting it.
    #[arg(long = "no-resample-pool")]
    no_resample_pool: bool,
    /// Sample pool budget in MiB before automatic downsampling kicks in.
    #[arg(long = "pool-budget", default_value_t = 2048)]
    pool_budget: u64,
    /// Log per-pass timings.
    #[arg(long)]
    profile: bool,
    /// Stop after this many seconds of output.
    #[arg(long)]
    seconds: Option<f64>,
    /// Force a wgpu backend: vulkan, dx12, metal, gl.
    #[arg(long = "gpu-backend")]
    gpu_backend: Option<String>,
    /// Substring match against the adapter name.
    #[arg(long = "gpu-adapter")]
    gpu_adapter: Option<String>,
    /// Check every block for NaN and Inf even in release builds.
    #[arg(long = "nan-guard")]
    nan_guard: bool,

    /// Write one CSV row per block: the admission decision and the level it
    /// produced. This is the diagnostic for block-rate pumping -- the audio is
    /// downstream of these numbers, so read them rather than the waveform.
    #[arg(long = "block-csv", value_name = "PATH")]
    block_csv: Option<PathBuf>,
    /// Compile shaders without automatic bounds clamps. Faster, and unsafe if
    /// anything upstream miscounts.
    #[arg(long = "unchecked-shaders")]
    unchecked_shaders: bool,
}

impl RenderArgs {
    fn to_config(&self) -> Result<(Config, BackendKind)> {
        let mut cfg = Config {
            sample_rate: self.rate,
            block_frames: self.block,
            reduce_tile: self.reduce_tile,
            gate_frames: self.gate_frames,
            workgroup_size: self.workgroup,
            max_render_workgroups: self.render_workgroups,
            max_voices: self.max_voices,
            max_layers: self.layers,
            max_steal_percent: self.steal_percent,
            master_volume: self.volume,
            limiter: !self.no_limiter,
            limiter_ceiling_db: self.ceiling_db,
            limiter_lookahead_ms: self.lookahead_ms,
            limiter_release_ms: self.limiter_release_ms,
            limiter_sustain_ms: self.limiter_sustain_ms,
            limiter_true_peak: !self.no_true_peak,
            filter_enabled: !self.no_filter,
            sort_voices: !self.no_sort,
            resample_pool: !self.no_resample_pool,
            sample_pool_budget: self.pool_budget << 20,
            profile: self.profile,
            gpu_backend: self.gpu_backend.clone(),
            gpu_adapter: self.gpu_adapter.clone(),
            ..Default::default()
        };
        if self.nan_guard {
            cfg.nan_guard = true;
        }
        cfg.unchecked_shaders = self.unchecked_shaders;
        cfg.interpolation = Interpolation::parse(&self.interp)
            .with_context(|| format!("unknown interpolation {:?}", self.interp))?;
        cfg.decay_curve = EnvelopeCurve::parse(&self.decay_curve)
            .with_context(|| format!("unknown decay curve {:?}", self.decay_curve))?;
        cfg.release_curve = EnvelopeCurve::parse(&self.release_curve)
            .with_context(|| format!("unknown release curve {:?}", self.release_curve))?;
        cfg.limiter_mode = LimiterMode::parse(&self.limiter)
            .with_context(|| format!("unknown limiter {:?}", self.limiter))?;
        if self.no_limiter {
            cfg.limiter_mode = LimiterMode::Off;
        }
        cfg.steal_rule = StealRule::parse(&self.steal)
            .with_context(|| format!("unknown steal rule {:?}", self.steal))?;
        cfg.admit_rule = AdmitRule::parse(&self.admit)
            .with_context(|| format!("unknown admit rule {:?}", self.admit))?;
        cfg.validate()?;
        let kind = if self.backend == "cpu" {
            BackendKind::Cpu
        } else {
            BackendKind::Gpu
        };
        Ok((cfg, kind))
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Render(args) => render(args),
        Cmd::Info {
            path,
            block,
            rate,
            layers,
            release,
        } => info(path, block, rate, layers, release),
        Cmd::Null { a, b, threshold } => null(a, b, threshold),
        Cmd::GpuInfo => gpu::print_adapters(),
        Cmd::GenAssets {
            dir,
            big_mb,
            sustained,
        } => gen_assets(dir, big_mb, sustained),
    }
}

fn render(args: RenderArgs) -> Result<()> {
    let (cfg, kind) = args.to_config()?;

    let t0 = Instant::now();
    let bank = Arc::new(load_bank(&args.soundfont, &cfg)?);
    log::info!("loaded {} in {:.2?}", bank.describe(), t0.elapsed());

    let mut driver = Driver::open(&cfg, bank.clone(), &args.midi)?;
    log::info!("{} has {} tracks", args.midi.display(), driver.track_count());

    let format = wav::SampleFormat::parse(&args.format).unwrap();
    let mut out = wav::WavWriter::create(&args.out, cfg.sample_rate, 2, format)?;
    let mut block = vec![0.0f32; cfg.block_samples()];

    let max_frames = args
        .seconds
        .map(|s| (s * cfg.sample_rate as f64) as u64)
        .unwrap_or(u64::MAX);

    let mut backend: Box<dyn Backend> = match kind {
        BackendKind::Cpu => Box::new(CpuSynth::new(&cfg, bank.clone())),
        BackendKind::Gpu => Box::new(gpu::GpuSynth::new(&cfg, bank.clone())?),
    };
    log::info!("rendering with the {} backend", backend.name());

    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut peak_voices = 0u64;

    let mut csv = match &args.block_csv {
        Some(p) => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(p)?);
            use std::io::Write;
            writeln!(f, "block,t,live,want,take,stolen,dropped,rms,peak,want_e,take_e")?;
            Some(f)
        }
        None => None,
    };

    loop {
        let more = driver.next_block(backend.as_mut(), &mut block)?;
        out.write_block(&block)?;

        let st = backend.stats();
        peak_voices = peak_voices.max(st.active_voices);

        if let Some(f) = csv.as_mut() {
            use std::io::Write;
            let n = block.len().max(1) as f64;
            let ss: f64 = block.iter().map(|v| *v as f64 * *v as f64).sum();
            let pk = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d = &driver.stats;
            writeln!(
                f,
                "{},{:.6},{},{},{},{},{},{:.9},{:.9},{},{}",
                d.blocks,
                driver.seconds_rendered(),
                d.last_live,
                d.last_want,
                d.last_take,
                d.last_stolen,
                d.dropped,
                (ss / n).sqrt(),
                pk,
                d.last_want_energy,
                d.last_take_energy
            )?;
        }

        if last_report.elapsed().as_secs_f64() > 1.0 {
            let secs = driver.seconds_rendered();
            let wall = start.elapsed().as_secs_f64();
            log::info!(
                "{:>8.2}s rendered | {:>10} voices | {:>12} notes | {:.2}x realtime",
                secs,
                st.active_voices,
                driver.stats.notes,
                secs / wall.max(1e-9)
            );
            if cfg.profile {
                let t = backend.timings();
                if !t.is_empty() {
                    let line: Vec<String> =
                        t.iter().map(|(n, ms)| format!("{n} {ms:.3}ms")).collect();
                    log::info!("  passes: {}", line.join("  "));
                }
            }
            last_report = Instant::now();
        }

        if !more || driver.stats.frames >= max_frames {
            break;
        }
    }

    let bytes = out.finish()?;
    let wall = start.elapsed().as_secs_f64();
    let secs = driver.seconds_rendered();
    let st = backend.stats();
    if driver.stats.variant_states > 1 || driver.stats.variant_fallbacks > 0 {
        log::info!(
            "sound controllers: {} distinct states, {} slots, {} rebuilds, {} approximated",
            driver.stats.variant_states,
            driver.stats.param_variants + 1,
            driver.stats.variant_rebuilds,
            driver.stats.variant_fallbacks
        );
    }
    log::info!(
        "wrote {} ({:.1} MiB, {:.2}s audio) in {:.2}s = {:.2}x realtime",
        args.out.display(),
        bytes as f64 / 1048576.0,
        secs,
        wall,
        secs / wall.max(1e-9)
    );
    log::info!(
        "{} notes, {} voices spawned, peak {} concurrent, {} stolen, {} dropped, peak level {:.3}",
        driver.stats.notes,
        driver.stats.voices_spawned,
        peak_voices,
        st.stolen,
        // From the driver: admission happens before a voice is built, so the
        // backend never sees a refused note-on to count.
        driver.stats.dropped,
        driver.stats.peak
    );
    if driver.stats.clipped > 0 {
        log::warn!(
            "{} samples were hard-clipped at full scale ({:.4}% of the render);              each one is a discontinuity the limiter let through.              --limiter brickwall cannot produce them.",
            driver.stats.clipped,
            100.0 * driver.stats.clipped as f64
                / (driver.stats.frames.max(1) * cfg.channels as u64) as f64
        );
    }
    Ok(())
}

fn fmt_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

fn info(path: PathBuf, block: u32, rate: u32, layers: u32, release: f64) -> Result<()> {
    let cfg = Config::default();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if ext == "mid" || ext == "midi" || ext == "rmi" {
        let mut s = kestrel::midi::MidiStream::open(&path)?;
        println!("format {} division {:?}", s.format, s.division);
        println!("{} tracks", s.track_count);
        let mut counts = [0u64; 8];
        let mut cc_counts = [0u64; 128];
        let mut last_tick = 0u64;

        // Per-block density. This is the number that decides host memory: a
        // block is admitted as a whole, so every note-on inside it is held
        // until the block ends, whether or not the pool can take it.
        let mut clock = kestrel::midi::TempoClock::new(s.division, rate);
        let mut cur_block = 0u64;
        let mut in_block = 0u64;
        // One note-on count per block, kept so a sliding window over it can
        // estimate concurrency afterwards.
        let mut per_block: Vec<u64> = Vec::new();

        while let Some((tick, ev)) = s.next() {
            last_tick = tick;
            if let kestrel::midi::Event::Tempo(us) = ev {
                clock.set_tempo(tick, us);
            }
            let b = clock.frame_at(tick) as u64 / block as u64;
            if b != cur_block {
                while per_block.len() as u64 <= cur_block {
                    per_block.push(0);
                }
                per_block[cur_block as usize] = in_block;
                cur_block = b;
                in_block = 0;
            }
            if matches!(ev, kestrel::midi::Event::NoteOn { .. }) {
                in_block += 1;
            }

            let i = match ev {
                kestrel::midi::Event::NoteOn { .. } => 0,
                kestrel::midi::Event::NoteOff { .. } => 1,
                kestrel::midi::Event::Cc { num, .. } => {
                    cc_counts[num as usize & 127] += 1;
                    2
                }
                kestrel::midi::Event::Program { .. } => 3,
                kestrel::midi::Event::PitchBend { .. } => 4,
                kestrel::midi::Event::Tempo(_) => 5,
                kestrel::midi::Event::Other => 6,
            };
            counts[i] += 1;
        }
        println!(
            "note-on {} note-off {} cc {} program {} bend {} tempo {} other {}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]
        );
        if counts[2] > 0 {
            use kestrel::driver::{cc_role, CcRole};
            println!("controllers used:");
            let mut missing = 0u64;
            for (num, n) in cc_counts.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                let (name, role) = cc_role(num as u8);
                let note = match role {
                    CcRole::Applied => "applied".to_string(),
                    CcRole::Inert(why) => format!("inert: {why}"),
                    CcRole::Missing(why) => {
                        missing += *n;
                        format!("MISSING: {why}")
                    }
                };
                println!("  cc{:<4}{:<20}{:>10}  {}", num, name, n, note);
            }
            println!(
                "  {} of {} controller events ({:.1}%) are unimplemented",
                missing,
                counts[2],
                missing as f64 * 100.0 / counts[2] as f64
            );
        }
        while per_block.len() as u64 <= cur_block {
            per_block.push(0);
        }
        per_block[cur_block as usize] = in_block;
        println!("last tick {last_tick}");

        let (peak_block, peak_at) = per_block
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i as u64))
            .max()
            .unwrap_or((0, 0));

        // Concurrency, as note-ons inside one release window.
        //
        // Counting notes between their note-on and note-off measures nothing
        // here: black MIDI notes are a tick long, so that peaks in the low
        // hundreds on a file that genuinely needs millions of voices. What
        // keeps a voice alive is its release tail, and since the notes are far
        // shorter than the release, "note-ons in the last `release` seconds" is
        // the estimate that tracks the real pool.
        let win = ((release * rate as f64) / block as f64).ceil().max(1.0) as usize;
        let mut sum: u64 = per_block.iter().take(win).sum();
        let mut peak_win = sum;
        let mut peak_win_at = 0usize;
        for i in win..per_block.len() {
            sum += per_block[i];
            sum -= per_block[i - win];
            if sum > peak_win {
                peak_win = sum;
                peak_win_at = i + 1 - win;
            }
        }

        // 24 B per candidate layer plus 12 B per note-on is what the driver
        // holds for a block while it waits to admit it. See ADMISSION.md. The
        // process peak runs above this -- the spawn list, the backend's copy
        // and the per-track read buffers are all on top, and on the reference
        // file that came to about 1.35x -- so treat it as a floor.
        let per_note = 24 * layers as u64 + 12;
        let bytes = peak_block * per_note;
        println!();
        println!("at --block {block}, --layers {layers}, {rate} Hz:");
        println!(
            "  busiest block  {:>14} note-ons, at {:.2}s",
            peak_block,
            peak_at as f64 * block as f64 / rate as f64
        );
        println!("  host memory    {:>14} for that block, at least", fmt_bytes(bytes));
        println!(
            "  peak {:.2}s span {:>14} voices, at {:.2}s -- roughly what a pool",
            release,
            peak_win * layers as u64,
            peak_win_at as f64 * block as f64 / rate as f64
        );
        println!("                                must hold for nothing to be stolen");

        // Two GiB for one block is where this stops being incidental.
        const COMFORTABLE: u64 = 2 << 30;
        if bytes > COMFORTABLE {
            let mut narrower = block;
            while narrower > 128 && peak_block * per_note / (block / narrower) as u64 > COMFORTABLE
            {
                narrower /= 2;
            }
            println!();
            println!("  That is a lot of host RAM for one block. Note-ons per block scale");
            println!("  with --block, so --block {narrower} would need about {},", 
                fmt_bytes(bytes / (block / narrower) as u64));
            println!("  and 128 is the floor. Lowering --max-voices does not help: the block");
            println!("  is held in full before admission thins it.");
        }
        return Ok(());
    }

    let bank = load_bank(&path, &cfg)?;
    println!("{}", bank.describe());
    for p in &bank.presets {
        println!(
            "  bank {:>3} program {:>3}  {:<24} {} regions",
            p.bank,
            p.program,
            p.name,
            p.regions.len()
        );
    }
    Ok(())
}

fn null(a: PathBuf, b: PathBuf, threshold: f64) -> Result<()> {
    let wa = wav::read(&a)?;
    let wb = wav::read(&b)?;
    if wa.channels != wb.channels {
        bail!("channel count differs: {} vs {}", wa.channels, wb.channels);
    }
    let n = wa.interleaved.len().min(wb.interleaved.len());
    if wa.interleaved.len() != wb.interleaved.len() {
        log::warn!(
            "lengths differ: {} vs {} samples, comparing the first {}",
            wa.interleaved.len(),
            wb.interleaved.len(),
            n
        );
    }

    let mut peak_diff = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut peak_ref = 0.0f64;
    let mut worst = 0usize;
    for i in 0..n {
        let d = (wa.interleaved[i] - wb.interleaved[i]).abs() as f64;
        if d > peak_diff {
            peak_diff = d;
            worst = i;
        }
        sum_sq += d * d;
        peak_ref = peak_ref.max(wa.interleaved[i].abs() as f64);
    }
    let rms = (sum_sq / n.max(1) as f64).sqrt();
    let db = |v: f64| if v <= 0.0 { -f64::INFINITY } else { 20.0 * v.log10() };

    let identical = wa.interleaved[..n] == wb.interleaved[..n];
    println!("samples compared : {n}");
    println!("bit-identical    : {identical}");
    println!("reference peak   : {:.6} ({:.2} dBFS)", peak_ref, db(peak_ref));
    println!("peak difference  : {:.9} ({:.2} dB)", peak_diff, db(peak_diff));
    println!("rms difference   : {:.9} ({:.2} dB)", rms, db(rms));
    println!("worst sample     : {worst}");

    if db(peak_diff) > threshold {
        bail!(
            "null test failed: peak difference {:.2} dB is above the {:.2} dB threshold",
            db(peak_diff),
            threshold
        );
    }
    println!("PASS (below {threshold:.1} dB)");
    Ok(())
}

fn gen_assets(dir: PathBuf, big_mb: Option<usize>, sustained: Option<usize>) -> Result<()> {
    std::fs::create_dir_all(&dir)?;
    testkit::simple_sf2(dir.join("simple.sf2"), 48000)?;
    testkit::rich_sf2(dir.join("rich.sf2"), 48000)?;
    testkit::single_note_midi(dir.join("single.mid"), 69, 127, 1.0)?;
    testkit::scatter_midi(dir.join("scatter1000.mid"), 1000, 10.0, 4, 36, 84)?;
    testkit::scatter_midi(dir.join("scatter100k.mid"), 100_000, 30.0, 16, 21, 108)?;
    testkit::simultaneous_midi(dir.join("stress2m.mid"), 2_000_000, 4.0)?;
    if let Some(mb) = big_mb {
        let p = dir.join(format!("big{mb}.sf2"));
        testkit::big_sf2(&p, 48000, mb)?;
        println!("wrote {} ({} MiB sample pool)", p.display(), mb);
    }
    if let Some(n) = sustained {
        let p = dir.join(format!("sustained{n}.mid"));
        testkit::sustained_midi(&p, n, 10.0)?;
        println!("wrote {} ({n} sustained notes)", p.display());
    }
    println!("wrote test assets to {}", dir.display());
    Ok(())
}
