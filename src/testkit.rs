//! Synthetic soundfonts and MIDI files for the test suite.
//!
//! The tests must not depend on a soundfont that happens to be installed, and
//! a hand-built SF2 doubles as a check on the loader: everything written here
//! is read back through the real parser.

use crate::midi::MidiWriter;
use anyhow::Result;
use std::f64::consts::PI;
use std::io::Write;
use std::path::Path;

// ---------------------------------------------------------------------------
// waveforms
// ---------------------------------------------------------------------------

/// One cycle repeated, so the loop points are exact and a looped render is a
/// pure tone with no discontinuity.
pub fn sine_cycles(freq: f64, rate: u32, cycles: usize, amp: f64) -> Vec<i16> {
    let per_cycle = (rate as f64 / freq).round() as usize;
    let n = per_cycle * cycles;
    (0..n)
        .map(|i| {
            let ph = i as f64 / per_cycle as f64 * 2.0 * PI;
            (ph.sin() * amp * 32767.0).round().clamp(-32767.0, 32767.0) as i16
        })
        .collect()
}

/// A ramp from -1 to 1 over `n` samples. Every sample is distinct, which makes
/// phase-accumulator bugs show up as an obviously wrong slope.
pub fn ramp(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1).max(1) as f64;
            ((t * 2.0 - 1.0) * 32767.0).round() as i16
        })
        .collect()
}

/// Deterministic pseudo-random noise. Not for listening, for catching indexing
/// bugs: neighbouring samples are uncorrelated, so an off-by-one is loud.
pub fn noise(n: usize, seed: u64) -> Vec<i16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 - 8192) as i16
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SF2 builder
// ---------------------------------------------------------------------------

pub struct TestSample {
    pub name: String,
    pub data: Vec<i16>,
    pub rate: u32,
    pub root: u8,
    pub correction: i8,
    pub loop_start: u32,
    pub loop_end: u32,
}

impl TestSample {
    pub fn new(name: &str, data: Vec<i16>, rate: u32, root: u8) -> Self {
        let n = data.len() as u32;
        TestSample {
            name: name.to_string(),
            data,
            rate,
            root,
            correction: 0,
            loop_start: 0,
            loop_end: n,
        }
    }
}

/// One instrument zone. `gens` are extra (operator, amount) pairs written
/// before the terminal sampleID generator.
pub struct TestZone {
    pub sample: u16,
    pub key: (u8, u8),
    pub vel: (u8, u8),
    pub gens: Vec<(u16, i16)>,
}

impl TestZone {
    pub fn new(sample: u16) -> Self {
        TestZone {
            sample,
            key: (0, 127),
            vel: (0, 127),
            gens: Vec::new(),
        }
    }
    pub fn keys(mut self, lo: u8, hi: u8) -> Self {
        self.key = (lo, hi);
        self
    }
    pub fn vels(mut self, lo: u8, hi: u8) -> Self {
        self.vel = (lo, hi);
        self
    }
    pub fn gen(mut self, op: u16, amount: i16) -> Self {
        self.gens.push((op, amount));
        self
    }
}

pub struct TestInstrument {
    pub name: String,
    pub zones: Vec<TestZone>,
}

pub struct TestPreset {
    pub name: String,
    pub bank: u16,
    pub program: u16,
    pub instrument: u16,
    pub gens: Vec<(u16, i16)>,
}

#[derive(Default)]
pub struct Sf2Builder {
    pub samples: Vec<TestSample>,
    pub instruments: Vec<TestInstrument>,
    pub presets: Vec<TestPreset>,
}

fn pad_name(name: &str) -> [u8; 20] {
    let mut b = [0u8; 20];
    let src = name.as_bytes();
    let n = src.len().min(19);
    b[..n].copy_from_slice(&src[..n]);
    b
}

/// An SF2 `ZSTR`: the string, then one or two zero bytes, whichever makes the
/// total length even.
///
/// The SoundFont spec asks for this explicitly, and it is not decoration. RIFF
/// pads an odd-sized chunk with a byte that is not counted in the chunk's size,
/// and BASSMIDI's reader advances by the size alone -- so an odd `INAM` puts it
/// one byte out for the rest of the file and it rejects the whole soundfont
/// with `BASS_ERROR_FILEFORM`. This crate's own reader never noticed, because
/// it seeks each chunk by name.
fn zstr(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(0);
    if v.len() % 2 == 1 {
        v.push(0);
    }
    v
}

fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(payload.len() + 9);
    v.extend_from_slice(id);
    v.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    v.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        v.push(0);
    }
    v
}

impl Sf2Builder {
    /// The gap the SF2 spec asks for between samples in `smpl`.
    const SAMPLE_GAP: usize = 46;

    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        // ---- sdta ----
        let mut smpl: Vec<i16> = Vec::new();
        let mut sample_ranges = Vec::new();
        for s in &self.samples {
            let start = smpl.len() as u32;
            smpl.extend_from_slice(&s.data);
            let end = smpl.len() as u32;
            smpl.extend(std::iter::repeat(0i16).take(Self::SAMPLE_GAP));
            sample_ranges.push((start, end));
        }
        let smpl_bytes: Vec<u8> = smpl.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut sdta = b"sdta".to_vec();
        sdta.extend_from_slice(&chunk(b"smpl", &smpl_bytes));

        // ---- pdta ----
        let mut phdr = Vec::new();
        let mut pbag = Vec::new();
        let mut pgen = Vec::new();
        let mut inst = Vec::new();
        let mut ibag = Vec::new();
        let mut igen = Vec::new();
        let mut shdr = Vec::new();

        for p in &self.presets {
            phdr.extend_from_slice(&pad_name(&p.name));
            phdr.extend_from_slice(&p.program.to_le_bytes());
            phdr.extend_from_slice(&p.bank.to_le_bytes());
            phdr.extend_from_slice(&((pbag.len() / 4) as u16).to_le_bytes());
            phdr.extend_from_slice(&0u32.to_le_bytes()); // library
            phdr.extend_from_slice(&0u32.to_le_bytes()); // genre
            phdr.extend_from_slice(&0u32.to_le_bytes()); // morphology

            pbag.extend_from_slice(&((pgen.len() / 4) as u16).to_le_bytes());
            pbag.extend_from_slice(&0u16.to_le_bytes()); // modNdx
            for (op, amt) in &p.gens {
                pgen.extend_from_slice(&op.to_le_bytes());
                pgen.extend_from_slice(&amt.to_le_bytes());
            }
            // instrument (41) must be the terminal generator of a preset zone
            pgen.extend_from_slice(&41u16.to_le_bytes());
            pgen.extend_from_slice(&p.instrument.to_le_bytes());
        }
        // EOP
        phdr.extend_from_slice(&pad_name("EOP"));
        phdr.extend_from_slice(&0u16.to_le_bytes());
        phdr.extend_from_slice(&0u16.to_le_bytes());
        phdr.extend_from_slice(&((pbag.len() / 4) as u16).to_le_bytes());
        phdr.extend_from_slice(&0u32.to_le_bytes());
        phdr.extend_from_slice(&0u32.to_le_bytes());
        phdr.extend_from_slice(&0u32.to_le_bytes());
        pbag.extend_from_slice(&((pgen.len() / 4) as u16).to_le_bytes());
        pbag.extend_from_slice(&0u16.to_le_bytes());
        pgen.extend_from_slice(&0u16.to_le_bytes());
        pgen.extend_from_slice(&0u16.to_le_bytes());

        for i in &self.instruments {
            inst.extend_from_slice(&pad_name(&i.name));
            inst.extend_from_slice(&((ibag.len() / 4) as u16).to_le_bytes());
            for z in &i.zones {
                ibag.extend_from_slice(&((igen.len() / 4) as u16).to_le_bytes());
                ibag.extend_from_slice(&0u16.to_le_bytes());

                let key = ((z.key.1 as u16) << 8) | z.key.0 as u16;
                igen.extend_from_slice(&43u16.to_le_bytes());
                igen.extend_from_slice(&key.to_le_bytes());
                let vel = ((z.vel.1 as u16) << 8) | z.vel.0 as u16;
                igen.extend_from_slice(&44u16.to_le_bytes());
                igen.extend_from_slice(&vel.to_le_bytes());
                for (op, amt) in &z.gens {
                    igen.extend_from_slice(&op.to_le_bytes());
                    igen.extend_from_slice(&amt.to_le_bytes());
                }
                // sampleID (53) terminates an instrument zone
                igen.extend_from_slice(&53u16.to_le_bytes());
                igen.extend_from_slice(&z.sample.to_le_bytes());
            }
        }
        // EOI
        inst.extend_from_slice(&pad_name("EOI"));
        inst.extend_from_slice(&((ibag.len() / 4) as u16).to_le_bytes());
        ibag.extend_from_slice(&((igen.len() / 4) as u16).to_le_bytes());
        ibag.extend_from_slice(&0u16.to_le_bytes());
        igen.extend_from_slice(&0u16.to_le_bytes());
        igen.extend_from_slice(&0u16.to_le_bytes());

        for (i, s) in self.samples.iter().enumerate() {
            let (start, end) = sample_ranges[i];
            shdr.extend_from_slice(&pad_name(&s.name));
            shdr.extend_from_slice(&start.to_le_bytes());
            shdr.extend_from_slice(&end.to_le_bytes());
            shdr.extend_from_slice(&(start + s.loop_start).to_le_bytes());
            shdr.extend_from_slice(&(start + s.loop_end).to_le_bytes());
            shdr.extend_from_slice(&s.rate.to_le_bytes());
            shdr.push(s.root);
            shdr.push(s.correction as u8);
            shdr.extend_from_slice(&0u16.to_le_bytes()); // sampleLink
            shdr.extend_from_slice(&1u16.to_le_bytes()); // monoSample
        }
        // EOS. A shdr record is 46 bytes: 20 of name, then start, end,
        // startloop, endloop and rate as u32, originalKey and correction as
        // bytes, sampleLink and sampleType as u16. The terminal record has to
        // be the same size as a real one -- this wrote 48 for a while, which
        // left the chunk size not a whole number of records, and BASSMIDI
        // rejects such a file outright with BASS_ERROR_FILEFORM even though
        // this crate's own loader divides through and carries on.
        shdr.extend_from_slice(&pad_name("EOS"));
        shdr.extend_from_slice(&[0u8; 22]);
        shdr.extend_from_slice(&0u16.to_le_bytes());
        shdr.extend_from_slice(&0u16.to_le_bytes());

        let mut pdta = b"pdta".to_vec();
        pdta.extend_from_slice(&chunk(b"phdr", &phdr));
        pdta.extend_from_slice(&chunk(b"pbag", &pbag));
        pdta.extend_from_slice(&chunk(b"pmod", &[0u8; 10]));
        pdta.extend_from_slice(&chunk(b"pgen", &pgen));
        pdta.extend_from_slice(&chunk(b"inst", &inst));
        pdta.extend_from_slice(&chunk(b"ibag", &ibag));
        pdta.extend_from_slice(&chunk(b"imod", &[0u8; 10]));
        pdta.extend_from_slice(&chunk(b"igen", &igen));
        pdta.extend_from_slice(&chunk(b"shdr", &shdr));

        // ---- INFO ----
        let mut info = b"INFO".to_vec();
        info.extend_from_slice(&chunk(b"ifil", &[2u8, 0, 1, 0]));
        info.extend_from_slice(&chunk(b"isng", &zstr("EMU8000")));
        info.extend_from_slice(&chunk(b"INAM", &zstr("kestrel test")));

        let mut body = b"sfbk".to_vec();
        body.extend_from_slice(&chunk(b"LIST", &info));
        body.extend_from_slice(&chunk(b"LIST", &sdta));
        body.extend_from_slice(&chunk(b"LIST", &pdta));

        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"RIFF")?;
        f.write_all(&(body.len() as u32).to_le_bytes())?;
        f.write_all(&body)?;
        f.flush()?;
        Ok(())
    }
}

/// The default test soundfont: one looped sine, no envelope shaping, so a
/// single note renders to exactly the sample repeated.
pub fn simple_sf2(path: impl AsRef<Path>, rate: u32) -> Result<()> {
    let data = sine_cycles(440.0, rate, 20, 0.5);
    let n = data.len() as u32;
    let mut s = TestSample::new("sine440", data, rate, 69);
    s.loop_start = 0;
    s.loop_end = n;

    let b = Sf2Builder {
        samples: vec![s],
        instruments: vec![TestInstrument {
            name: "sine".into(),
            zones: vec![TestZone::new(0)
                .gen(54, 1) // sampleModes: loop continuously
                .gen(34, -12000) // attackVolEnv: instant
                .gen(36, -12000) // decayVolEnv: instant
                .gen(37, 0) // sustainVolEnv: full level
                .gen(38, -12000)], // releaseVolEnv: instant
        }],
        presets: vec![TestPreset {
            name: "sine".into(),
            bank: 0,
            program: 0,
            instrument: 0,
            gens: Vec::new(),
        }],
    };
    b.write(path)
}

/// A soundfont with several regions, key splits, panning, an envelope and a
/// filter. Exercises the parts of the loader a real soundfont uses.
pub fn rich_sf2(path: impl AsRef<Path>, rate: u32) -> Result<()> {
    let low = sine_cycles(220.0, rate, 40, 0.6);
    let mid = ramp(2048);
    let high = noise(4096, 0x9E3779B97F4A7C15);

    let mut s0 = TestSample::new("low", low, rate, 57);
    s0.loop_start = 0;
    s0.loop_end = 4096;
    let s1 = TestSample::new("ramp", mid, rate, 69);
    let s2 = TestSample::new("noise", high, 22050, 81);

    let b = Sf2Builder {
        samples: vec![s0, s1, s2],
        instruments: vec![TestInstrument {
            name: "split".into(),
            zones: vec![
                TestZone::new(0)
                    .keys(0, 59)
                    .gen(54, 1)
                    .gen(34, -3000) // slow-ish attack
                    .gen(36, 1200) // 2 s decay
                    .gen(37, 200) // sustain at -20 dB
                    .gen(38, 0) // 1 s release
                    .gen(17, -400), // pan left
                TestZone::new(1)
                    .keys(60, 71)
                    .gen(34, -12000)
                    .gen(36, -12000)
                    .gen(37, 0)
                    .gen(38, -1200)
                    .gen(8, 8000), // filter at ~800 Hz
                TestZone::new(2)
                    .keys(72, 127)
                    .vels(0, 127)
                    .gen(34, -12000)
                    .gen(36, 600)
                    .gen(37, 100)
                    .gen(38, -600)
                    .gen(17, 400)
                    .gen(48, 100), // -10 dB attenuation
            ],
        }],
        presets: vec![
            TestPreset {
                name: "split".into(),
                bank: 0,
                program: 0,
                instrument: 0,
                gens: Vec::new(),
            },
            TestPreset {
                name: "split quiet".into(),
                bank: 0,
                program: 1,
                instrument: 0,
                gens: vec![(48, 200)], // preset-level attenuation offset
            },
        ],
    };
    b.write(path)
}

/// A soundfont whose sample pool is too big to sit in cache, one looped sample
/// per key. This is what a real piano library looks like to the memory system,
/// and it is the only way to see whether the region sort is doing anything:
/// with a pool that fits in L2, every access pattern looks equally good.
pub fn big_sf2(path: impl AsRef<Path>, rate: u32, target_mb: usize) -> Result<()> {
    const KEYS: usize = 88;
    let frames_per_sample = (target_mb * 1024 * 1024 / 2 / KEYS).max(1024);

    let mut samples = Vec::with_capacity(KEYS);
    let mut zones = Vec::with_capacity(KEYS);
    for k in 0..KEYS {
        let key = 21 + k as u8;
        // A detuned pair beating slowly, so neighbouring samples are not
        // identical and the compiler cannot fold them away.
        let f0 = 440.0 * 2f64.powf((key as f64 - 69.0) / 12.0);
        let data: Vec<i16> = (0..frames_per_sample)
            .map(|i| {
                let t = i as f64 / rate as f64;
                let v = (2.0 * PI * f0 * t).sin() * 0.5
                    + (2.0 * PI * f0 * 1.003 * t).sin() * 0.3
                    + (2.0 * PI * f0 * 2.0 * t).sin() * 0.15;
                (v * 20000.0).clamp(-32767.0, 32767.0) as i16
            })
            .collect();
        let n = data.len() as u32;
        let mut s = TestSample::new(&format!("k{key}"), data, rate, key);
        s.loop_start = n / 2;
        s.loop_end = n;
        samples.push(s);
        zones.push(
            TestZone::new(k as u16)
                .keys(key, key)
                .gen(54, 1) // loop continuously
                .gen(34, -8000) // ~4 ms attack
                .gen(36, 1200) // 2 s decay
                .gen(37, 100) // sustain at -10 dB
                .gen(38, -1200), // 0.5 s release
        );
    }

    Sf2Builder {
        samples,
        instruments: vec![TestInstrument {
            name: "big".into(),
            zones,
        }],
        presets: vec![TestPreset {
            name: "big".into(),
            bank: 0,
            program: 0,
            instrument: 0,
            gens: Vec::new(),
        }],
    }
    .write(path)
}

// ---------------------------------------------------------------------------
// MIDI generators
// ---------------------------------------------------------------------------

/// `notes` note-ons spread over `seconds`, cycling through keys and channels.
pub fn scatter_midi(
    path: impl AsRef<Path>,
    notes: usize,
    seconds: f64,
    tracks: usize,
    key_lo: u8,
    key_hi: u8,
) -> Result<()> {
    const PPQ: u16 = 960;
    let us_per_qn = 500_000u32;
    let ticks_per_second = PPQ as f64 * 1e6 / us_per_qn as f64;
    let total_ticks = (seconds * ticks_per_second) as u64;

    let mut w = MidiWriter::new(PPQ);
    w.tempo_track(us_per_qn);

    let span = (key_hi - key_lo + 1) as usize;
    let per_track = notes.div_ceil(tracks.max(1));
    let mut emitted = 0usize;

    for t in 0..tracks.max(1) {
        let mut evs: Vec<(u64, [u8; 3], usize)> = Vec::new();
        for _ in 0..per_track {
            if emitted >= notes {
                break;
            }
            let n = emitted;
            emitted += 1;
            let on = total_ticks * n as u64 / notes.max(1) as u64;
            // Vary the length so releases overlap note-ons.
            let len = (ticks_per_second * 0.1) as u64 + (n as u64 % 7) * 40;
            let key = key_lo + (n % span) as u8;
            let ch = (t % 16) as u8;
            let vel = 40 + (n % 80) as u8;
            evs.push((on, [0x90 | ch, key, vel], 3));
            evs.push((on + len, [0x80 | ch, key, 0], 3));
        }
        if !evs.is_empty() {
            w.track(evs);
        }
    }
    w.save(path)
}

/// Every voice starts at the same instant. Used for the stress test.
pub fn simultaneous_midi(path: impl AsRef<Path>, notes: usize, hold_seconds: f64) -> Result<()> {
    const PPQ: u16 = 960;
    let us_per_qn = 500_000u32;
    let ticks_per_second = PPQ as f64 * 1e6 / us_per_qn as f64;
    let hold = (hold_seconds * ticks_per_second) as u64;

    let mut w = MidiWriter::new(PPQ);
    w.tempo_track(us_per_qn);

    // 16 channels x 128 keys is only 2048 distinct notes, so the rest are
    // retriggers of the same keys, which is exactly what a black MIDI does.
    let per_track = 65536usize;
    let mut left = notes;
    let mut n = 0usize;
    while left > 0 {
        let count = left.min(per_track);
        let mut evs = Vec::with_capacity(count * 2);
        for _ in 0..count {
            let key = 21 + (n % 88) as u8;
            let ch = (n % 16) as u8;
            evs.push((1u64, [0x90 | ch, key, 100u8], 3));
            evs.push((1 + hold, [0x80 | ch, key, 0], 3));
            n += 1;
        }
        w.track(evs);
        left -= count;
    }
    w.save(path)
}

/// `notes` note-ons spread over `seconds`, none of them released until the
/// very end. The voice count only ever climbs, which is what forces the pool
/// to overflow and the stealing rule to be exercised.
pub fn sustained_midi(path: impl AsRef<Path>, notes: usize, seconds: f64) -> Result<()> {
    const PPQ: u16 = 960;
    let us_per_qn = 500_000u32;
    let ticks_per_second = PPQ as f64 * 1e6 / us_per_qn as f64;
    let total = (seconds * ticks_per_second) as u64;

    let mut w = MidiWriter::new(PPQ);
    w.tempo_track(us_per_qn);

    let per_track = 100_000usize;
    let mut n = 0usize;
    while n < notes {
        let count = per_track.min(notes - n);
        let mut evs = Vec::with_capacity(count * 2);
        for _ in 0..count {
            let on = total * n as u64 / notes.max(1) as u64;
            let key = 21 + (n % 88) as u8;
            let ch = (n % 16) as u8;
            evs.push((on, [0x90 | ch, key, 90u8], 3));
            evs.push((total + 1, [0x80 | ch, key, 0], 3));
            n += 1;
        }
        w.track(evs);
    }
    w.save(path)
}

/// One note, one key, held for a known length. The single-voice test.
/// A MIDI built from raw `(seconds, message, length)` triples at a fixed
/// tempo. Controller tests care about the exact ordering of events around a
/// note, which the shaped helpers above deliberately hide.
pub fn event_midi(path: impl AsRef<Path>, events: &[(f64, [u8; 3], usize)]) -> Result<()> {
    const PPQ: u16 = 960;
    let us_per_qn = 500_000u32;
    let ticks_per_second = PPQ as f64 * 1e6 / us_per_qn as f64;
    let mut w = MidiWriter::new(PPQ);
    w.tempo_track(us_per_qn);
    w.track(
        events
            .iter()
            .map(|(t, m, l)| ((t * ticks_per_second) as u64, *m, *l))
            .collect(),
    );
    w.save(path)
}

/// The three messages that set a channel's bend range through RPN 0. Without
/// them the range is the GM default of two semitones, which cannot express an
/// exact interval: the largest bend is 8191/8192 of the range, not all of it.
pub fn set_bend_range(ch: u8, semitones: u8) -> [(f64, [u8; 3], usize); 3] {
    [
        (0.0, [0xB0 | ch, 101, 0], 3),
        (0.0, [0xB0 | ch, 100, 0], 3),
        (0.0, [0xB0 | ch, 6, semitones], 3),
    ]
}

/// A pitch bend message. `value` is the raw 14-bit quantity, 8192 centred.
pub fn bend_msg(ch: u8, value: u16) -> [u8; 3] {
    [0xE0 | ch, (value & 0x7F) as u8, (value >> 7) as u8]
}

pub fn single_note_midi(path: impl AsRef<Path>, key: u8, vel: u8, seconds: f64) -> Result<()> {
    const PPQ: u16 = 960;
    let us_per_qn = 500_000u32;
    let ticks_per_second = PPQ as f64 * 1e6 / us_per_qn as f64;
    let mut w = MidiWriter::new(PPQ);
    w.tempo_track(us_per_qn);
    w.track(vec![
        (0, [0x90, key, vel], 3),
        ((seconds * ticks_per_second) as u64, [0x80, key, 0], 3),
    ]);
    w.save(path)
}
