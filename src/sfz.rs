//! SFZ loader.
//!
//! Covers the opcode subset a sampled-instrument library actually uses:
//! region/group/global/control headers, key and velocity ranges, tuning,
//! volume and pan, loop modes, the amplitude envelope, and the low-pass
//! filter. Unknown opcodes are counted and reported once rather than
//! silently ignored, so a library that leans on something unimplemented is
//! visible instead of quietly wrong.
//!
//! Stereo sample files become two mono regions panned hard left and right,
//! because the engine's voice is mono by construction.

use crate::bank::*;
use crate::config::Config;
use crate::resample;
use crate::wav;
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Ignore the note-off gate entirely (SFZ `loop_mode=one_shot`).
pub const VF_ONE_SHOT: u32 = 1 << 2;

#[derive(Clone, Default)]
struct OpcodeSet(BTreeMap<String, String>);

impl OpcodeSet {
    fn merged(&self, other: &OpcodeSet) -> OpcodeSet {
        let mut m = self.0.clone();
        for (k, v) in &other.0 {
            m.insert(k.clone(), v.clone());
        }
        OpcodeSet(m)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(|s| s.as_str())
    }
    fn f32(&self, k: &str) -> Option<f32> {
        self.get(k).and_then(|v| v.trim().parse::<f32>().ok())
    }
    fn i32(&self, k: &str) -> Option<i32> {
        self.get(k)
            .and_then(|v| v.trim().parse::<i32>().ok().or_else(|| v.trim().parse::<f32>().ok().map(|f| f as i32)))
    }
    /// Note names as well as numbers: `c4`, `a#3`, `60`.
    fn key(&self, k: &str) -> Option<i32> {
        let v = self.get(k)?.trim();
        if let Ok(n) = v.parse::<i32>() {
            return Some(n);
        }
        parse_note_name(v)
    }
}

fn parse_note_name(s: &str) -> Option<i32> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    let step = match b[0].to_ascii_lowercase() {
        b'c' => 0,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };
    let mut i = 1;
    let mut acc = 0i32;
    while i < b.len() && (b[i] == b'#' || b[i] == b'b' || b[i] == b'-' && i == 1) {
        match b[i] {
            b'#' => acc += 1,
            b'b' => acc -= 1,
            _ => break,
        }
        i += 1;
    }
    let octave: i32 = s[i..].trim().parse().ok()?;
    // SFZ follows the convention where middle C (60) is c4.
    Some((octave + 1) * 12 + step + acc)
}

struct Parser {
    root: PathBuf,
    default_path: PathBuf,
    unknown: HashMap<String, u32>,
    depth: u32,
}

impl Parser {
    fn resolve(&self, rel: &str) -> PathBuf {
        // SFZ paths use backslashes on Windows-authored libraries.
        let rel = rel.replace('\\', "/");
        let joined = self.default_path.join(&rel);
        let cand = self.root.join(&joined);
        if cand.exists() {
            return cand;
        }
        // Case-insensitive fallback, which matters when a Windows-authored
        // library is rendered on a case-sensitive filesystem.
        let parent = cand.parent().unwrap_or(&self.root).to_path_buf();
        let name = cand.file_name().map(|n| n.to_string_lossy().to_lowercase());
        if let (Some(name), Ok(rd)) = (name, std::fs::read_dir(&parent)) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().to_lowercase() == name {
                    return e.path();
                }
            }
        }
        cand
    }

    fn parse_file(&mut self, path: &Path, out: &mut Vec<(String, OpcodeSet)>) -> Result<()> {
        if self.depth > 16 {
            bail!("#include nesting is too deep at {}", path.display());
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;

        let mut current_header = String::from("global");
        let mut current = OpcodeSet::default();
        let mut started = false;

        for raw_line in text.lines() {
            let line = strip_comment(raw_line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("#include") {
                let inc = rest.trim().trim_matches('"').trim();
                let p = self.resolve(inc);
                if started {
                    out.push((current_header.clone(), std::mem::take(&mut current)));
                    started = false;
                }
                self.depth += 1;
                let r = self.parse_file(&p, out);
                self.depth -= 1;
                if let Err(e) = r {
                    log::warn!("{}", e);
                }
                continue;
            }
            if line.starts_with("#define") {
                continue;
            }

            // A line can hold several headers and opcodes.
            let mut rest = line;
            while !rest.is_empty() {
                rest = rest.trim_start();
                if let Some(stripped) = rest.strip_prefix('<') {
                    let Some(end) = stripped.find('>') else { break };
                    if started {
                        out.push((current_header.clone(), std::mem::take(&mut current)));
                    }
                    current_header = stripped[..end].trim().to_ascii_lowercase();
                    current = OpcodeSet::default();
                    started = true;
                    rest = &stripped[end + 1..];
                    continue;
                }
                let Some(eq) = rest.find('=') else { break };
                let key = rest[..eq].trim().to_ascii_lowercase();
                let after = &rest[eq + 1..];
                // A value runs to the next `opcode=` on the line, so file
                // names with spaces survive.
                let value_end = find_value_end(after);
                let value = after[..value_end].trim().to_string();
                rest = &after[value_end..];
                if key.is_empty() {
                    continue;
                }
                current.0.insert(key, value);
                started = true;
            }
        }
        if started {
            out.push((current_header, current));
        }
        Ok(())
    }
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// Find where a value ends: just before the last whitespace-separated token
/// that itself contains an `=`.
fn find_value_end(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut last_ws = None;
    while i < bytes.len() {
        if bytes[i] == b'=' {
            return match last_ws {
                Some(w) => w,
                None => i,
            };
        }
        if bytes[i] == b'<' {
            return last_ws.unwrap_or(i);
        }
        if bytes[i].is_ascii_whitespace() {
            // Remember the start of the token after this whitespace run.
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            last_ws = Some(i);
            i = j;
            continue;
        }
        i += 1;
    }
    s.len()
}

pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Bank> {
    let path = path.as_ref();
    let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();

    let mut parser = Parser {
        root: root.clone(),
        default_path: PathBuf::new(),
        unknown: HashMap::new(),
        depth: 0,
    };

    // First pass just to pick up <control> default_path, which can appear
    // before any region but affects every sample path.
    let mut sections: Vec<(String, OpcodeSet)> = Vec::new();
    parser.parse_file(path, &mut sections)?;
    for (h, ops) in &sections {
        if h == "control" {
            if let Some(dp) = ops.get("default_path") {
                parser.default_path = PathBuf::from(dp.replace('\\', "/"));
            }
        }
    }

    let mut global = OpcodeSet::default();
    let mut group = OpcodeSet::default();
    let mut region_sets: Vec<OpcodeSet> = Vec::new();

    for (h, ops) in &sections {
        match h.as_str() {
            "global" => {
                global = ops.clone();
                group = OpcodeSet::default();
            }
            "master" => {
                group = OpcodeSet::default();
                global = global.merged(ops);
            }
            "group" => group = ops.clone(),
            "region" => region_sets.push(global.merged(&group).merged(ops)),
            "control" | "curve" | "effect" => {}
            other => {
                *parser.unknown.entry(format!("<{other}>")).or_default() += 1;
            }
        }
    }

    if region_sets.is_empty() {
        bail!("{}: no <region> sections", path.display());
    }

    // ---- load every referenced wav once -----------------------------------
    let mut pool: Vec<i16> = Vec::new();
    let mut samples: Vec<SampleInfo> = Vec::new();
    // (path, channel) -> sample index
    let mut sample_cache: HashMap<(PathBuf, usize), u32> = HashMap::new();
    let pool_rate = if cfg.resample_pool { cfg.sample_rate } else { 0 };

    let mut regions: Vec<Region> = Vec::new();
    let mut region_ids: Vec<u32> = Vec::new();

    for ops in &region_sets {
        let Some(sample_rel) = ops.get("sample") else {
            continue;
        };
        if ops.i32("end") == Some(-1) {
            continue; // conventional way to disable a region
        }
        let spath = parser.resolve(sample_rel);
        let channels = match load_sample_channels(
            &spath,
            &mut pool,
            &mut samples,
            &mut sample_cache,
            pool_rate,
        ) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("{e:#}");
                continue;
            }
        };

        for (ch, sidx) in channels.iter().enumerate() {
            note_unhandled(ops, &mut parser.unknown);
            let mut r = region_from_opcodes(ops, *sidx, &samples[*sidx as usize]);
            if channels.len() == 2 {
                // Hard pan the two halves of a stereo file, then let the
                // region's own pan bias the pair.
                r.pan = if ch == 0 { -1.0 } else { 1.0 };
            }
            region_ids.push(regions.len() as u32);
            regions.push(r);
        }
    }

    if regions.is_empty() {
        bail!("{}: every region was unusable", path.display());
    }

    for (op, n) in &parser.unknown {
        log::warn!("sfz: ignored unsupported {op} ({n} times)");
    }

    let preset = Preset {
        bank: 0,
        program: 0,
        name: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        regions: region_ids,
        key_index: Vec::new(),
        key_regions: Vec::new(),
    };

    let mut bank = Bank {
        pool,
        pool_rate,
        samples,
        regions,
        params: Vec::new(),
        presets: vec![preset],
        index: Vec::new(),
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };
    bank.build_params(cfg);
    bank.finish();
    Ok(bank)
}

const POOL_GUARD: usize = 8;

fn load_sample_channels(
    spath: &Path,
    pool: &mut Vec<i16>,
    samples: &mut Vec<SampleInfo>,
    cache: &mut HashMap<(PathBuf, usize), u32>,
    pool_rate: u32,
) -> Result<Vec<u32>> {
    if let Some(&first) = cache.get(&(spath.to_path_buf(), 0)) {
        let mut out = vec![first];
        if let Some(&second) = cache.get(&(spath.to_path_buf(), 1)) {
            out.push(second);
        }
        return Ok(out);
    }

    let w = wav::read(spath)?;
    let nch = (w.channels as usize).clamp(1, 2);
    let mut out = Vec::with_capacity(nch);

    for ch in 0..nch {
        let raw = w.channel_i16(ch);
        let src_rate = w.sample_rate.max(1);
        let ratio = if pool_rate != 0 && pool_rate != src_rate {
            pool_rate as f64 / src_rate as f64
        } else {
            1.0
        };
        let data = if (ratio - 1.0).abs() > 1e-12 {
            resample::resample_i16(&raw, ratio)
        } else {
            raw
        };

        let len = data.len() as u32;
        let (mut ls, mut le) = w.loop_points.unwrap_or((0, len.saturating_sub(1)));
        ls = (ls as f64 * ratio).round() as u32;
        le = (le as f64 * ratio).round() as u32;

        let start = pool.len() as u32;
        pool.extend_from_slice(&data);
        pool.extend(std::iter::repeat(0i16).take(POOL_GUARD));

        let idx = samples.len() as u32;
        samples.push(SampleInfo {
            start,
            len,
            loop_start: ls.min(len.saturating_sub(1)),
            loop_end: le.min(len),
            rate: if pool_rate != 0 { pool_rate } else { src_rate },
            root_key: w.root_key.unwrap_or(60),
            correction_cents: w.fine_tune_cents,
            resample_ratio: ratio as f32,
            name: spath.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        });
        cache.insert((spath.to_path_buf(), ch), idx);
        out.push(idx);
    }
    Ok(out)
}

/// Every opcode this loader reads. Anything outside it is dropped, and being
/// dropped silently is how a soundfont ends up sounding wrong for a session:
/// the EastWest Steinway port carries `fil_veltrack=9600` against an 89 Hz
/// cutoff, and without it every note plays under an 89 Hz lowpass. So the set
/// is written down and what falls outside it is counted and reported.
const KNOWN_OPCODES: &[&str] = &[
    "sample",
    "lokey",
    "hikey",
    "key",
    "lovel",
    "hivel",
    "pitch_keycenter",
    "tune",
    "pitch",
    "transpose",
    "volume",
    "pan",
    "loop_mode",
    "loopmode",
    "loop_start",
    "loop_end",
    "offset",
    "end",
    "ampeg_delay",
    "ampeg_attack",
    "ampeg_hold",
    "ampeg_decay",
    "ampeg_sustain",
    "ampeg_release",
    "cutoff",
    "resonance",
    "fil_veltrack",
    "fil_type",
    "group",
    "off_by",
    "default_path",
];

/// Opcodes that are recognised as deliberately unimplemented, so they are
/// reported once as a group rather than as unknown noise. These are real
/// features this synth does not have yet, not typos.
const UNIMPLEMENTED_PREFIXES: &[&str] = &["amplfo_", "fillfo_", "pitchlfo_", "set_cc", "label_cc"];

fn note_unhandled(ops: &OpcodeSet, unknown: &mut HashMap<String, u32>) {
    for k in ops.0.keys() {
        if KNOWN_OPCODES.contains(&k.as_str()) {
            continue;
        }
        let key = match UNIMPLEMENTED_PREFIXES.iter().find(|p| k.starts_with(**p)) {
            Some(p) => format!("{p}* (not implemented)"),
            None => k.clone(),
        };
        *unknown.entry(key).or_default() += 1;
    }
}

fn region_from_opcodes(ops: &OpcodeSet, sample: u32, info: &SampleInfo) -> Region {
    let mut r = Region {
        sample,
        ..Default::default()
    };

    if let Some(k) = ops.key("key") {
        let k = k.clamp(0, 127) as u8;
        r.key_lo = k;
        r.key_hi = k;
        r.root_key_override = k as i16;
    }
    if let Some(k) = ops.key("lokey") {
        r.key_lo = k.clamp(0, 127) as u8;
    }
    if let Some(k) = ops.key("hikey") {
        r.key_hi = k.clamp(0, 127) as u8;
    }
    if let Some(k) = ops.key("pitch_keycenter") {
        r.root_key_override = k.clamp(0, 127) as i16;
    }
    if r.root_key_override < 0 {
        r.root_key_override = info.root_key as i16;
    }
    if let Some(v) = ops.i32("lovel") {
        r.vel_lo = v.clamp(0, 127) as u8;
    }
    if let Some(v) = ops.i32("hivel") {
        r.vel_hi = v.clamp(0, 127) as u8;
    }

    let tune = ops.f32("tune").or_else(|| ops.f32("pitch")).unwrap_or(0.0);
    r.fine_tune = tune.round().clamp(-32768.0, 32767.0) as i16;
    if let Some(t) = ops.i32("transpose") {
        r.coarse_tune = t.clamp(-127, 127) as i16;
    }
    if let Some(s) = ops.f32("pitch_keytrack") {
        r.scale_tuning = s.round().clamp(-1200.0, 1200.0) as i16;
    }

    if let Some(v) = ops.f32("volume") {
        r.attenuation_cb = -v * 10.0;
    }
    if let Some(p) = ops.f32("pan") {
        r.pan = (p / 100.0).clamp(-1.0, 1.0);
    }

    r.loop_mode = match ops.get("loop_mode").unwrap_or("") {
        "loop_continuous" => LoopMode::Continuous,
        "loop_sustain" => LoopMode::UntilRelease,
        "one_shot" => LoopMode::NoLoop,
        "no_loop" => LoopMode::NoLoop,
        _ => {
            // Default follows the sample: if the wav declares a loop, use it.
            if info.loop_end > info.loop_start + 1 {
                LoopMode::Continuous
            } else {
                LoopMode::NoLoop
            }
        }
    };

    if let Some(o) = ops.i32("offset") {
        r.addr_start = o.max(0);
    }
    if let Some(e) = ops.i32("end") {
        if e > 0 {
            r.addr_end = e - info.len as i32;
        }
    }
    if let Some(ls) = ops.i32("loop_start") {
        r.addr_loop_start = ls - info.loop_start as i32;
    }
    if let Some(le) = ops.i32("loop_end") {
        r.addr_loop_end = le - info.loop_end as i32;
    }

    r.delay = ops.f32("ampeg_delay").unwrap_or(0.0).max(0.0);
    r.attack = ops.f32("ampeg_attack").unwrap_or(0.001).max(0.0);
    r.hold = ops.f32("ampeg_hold").unwrap_or(0.0).max(0.0);
    r.decay = ops.f32("ampeg_decay").unwrap_or(100.0).max(0.0);
    r.sustain = (ops.f32("ampeg_sustain").unwrap_or(100.0) / 100.0).clamp(0.0, 1.0);
    r.release = ops.f32("ampeg_release").unwrap_or(0.05).max(0.0);

    if let Some(hz) = ops.f32("cutoff") {
        if hz > 0.0 {
            r.filter_fc_cents = 1200.0 * (hz / 8.176).log2();
        }
    }
    if let Some(res) = ops.f32("resonance") {
        r.filter_q_cb = res * 10.0;
    }
    if let Some(vt) = ops.f32("fil_veltrack") {
        r.filter_veltrack_cents = vt.clamp(-9600.0, 9600.0);
    }
    // Only lowpasses are implemented. Applying a lowpass where the file asked
    // for a highpass would be worse than applying nothing, so anything else
    // switches the filter off for the region rather than being approximated.
    if let Some(kind) = ops.get("fil_type") {
        if !kind.starts_with("lpf") {
            r.filter_fc_cents = 13500.0;
            r.filter_veltrack_cents = 0.0;
        }
    }
    if let Some(g) = ops.i32("group") {
        r.exclusive_class = g.clamp(0, 255) as u8;
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_names() {
        assert_eq!(parse_note_name("c4"), Some(60));
        assert_eq!(parse_note_name("a4"), Some(69));
        assert_eq!(parse_note_name("c#4"), Some(61));
        assert_eq!(parse_note_name("c-1"), Some(0));
    }

    #[test]
    fn value_with_spaces_survives() {
        let s = "My Sample Name.wav lokey=30";
        let end = find_value_end(s);
        assert_eq!(s[..end].trim(), "My Sample Name.wav");
    }

    #[test]
    fn single_value_runs_to_end() {
        let s = "60";
        assert_eq!(find_value_end(s), 2);
    }
}
