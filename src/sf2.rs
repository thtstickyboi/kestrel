//! SoundFont 2 loader.
//!
//! Parses the `pdta` hunk in full (it is small, a few MB even for a 10 GB
//! soundfont), works out which samples the presets actually reference, and
//! then pulls only those byte ranges out of `smpl`. A 10 GB soundfont where
//! one preset is in use costs one preset's worth of VRAM.
//!
//! Modulators are not implemented. The one that matters in practice, velocity
//! to initial attenuation, is applied directly in `bank::velocity_atten_cb`.

use crate::bank::*;
use crate::config::Config;
use crate::resample;
use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// Generator operator numbers used here.
const G_START_ADDRS: u16 = 0;
const G_END_ADDRS: u16 = 1;
const G_STARTLOOP_ADDRS: u16 = 2;
const G_ENDLOOP_ADDRS: u16 = 3;
const G_START_ADDRS_COARSE: u16 = 4;
const G_INITIAL_FILTER_FC: u16 = 8;
const G_INITIAL_FILTER_Q: u16 = 9;
const G_END_ADDRS_COARSE: u16 = 12;
const G_PAN: u16 = 17;
const G_DELAY_VOL_ENV: u16 = 33;
const G_ATTACK_VOL_ENV: u16 = 34;
const G_HOLD_VOL_ENV: u16 = 35;
const G_DECAY_VOL_ENV: u16 = 36;
const G_SUSTAIN_VOL_ENV: u16 = 37;
const G_RELEASE_VOL_ENV: u16 = 38;
const G_KEYNUM_TO_HOLD: u16 = 39;
const G_KEYNUM_TO_DECAY: u16 = 40;
const G_INSTRUMENT: u16 = 41;
const G_KEY_RANGE: u16 = 43;
const G_VEL_RANGE: u16 = 44;
const G_STARTLOOP_ADDRS_COARSE: u16 = 45;
const G_KEYNUM: u16 = 46;
const G_VELOCITY: u16 = 47;
const G_INITIAL_ATTENUATION: u16 = 48;
const G_ENDLOOP_ADDRS_COARSE: u16 = 50;
const G_COARSE_TUNE: u16 = 51;
const G_FINE_TUNE: u16 = 52;
const G_SAMPLE_ID: u16 = 53;
const G_SAMPLE_MODES: u16 = 54;
const G_SCALE_TUNING: u16 = 56;
const G_EXCLUSIVE_CLASS: u16 = 57;
const G_OVERRIDING_ROOT_KEY: u16 = 58;
const GEN_COUNT: usize = 60;

/// Generators a preset zone may not set, only an instrument zone may.
fn is_absolute_only(op: u16) -> bool {
    matches!(
        op,
        G_START_ADDRS
            | G_END_ADDRS
            | G_STARTLOOP_ADDRS
            | G_ENDLOOP_ADDRS
            | G_START_ADDRS_COARSE
            | G_END_ADDRS_COARSE
            | G_STARTLOOP_ADDRS_COARSE
            | G_ENDLOOP_ADDRS_COARSE
            | G_SAMPLE_ID
            | G_SAMPLE_MODES
            | G_EXCLUSIVE_CLASS
            | G_OVERRIDING_ROOT_KEY
            | G_KEYNUM
            | G_VELOCITY
            | G_INSTRUMENT
            | G_KEY_RANGE
            | G_VEL_RANGE
    )
}

fn default_generators() -> [i16; GEN_COUNT] {
    let mut g = [0i16; GEN_COUNT];
    g[G_INITIAL_FILTER_FC as usize] = 13500;
    g[21] = -12000; // delayModLFO
    g[23] = -12000; // delayVibLFO
    g[25] = -12000; // delayModEnv
    g[26] = -12000;
    g[27] = -12000;
    g[28] = -12000;
    g[30] = -12000;
    g[G_DELAY_VOL_ENV as usize] = -12000;
    g[G_ATTACK_VOL_ENV as usize] = -12000;
    g[G_HOLD_VOL_ENV as usize] = -12000;
    g[G_DECAY_VOL_ENV as usize] = -12000;
    g[G_RELEASE_VOL_ENV as usize] = -12000;
    g[G_KEY_RANGE as usize] = 0x7F00u16 as i16; // lo 0, hi 127
    g[G_VEL_RANGE as usize] = 0x7F00u16 as i16;
    g[G_KEYNUM as usize] = -1;
    g[G_VELOCITY as usize] = -1;
    g[G_SCALE_TUNING as usize] = 100;
    g[G_OVERRIDING_ROOT_KEY as usize] = -1;
    g
}

#[derive(Clone, Copy)]
struct Bag {
    gen_ndx: u16,
    _mod_ndx: u16,
}

#[derive(Clone, Copy)]
struct Gen {
    op: u16,
    amount: i16,
}

struct Shdr {
    name: String,
    start: u32,
    end: u32,
    start_loop: u32,
    end_loop: u32,
    rate: u32,
    original_pitch: u8,
    pitch_correction: i8,
    sample_type: u16,
}

struct Phdr {
    name: String,
    preset: u16,
    bank: u16,
    bag_ndx: u16,
}

struct Inst {
    _name: String,
    bag_ndx: u16,
}

struct Chunk {
    id: [u8; 4],
    offset: u64,
    size: u64,
}

fn read_chunk_header(r: &mut impl Read) -> Result<([u8; 4], u32)> {
    let mut id = [0u8; 4];
    r.read_exact(&mut id)?;
    let mut sz = [0u8; 4];
    r.read_exact(&mut sz)?;
    Ok((id, u32::from_le_bytes(sz)))
}

/// Walk the RIFF tree without reading payloads.
fn scan_chunks(file: &mut File) -> Result<(Vec<Chunk>, Vec<Chunk>)> {
    file.seek(SeekFrom::Start(0))?;
    let file_len = file.metadata()?.len();
    let (id, _size) = read_chunk_header(file)?;
    if &id != b"RIFF" {
        bail!("not a RIFF file");
    }
    let mut form = [0u8; 4];
    file.read_exact(&mut form)?;
    if &form != b"sfbk" {
        bail!("RIFF form is {:?}, expected sfbk", String::from_utf8_lossy(&form));
    }

    let mut sdta = Vec::new();
    let mut pdta = Vec::new();
    let mut pos = 12u64;
    while pos + 8 <= file_len {
        file.seek(SeekFrom::Start(pos))?;
        let (id, size) = match read_chunk_header(file) {
            Ok(v) => v,
            Err(_) => break,
        };
        let size = (size as u64).min(file_len - pos - 8);
        if &id == b"LIST" {
            let mut list_id = [0u8; 4];
            file.read_exact(&mut list_id)?;
            let mut inner = pos + 12;
            let list_end = pos + 8 + size;
            let target = match &list_id {
                b"sdta" => Some(&mut sdta),
                b"pdta" => Some(&mut pdta),
                _ => None,
            };
            if let Some(target) = target {
                while inner + 8 <= list_end {
                    file.seek(SeekFrom::Start(inner))?;
                    let (cid, csz) = match read_chunk_header(file) {
                        Ok(v) => v,
                        Err(_) => break,
                    };
                    let csz = (csz as u64).min(list_end - inner - 8);
                    target.push(Chunk {
                        id: cid,
                        offset: inner + 8,
                        size: csz,
                    });
                    inner += 8 + csz + (csz & 1);
                }
            }
        }
        pos += 8 + size + (size & 1);
    }
    Ok((sdta, pdta))
}

fn load_chunk(file: &mut File, c: &Chunk) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(c.offset))?;
    let mut buf = vec![0u8; c.size as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

fn cstr20(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_string()
}

pub fn load(path: impl AsRef<Path>, cfg: &Config) -> Result<Bank> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let (sdta, pdta) = scan_chunks(&mut file)
        .with_context(|| format!("{}: scanning RIFF structure", path.display()))?;

    let smpl = sdta
        .iter()
        .find(|c| &c.id == b"smpl")
        .context("soundfont has no smpl chunk")?;
    let smpl_offset = smpl.offset;
    let smpl_frames = (smpl.size / 2) as u32;

    let find = |name: &[u8; 4]| -> Result<&Chunk> {
        pdta.iter()
            .find(|c| &c.id == name)
            .with_context(|| format!("pdta is missing {}", String::from_utf8_lossy(name)))
    };

    let phdr_raw = load_chunk(&mut file, find(b"phdr")?)?;
    let pbag_raw = load_chunk(&mut file, find(b"pbag")?)?;
    let pgen_raw = load_chunk(&mut file, find(b"pgen")?)?;
    let inst_raw = load_chunk(&mut file, find(b"inst")?)?;
    let ibag_raw = load_chunk(&mut file, find(b"ibag")?)?;
    let igen_raw = load_chunk(&mut file, find(b"igen")?)?;
    let shdr_raw = load_chunk(&mut file, find(b"shdr")?)?;

    let phdrs: Vec<Phdr> = phdr_raw
        .chunks_exact(38)
        .map(|c| Phdr {
            name: cstr20(&c[0..20]),
            preset: u16::from_le_bytes([c[20], c[21]]),
            bank: u16::from_le_bytes([c[22], c[23]]),
            bag_ndx: u16::from_le_bytes([c[24], c[25]]),
        })
        .collect();
    let pbags: Vec<Bag> = pbag_raw
        .chunks_exact(4)
        .map(|c| Bag {
            gen_ndx: u16::from_le_bytes([c[0], c[1]]),
            _mod_ndx: u16::from_le_bytes([c[2], c[3]]),
        })
        .collect();
    let pgens: Vec<Gen> = pgen_raw
        .chunks_exact(4)
        .map(|c| Gen {
            op: u16::from_le_bytes([c[0], c[1]]),
            amount: i16::from_le_bytes([c[2], c[3]]),
        })
        .collect();
    let insts: Vec<Inst> = inst_raw
        .chunks_exact(22)
        .map(|c| Inst {
            _name: cstr20(&c[0..20]),
            bag_ndx: u16::from_le_bytes([c[20], c[21]]),
        })
        .collect();
    let ibags: Vec<Bag> = ibag_raw
        .chunks_exact(4)
        .map(|c| Bag {
            gen_ndx: u16::from_le_bytes([c[0], c[1]]),
            _mod_ndx: u16::from_le_bytes([c[2], c[3]]),
        })
        .collect();
    let igens: Vec<Gen> = igen_raw
        .chunks_exact(4)
        .map(|c| Gen {
            op: u16::from_le_bytes([c[0], c[1]]),
            amount: i16::from_le_bytes([c[2], c[3]]),
        })
        .collect();
    let shdrs: Vec<Shdr> = shdr_raw
        .chunks_exact(46)
        .map(|c| Shdr {
            name: cstr20(&c[0..20]),
            start: u32::from_le_bytes(c[20..24].try_into().unwrap()),
            end: u32::from_le_bytes(c[24..28].try_into().unwrap()),
            start_loop: u32::from_le_bytes(c[28..32].try_into().unwrap()),
            end_loop: u32::from_le_bytes(c[32..36].try_into().unwrap()),
            rate: u32::from_le_bytes(c[36..40].try_into().unwrap()),
            original_pitch: c[40],
            pitch_correction: c[41] as i8,
            sample_type: u16::from_le_bytes([c[44], c[45]]),
        })
        .collect();

    if phdrs.len() < 2 {
        bail!("soundfont has no presets");
    }

    // ---- flatten every preset into regions --------------------------------
    let mut regions: Vec<Region> = Vec::new();
    let mut presets: Vec<Preset> = Vec::new();
    let mut used_samples: BTreeSet<u32> = BTreeSet::new();

    // phdr, pbag, inst, ibag all carry a terminal record whose index bounds
    // the previous one.
    for pi in 0..phdrs.len().saturating_sub(1) {
        let ph = &phdrs[pi];
        let bag_lo = ph.bag_ndx as usize;
        let bag_hi = phdrs[pi + 1].bag_ndx as usize;
        if bag_lo >= bag_hi || bag_hi > pbags.len() {
            continue;
        }

        let mut preset_global = [0i16; GEN_COUNT];
        let mut preset_global_set = [false; GEN_COUNT];
        let mut first_zone = true;
        let mut region_ids: Vec<u32> = Vec::new();

        for bi in bag_lo..bag_hi {
            let g_lo = pbags[bi].gen_ndx as usize;
            let g_hi = if bi + 1 < pbags.len() {
                pbags[bi + 1].gen_ndx as usize
            } else {
                pgens.len()
            };
            if g_lo > g_hi || g_hi > pgens.len() {
                continue;
            }

            let mut zone = [0i16; GEN_COUNT];
            let mut zone_set = [false; GEN_COUNT];
            let mut instrument: Option<u16> = None;
            for g in &pgens[g_lo..g_hi] {
                if g.op == G_INSTRUMENT {
                    instrument = Some(g.amount as u16);
                } else if (g.op as usize) < GEN_COUNT {
                    zone[g.op as usize] = g.amount;
                    zone_set[g.op as usize] = true;
                }
            }

            let Some(inst_idx) = instrument else {
                // A zone with no instrument terminal generator is the global
                // zone, and only the first zone may be global.
                if first_zone {
                    preset_global = zone;
                    preset_global_set = zone_set;
                }
                first_zone = false;
                continue;
            };
            first_zone = false;

            // Preset-level values: zone overrides global.
            let mut p_gen = preset_global;
            let mut p_set = preset_global_set;
            for i in 0..GEN_COUNT {
                if zone_set[i] {
                    p_gen[i] = zone[i];
                    p_set[i] = true;
                }
            }

            let p_key = range_of(&p_gen, &p_set, G_KEY_RANGE);
            let p_vel = range_of(&p_gen, &p_set, G_VEL_RANGE);

            let Some(inst) = insts.get(inst_idx as usize) else {
                continue;
            };
            let ib_lo = inst.bag_ndx as usize;
            let ib_hi = insts
                .get(inst_idx as usize + 1)
                .map(|i| i.bag_ndx as usize)
                .unwrap_or(ibags.len());
            if ib_lo >= ib_hi || ib_hi > ibags.len() {
                continue;
            }

            let mut inst_global = default_generators();
            let mut first_izone = true;

            for ibi in ib_lo..ib_hi {
                let ig_lo = ibags[ibi].gen_ndx as usize;
                let ig_hi = if ibi + 1 < ibags.len() {
                    ibags[ibi + 1].gen_ndx as usize
                } else {
                    igens.len()
                };
                if ig_lo > ig_hi || ig_hi > igens.len() {
                    continue;
                }

                let mut i_gen = inst_global;
                let mut sample_id: Option<u16> = None;
                for g in &igens[ig_lo..ig_hi] {
                    if g.op == G_SAMPLE_ID {
                        sample_id = Some(g.amount as u16);
                    } else if (g.op as usize) < GEN_COUNT {
                        i_gen[g.op as usize] = g.amount;
                    }
                }

                let Some(sid) = sample_id else {
                    if first_izone {
                        inst_global = i_gen;
                    }
                    first_izone = false;
                    continue;
                };
                first_izone = false;

                let Some(sh) = shdrs.get(sid as usize) else {
                    continue;
                };
                if sh.sample_type & 0x8000 != 0 {
                    continue; // ROM sample, we have no ROM
                }
                if sh.end <= sh.start || sh.end > smpl_frames {
                    continue;
                }

                let i_key = range_of(&i_gen, &[true; GEN_COUNT], G_KEY_RANGE);
                let i_vel = range_of(&i_gen, &[true; GEN_COUNT], G_VEL_RANGE);
                let key = match intersect(p_key, i_key) {
                    Some(r) => r,
                    None => continue,
                };
                let vel = match intersect(p_vel, i_vel) {
                    Some(r) => r,
                    None => continue,
                };

                // Preset generators offset the instrument's, except for the
                // ones the spec reserves to instrument level.
                let mut f = i_gen;
                for op in 0..GEN_COUNT {
                    if p_set[op] && !is_absolute_only(op as u16) {
                        f[op] = f[op].saturating_add(p_gen[op]);
                    }
                }

                used_samples.insert(sid as u32);
                let region = build_region(&f, sid as u32, key, vel);
                region_ids.push(regions.len() as u32);
                regions.push(region);
            }
        }

        if !region_ids.is_empty() {
            presets.push(Preset {
                bank: ph.bank,
                program: ph.preset,
                name: ph.name.clone(),
                regions: region_ids,
                key_index: Vec::new(),
                key_regions: Vec::new(),
            });
        }
    }

    if presets.is_empty() {
        bail!("{}: no usable presets", path.display());
    }

    // ---- pull the referenced sample data ---------------------------------
    let (pool, samples, pool_rate) = build_pool(
        &mut file,
        smpl_offset,
        &shdrs,
        &used_samples,
        &mut regions,
        cfg,
    )?;

    let mut bank = Bank {
        pool,
        pool_rate,
        samples,
        regions,
        params: Vec::new(),
        gain_table: Vec::new(),
        delay_frames: Vec::new(),
        key_ok: Vec::new(),
        presets,
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

fn range_of(gen: &[i16; GEN_COUNT], set: &[bool; GEN_COUNT], op: u16) -> (u8, u8) {
    if !set[op as usize] {
        return (0, 127);
    }
    let raw = gen[op as usize] as u16;
    let lo = (raw & 0xFF) as u8;
    let hi = (raw >> 8) as u8;
    if lo > hi {
        (hi, lo)
    } else {
        (lo, hi)
    }
}

fn intersect(a: (u8, u8), b: (u8, u8)) -> Option<(u8, u8)> {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    if lo > hi {
        None
    } else {
        Some((lo, hi))
    }
}

fn build_region(g: &[i16; GEN_COUNT], sample: u32, key: (u8, u8), vel: (u8, u8)) -> Region {
    let sustain_cb = g[G_SUSTAIN_VOL_ENV as usize] as f32;
    Region {
        sample,
        key_lo: key.0,
        key_hi: key.1,
        vel_lo: vel.0,
        vel_hi: vel.1,
        root_key_override: g[G_OVERRIDING_ROOT_KEY as usize],
        fixed_key: g[G_KEYNUM as usize],
        fixed_vel: g[G_VELOCITY as usize],
        coarse_tune: g[G_COARSE_TUNE as usize],
        fine_tune: g[G_FINE_TUNE as usize],
        scale_tuning: g[G_SCALE_TUNING as usize],
        attenuation_cb: g[G_INITIAL_ATTENUATION as usize] as f32,
        pan: (g[G_PAN as usize] as f32 / 500.0).clamp(-1.0, 1.0),
        loop_mode: match g[G_SAMPLE_MODES as usize] & 3 {
            1 => LoopMode::Continuous,
            3 => LoopMode::UntilRelease,
            _ => LoopMode::NoLoop,
        },
        addr_start: g[G_START_ADDRS as usize] as i32
            + g[G_START_ADDRS_COARSE as usize] as i32 * 32768,
        addr_end: g[G_END_ADDRS as usize] as i32 + g[G_END_ADDRS_COARSE as usize] as i32 * 32768,
        addr_loop_start: g[G_STARTLOOP_ADDRS as usize] as i32
            + g[G_STARTLOOP_ADDRS_COARSE as usize] as i32 * 32768,
        addr_loop_end: g[G_ENDLOOP_ADDRS as usize] as i32
            + g[G_ENDLOOP_ADDRS_COARSE as usize] as i32 * 32768,
        exclusive_class: g[G_EXCLUSIVE_CLASS as usize].clamp(0, 255) as u8,
        delay: timecents_to_secs(g[G_DELAY_VOL_ENV as usize] as f32),
        attack: timecents_to_secs(g[G_ATTACK_VOL_ENV as usize] as f32),
        hold: timecents_to_secs(g[G_HOLD_VOL_ENV as usize] as f32),
        decay: timecents_to_secs(g[G_DECAY_VOL_ENV as usize] as f32),
        sustain: cb_to_gain(sustain_cb.max(0.0)),
        release: timecents_to_secs(g[G_RELEASE_VOL_ENV as usize] as f32),
        keynum_to_hold: g[G_KEYNUM_TO_HOLD as usize],
        keynum_to_decay: g[G_KEYNUM_TO_DECAY as usize],
        filter_fc_cents: g[G_INITIAL_FILTER_FC as usize] as f32,
        filter_q_cb: g[G_INITIAL_FILTER_Q as usize] as f32,
        params_base: 0,
        filter_veltrack_cents: 0.0,
        params_stride: 0,
        params_vel_span: 1,
    }
}

/// Frames of silence between pool entries so the interpolator can read past
/// the end of a sample without touching its neighbour.
const POOL_GUARD: u32 = 8;

fn build_pool(
    file: &mut File,
    smpl_offset: u64,
    shdrs: &[Shdr],
    used: &BTreeSet<u32>,
    regions: &mut [Region],
    cfg: &Config,
) -> Result<(Vec<i16>, Vec<SampleInfo>, u32)> {
    // Decide the pool rate. Start at the output rate and halve it until the
    // estimated pool fits the budget. Degrade the pool rather than fail the
    // load: a downsampled render beats no render.
    let mut pool_rate = if cfg.resample_pool { cfg.sample_rate } else { 0 };

    let raw_frames: u64 = used
        .iter()
        .filter_map(|&i| shdrs.get(i as usize))
        .map(|s| (s.end - s.start) as u64 + POOL_GUARD as u64)
        .sum();

    if pool_rate != 0 {
        loop {
            let est: u64 = used
                .iter()
                .filter_map(|&i| shdrs.get(i as usize))
                .map(|s| {
                    let n = (s.end - s.start) as f64 * pool_rate as f64 / s.rate.max(1) as f64;
                    n as u64 + POOL_GUARD as u64
                })
                .sum();
            if est * 2 <= cfg.sample_pool_budget || pool_rate <= 8000 {
                if est * 2 > cfg.sample_pool_budget {
                    log::warn!(
                        "sample pool is {:.1} MiB at {} Hz, over the {:.1} MiB budget; \
                         continuing anyway",
                        est as f64 * 2.0 / 1048576.0,
                        pool_rate,
                        cfg.sample_pool_budget as f64 / 1048576.0
                    );
                }
                break;
            }
            pool_rate /= 2;
            log::warn!(
                "sample pool does not fit the budget; downsampling the pool to {} Hz",
                pool_rate
            );
        }
    } else if raw_frames * 2 > cfg.sample_pool_budget {
        log::warn!(
            "sample pool is {:.1} MiB, over the {:.1} MiB budget; \
             enable --resample-pool to allow automatic downsampling",
            raw_frames as f64 * 2.0 / 1048576.0,
            cfg.sample_pool_budget as f64 / 1048576.0
        );
    }

    let mut pool: Vec<i16> = Vec::with_capacity(raw_frames as usize + 64);
    let mut samples: Vec<SampleInfo> = Vec::new();
    // Dense index for the sparse set of used sample ids.
    let mut remap = vec![u32::MAX; shdrs.len()];
    let mut reader = BufReader::with_capacity(1 << 20, file.try_clone()?);

    for &sid in used {
        let sh = &shdrs[sid as usize];
        let n = (sh.end - sh.start) as usize;
        let mut raw = vec![0i16; n];
        reader.seek(SeekFrom::Start(smpl_offset + sh.start as u64 * 2))?;
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut raw[..]);
            reader.read_exact(bytes)?;
        }

        let src_rate = sh.rate.max(1);
        let mut loop_start = sh.start_loop.saturating_sub(sh.start);
        let mut loop_end = sh.end_loop.saturating_sub(sh.start);

        let data = if pool_rate != 0 && pool_rate != src_rate {
            let ratio = pool_rate as f64 / src_rate as f64;
            let out = resample::resample_i16(&raw, ratio);
            loop_start = (loop_start as f64 * ratio).round() as u32;
            loop_end = (loop_end as f64 * ratio).round() as u32;
            out
        } else {
            raw
        };

        let len = data.len() as u32;
        loop_start = loop_start.min(len.saturating_sub(1));
        loop_end = loop_end.min(len);

        let start = pool.len() as u32;
        pool.extend_from_slice(&data);
        pool.extend(std::iter::repeat(0i16).take(POOL_GUARD as usize));

        remap[sid as usize] = samples.len() as u32;
        samples.push(SampleInfo {
            start,
            len,
            loop_start,
            loop_end,
            rate: if pool_rate != 0 { pool_rate } else { src_rate },
            root_key: sh.original_pitch.min(127),
            correction_cents: sh.pitch_correction as f32,
            resample_ratio: if pool_rate != 0 {
                pool_rate as f32 / src_rate as f32
            } else {
                1.0
            },
            name: sh.name.clone(),
        });
    }

    for r in regions.iter_mut() {
        let old = r.sample as usize;
        r.sample = remap.get(old).copied().unwrap_or(u32::MAX);
    }

    Ok((pool, samples, pool_rate))
}
