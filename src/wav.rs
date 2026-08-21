//! Minimal RIFF/WAVE reader and writer.
//!
//! Written by hand rather than pulled in as a dependency because the SFZ loader
//! needs the `smpl` chunk loop points, the writer needs to stream multi-gigabyte
//! files without buffering them, and the tests need bit-exact comparison.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer.
    Pcm16,
    /// 32-bit IEEE float.
    Float32,
}

impl SampleFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "pcm16" | "16" | "s16" | "int16" => Some(SampleFormat::Pcm16),
            "float32" | "f32" | "32" | "float" => Some(SampleFormat::Float32),
            _ => None,
        }
    }
    fn bits(self) -> u16 {
        match self {
            SampleFormat::Pcm16 => 16,
            SampleFormat::Float32 => 32,
        }
    }
    fn tag(self) -> u16 {
        match self {
            SampleFormat::Pcm16 => 1,     // WAVE_FORMAT_PCM
            SampleFormat::Float32 => 3,   // WAVE_FORMAT_IEEE_FLOAT
        }
    }
    fn bytes(self) -> u32 {
        u32::from(self.bits()) / 8
    }
}

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

/// Streaming WAVE writer. Sizes are patched into the header on `finish`.
pub struct WavWriter {
    out: BufWriter<File>,
    format: SampleFormat,
    channels: u16,
    data_bytes: u64,
    finished: bool,
}

impl WavWriter {
    pub fn create(
        path: impl AsRef<Path>,
        sample_rate: u32,
        channels: u16,
        format: SampleFormat,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut out = BufWriter::with_capacity(1 << 20, file);

        let block_align = channels as u32 * format.bytes();
        out.write_all(b"RIFF")?;
        out.write_all(&0u32.to_le_bytes())?; // patched
        out.write_all(b"WAVE")?;
        out.write_all(b"fmt ")?;
        out.write_all(&16u32.to_le_bytes())?;
        out.write_all(&format.tag().to_le_bytes())?;
        out.write_all(&channels.to_le_bytes())?;
        out.write_all(&sample_rate.to_le_bytes())?;
        out.write_all(&(sample_rate * block_align).to_le_bytes())?;
        out.write_all(&(block_align as u16).to_le_bytes())?;
        out.write_all(&format.bits().to_le_bytes())?;
        out.write_all(b"data")?;
        out.write_all(&0u32.to_le_bytes())?; // patched

        Ok(WavWriter {
            out,
            format,
            channels,
            data_bytes: 0,
            finished: false,
        })
    }

    /// Write one interleaved block of f32 samples.
    pub fn write_block(&mut self, samples: &[f32]) -> Result<()> {
        match self.format {
            SampleFormat::Float32 => {
                // bytemuck keeps this a single memcpy on little-endian hosts.
                let bytes: &[u8] = bytemuck::cast_slice(samples);
                self.out.write_all(bytes)?;
                self.data_bytes += bytes.len() as u64;
            }
            SampleFormat::Pcm16 => {
                let mut buf = Vec::with_capacity(samples.len() * 2);
                for &s in samples {
                    let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                self.out.write_all(&buf)?;
                self.data_bytes += buf.len() as u64;
            }
        }
        Ok(())
    }

    pub fn frames_written(&self) -> u64 {
        self.data_bytes / (self.channels as u64 * self.format.bytes() as u64)
    }

    pub fn finish(mut self) -> Result<u64> {
        self.finish_inner()?;
        Ok(self.data_bytes)
    }

    fn finish_inner(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        self.out.flush()?;

        let riff_size = 36u64 + self.data_bytes;
        if riff_size > u32::MAX as u64 {
            // Loud, not silent: a >4 GiB render cannot be described by a plain
            // RIFF header. The audio data is intact, but the size fields are
            // saturated and some players will stop early.
            log::error!(
                "output exceeds 4 GiB ({} bytes); RIFF size fields saturated, \
                 use --format pcm16 or split the render",
                self.data_bytes
            );
        }
        let file = self.out.get_mut();
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(riff_size.min(u32::MAX as u64) as u32).to_le_bytes())?;
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&(self.data_bytes.min(u32::MAX as u64) as u32).to_le_bytes())?;
        file.flush()?;
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        if let Err(e) = self.finish_inner() {
            log::error!("failed to finalize wav header: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// reader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    /// Deinterleaved to mono by taking the first channel if `channels > 1`.
    /// The synth is a mono-sample engine; stereo SFZ samples are handled by
    /// the caller splitting them into two regions.
    pub interleaved: Vec<f32>,
    /// From the `smpl` chunk, if present: (start, end) in frames.
    pub loop_points: Option<(u32, u32)>,
    /// From the `smpl` chunk: MIDI note the sample was recorded at.
    pub root_key: Option<u8>,
    /// From the `smpl` chunk: pitch correction in cents.
    pub fine_tune_cents: f32,
}

impl WavData {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.interleaved.len() / self.channels as usize
        }
    }

    /// Channel `ch` as a mono i16 buffer, which is the pool's storage format.
    pub fn channel_i16(&self, ch: usize) -> Vec<i16> {
        let nch = self.channels as usize;
        let ch = ch.min(nch.saturating_sub(1));
        self.interleaved
            .iter()
            .skip(ch)
            .step_by(nch.max(1))
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect()
    }
}

fn rd_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_u16(r: &mut impl Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn rd_tag(r: &mut impl Read) -> Result<[u8; 4]> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(b)
}

pub fn read(path: impl AsRef<Path>) -> Result<WavData> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 16, file);

    if &rd_tag(&mut r)? != b"RIFF" {
        bail!("{}: not a RIFF file", path.display());
    }
    let _riff_size = rd_u32(&mut r)?;
    if &rd_tag(&mut r)? != b"WAVE" {
        bail!("{}: not a WAVE file", path.display());
    }

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut tag = 0u16;
    let mut data: Option<Vec<u8>> = None;
    let mut loop_points = None;
    let mut root_key = None;
    let mut fine_tune_cents = 0.0f32;

    loop {
        let id = match rd_tag(&mut r) {
            Ok(t) => t,
            Err(_) => break,
        };
        let size = rd_u32(&mut r)? as u64;
        let padded = size + (size & 1);
        if size > file_len {
            bail!("{}: chunk {:?} claims {} bytes", path.display(), std::str::from_utf8(&id), size);
        }
        match &id {
            b"fmt " => {
                tag = rd_u16(&mut r)?;
                channels = rd_u16(&mut r)?;
                sample_rate = rd_u32(&mut r)?;
                let _byte_rate = rd_u32(&mut r)?;
                let _block_align = rd_u16(&mut r)?;
                bits = rd_u16(&mut r)?;
                if tag == 0xFFFE && size >= 40 {
                    // WAVE_FORMAT_EXTENSIBLE: the real tag is the first two
                    // bytes of the sub-format GUID.
                    let _cb = rd_u16(&mut r)?;
                    let _valid_bits = rd_u16(&mut r)?;
                    let _mask = rd_u32(&mut r)?;
                    tag = rd_u16(&mut r)?;
                    r.seek_relative(padded as i64 - 26)?;
                } else {
                    r.seek_relative(padded as i64 - 16)?;
                }
            }
            b"data" => {
                let mut buf = vec![0u8; size as usize];
                r.read_exact(&mut buf)?;
                if padded > size {
                    r.seek_relative(1)?;
                }
                data = Some(buf);
            }
            b"smpl" => {
                let mut buf = vec![0u8; padded as usize];
                r.read_exact(&mut buf)?;
                if buf.len() >= 36 {
                    let midi_note = u32::from_le_bytes(buf[20..24].try_into().unwrap());
                    let pitch_frac = u32::from_le_bytes(buf[24..28].try_into().unwrap());
                    let num_loops = u32::from_le_bytes(buf[28..32].try_into().unwrap());
                    if midi_note < 128 {
                        root_key = Some(midi_note as u8);
                    }
                    // MIDIPitchFraction is a 0.32 fraction of a semitone.
                    fine_tune_cents = (pitch_frac as f64 / 4294967296.0 * 100.0) as f32;
                    if num_loops > 0 && buf.len() >= 36 + 24 {
                        let start = u32::from_le_bytes(buf[44..48].try_into().unwrap());
                        let end = u32::from_le_bytes(buf[48..52].try_into().unwrap());
                        loop_points = Some((start, end));
                    }
                }
            }
            _ => {
                r.seek_relative(padded as i64)?;
            }
        }
    }

    let data = data.with_context(|| format!("{}: no data chunk", path.display()))?;
    if channels == 0 {
        bail!("{}: no fmt chunk", path.display());
    }

    let interleaved = decode_samples(&data, tag, bits)
        .with_context(|| format!("{}: unsupported format tag {tag} / {bits} bits", path.display()))?;

    Ok(WavData {
        sample_rate,
        channels,
        interleaved,
        loop_points,
        root_key,
        fine_tune_cents,
    })
}

fn decode_samples(data: &[u8], tag: u16, bits: u16) -> Result<Vec<f32>> {
    let out = match (tag, bits) {
        (1, 8) => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (1, 24) => data
            .chunks_exact(3)
            .map(|c| {
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        (1, 32) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (3, 64) => data
            .chunks_exact(8)
            .map(|c| {
                f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect(),
        _ => bail!("unsupported"),
    };
    Ok(out)
}
