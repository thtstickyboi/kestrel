//! Streaming SMF parser and track merger.
//!
//! Nothing here ever holds a whole MIDI file in memory. Each track keeps a
//! 256 KiB window over its own chunk, and a min-heap merges the per-track
//! cursors into one tick-ordered stream. A 40 GB black MIDI costs the same
//! resident memory as a 4 KB one.

use anyhow::{bail, Context, Result};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const TRACK_BUF: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    NoteOn { ch: u8, key: u8, vel: u8 },
    NoteOff { ch: u8, key: u8 },
    Cc { ch: u8, num: u8, val: u8 },
    Program { ch: u8, val: u8 },
    PitchBend { ch: u8, val: i16 },
    /// Microseconds per quarter note.
    Tempo(u32),
    /// Anything the synth does not act on. Kept in the stream so callers can
    /// count events without a second pass.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Division {
    /// Ticks per quarter note.
    Ppq(u16),
    /// SMPTE: frames per second and ticks per frame.
    Smpte { fps: u8, ticks_per_frame: u8 },
}

/// A single track chunk with its own file cursor and read window.
struct TrackReader {
    file: File,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    /// Bytes of the track chunk not yet pulled into `buf`.
    remaining: u64,
    tick: u64,
    running: u8,
    ended: bool,
}

impl TrackReader {
    fn open(path: &Path, offset: u64, len: u64) -> Result<Self> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        Ok(TrackReader {
            file,
            buf: vec![0u8; TRACK_BUF].into_boxed_slice(),
            pos: 0,
            filled: 0,
            remaining: len,
            tick: 0,
            running: 0,
            ended: false,
        })
    }

    #[inline]
    fn fill(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        let want = self.remaining.min(TRACK_BUF as u64) as usize;
        let mut got = 0usize;
        while got < want {
            match self.file.read(&mut self.buf[got..want]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        self.remaining -= got as u64;
        self.pos = 0;
        self.filled = got;
        got > 0
    }

    #[inline(always)]
    fn byte(&mut self) -> Option<u8> {
        if self.pos == self.filled && !self.fill() {
            return None;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Some(b)
    }

    #[inline]
    fn varlen(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..8 {
            let b = self.byte()?;
            v = (v << 7) | (b & 0x7F) as u64;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn skip(&mut self, mut n: u64) -> Option<()> {
        while n > 0 {
            if self.pos == self.filled && !self.fill() {
                return None;
            }
            let avail = (self.filled - self.pos) as u64;
            let take = avail.min(n);
            self.pos += take as usize;
            n -= take;
        }
        Some(())
    }

    /// Decode one event, advancing `tick`. Returns None at end of track.
    fn next_event(&mut self) -> Option<Event> {
        if self.ended {
            return None;
        }
        let delta = match self.varlen() {
            Some(d) => d,
            None => {
                self.ended = true;
                return None;
            }
        };
        self.tick += delta;

        let mut status = match self.byte() {
            Some(b) => b,
            None => {
                self.ended = true;
                return None;
            }
        };

        // Running status: a data byte where a status byte was expected reuses
        // the previous channel status and is itself the first data byte.
        let first_data;
        if status < 0x80 {
            first_data = status;
            status = self.running;
            if status < 0x80 {
                // Garbage before any status byte was ever seen.
                self.ended = true;
                return None;
            }
        } else {
            if status < 0xF0 {
                self.running = status;
            } else if status != 0xF7 && status != 0xF0 {
                // Meta and realtime clear running status per the spec; many
                // writers disagree, so only clear on meta.
                if status == 0xFF {
                    self.running = 0;
                }
            }
            first_data = match status {
                0xFF | 0xF0 | 0xF7 => 0,
                _ => match self.byte() {
                    Some(b) => b,
                    None => {
                        self.ended = true;
                        return None;
                    }
                },
            };
        }

        let ch = status & 0x0F;
        match status & 0xF0 {
            0x80 => {
                let _vel = self.byte()?;
                Some(Event::NoteOff { ch, key: first_data & 0x7F })
            }
            0x90 => {
                let vel = self.byte()?;
                if vel == 0 {
                    Some(Event::NoteOff { ch, key: first_data & 0x7F })
                } else {
                    Some(Event::NoteOn {
                        ch,
                        key: first_data & 0x7F,
                        vel: vel & 0x7F,
                    })
                }
            }
            0xA0 => {
                let _ = self.byte()?;
                Some(Event::Other)
            }
            0xB0 => {
                let val = self.byte()?;
                Some(Event::Cc {
                    ch,
                    num: first_data & 0x7F,
                    val: val & 0x7F,
                })
            }
            0xC0 => Some(Event::Program { ch, val: first_data & 0x7F }),
            0xD0 => Some(Event::Other),
            0xE0 => {
                let msb = self.byte()?;
                let raw = ((msb as i16 & 0x7F) << 7) | (first_data as i16 & 0x7F);
                Some(Event::PitchBend { ch, val: raw - 8192 })
            }
            _ => match status {
                0xFF => {
                    let meta = self.byte()?;
                    let len = self.varlen()?;
                    if meta == 0x2F {
                        self.skip(len)?;
                        self.ended = true;
                        return Some(Event::Other);
                    }
                    if meta == 0x51 && len == 3 {
                        let a = self.byte()? as u32;
                        let b = self.byte()? as u32;
                        let c = self.byte()? as u32;
                        return Some(Event::Tempo((a << 16) | (b << 8) | c));
                    }
                    self.skip(len)?;
                    Some(Event::Other)
                }
                0xF0 | 0xF7 => {
                    let len = self.varlen()?;
                    self.skip(len)?;
                    Some(Event::Other)
                }
                _ => Some(Event::Other),
            },
        }
    }
}

/// Tick-ordered merge of every track in a standard MIDI file.
pub struct MidiStream {
    readers: Vec<TrackReader>,
    /// (tick, track index) so ties break on track order, deterministically.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    pending: Vec<Option<Event>>,
    pub division: Division,
    pub format: u16,
    pub track_count: u16,
    pub path: PathBuf,
}

impl MidiStream {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let file_len = file.metadata()?.len();

        let mut hdr = [0u8; 14];
        file.read_exact(&mut hdr)
            .with_context(|| format!("{}: too short for an SMF header", path.display()))?;
        if &hdr[0..4] != b"MThd" {
            bail!("{}: not a standard MIDI file", path.display());
        }
        let hdr_len = u32::from_be_bytes(hdr[4..8].try_into().unwrap()) as u64;
        let format = u16::from_be_bytes(hdr[8..10].try_into().unwrap());
        let track_count = u16::from_be_bytes(hdr[10..12].try_into().unwrap());
        let div_raw = i16::from_be_bytes(hdr[12..14].try_into().unwrap());
        let division = if div_raw > 0 {
            Division::Ppq(div_raw as u16)
        } else {
            Division::Smpte {
                fps: (-(div_raw >> 8)) as u8,
                ticks_per_frame: (div_raw & 0xFF) as u8,
            }
        };

        // Walk the chunk list. Only the 8-byte headers are read, so this is a
        // handful of seeks even on a 40 GB file.
        let mut offset = 8 + hdr_len;
        let mut locs: Vec<(u64, u64)> = Vec::new();
        while offset + 8 <= file_len {
            file.seek(SeekFrom::Start(offset))?;
            let mut ch = [0u8; 8];
            if file.read_exact(&mut ch).is_err() {
                break;
            }
            let len = u32::from_be_bytes(ch[4..8].try_into().unwrap()) as u64;
            let data_start = offset + 8;
            let avail = file_len.saturating_sub(data_start);
            let len = len.min(avail);
            if &ch[0..4] == b"MTrk" {
                locs.push((data_start, len));
            } else if len == 0 {
                // An unrecognised tag with no body is not a chunk, and the
                // walk must stop rather than step over it. Exporters pad the
                // end of a file with zeros, and every 8 zero bytes then parse
                // as one of these: one real 6.15 GB file carries 1.38 GB of
                // padding after its last track, which is 172 million
                // seek-and-read pairs at 8 bytes a step. It finishes
                // eventually and it looks exactly like a hang -- one core
                // busy, the disk idle, and nothing logged between the
                // soundfont and the track count.
                let trailing = file_len - offset;
                if trailing > 0 {
                    log::warn!(
                        "{}: {} bytes after the last chunk are not chunks; ignored",
                        path.display(),
                        trailing
                    );
                }
                break;
            }
            offset = data_start + len;
        }

        if locs.is_empty() {
            bail!("{}: no MTrk chunks", path.display());
        }
        if locs.len() != track_count as usize {
            log::warn!(
                "{}: header claims {} tracks, found {}",
                path.display(),
                track_count,
                locs.len()
            );
        }

        let mut readers = Vec::with_capacity(locs.len());
        for (off, len) in &locs {
            readers.push(TrackReader::open(&path, *off, *len)?);
        }

        let mut pending = vec![None; readers.len()];
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (i, r) in readers.iter_mut().enumerate() {
            if let Some(ev) = r.next_event() {
                pending[i] = Some(ev);
                heap.push(Reverse((r.tick, i as u32)));
            }
        }

        Ok(MidiStream {
            readers,
            heap,
            pending,
            division,
            format,
            track_count: locs.len() as u16,
            path,
        })
    }

    /// Next event in tick order, or None at the end of the file.
    ///
    /// Deliberately not an `Iterator`: the borrow of `self` that an iterator
    /// would need conflicts with the per-track readers this pulls from.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u64, Event)> {
        let Reverse((tick, idx)) = self.heap.pop()?;
        let i = idx as usize;
        let ev = self.pending[i].take().expect("heap entry without pending event");

        if let Some(next) = self.readers[i].next_event() {
            self.pending[i] = Some(next);
            self.heap.push(Reverse((self.readers[i].tick, idx)));
        }
        Some((tick, ev))
    }
}

/// Converts ticks to absolute output frames, tracking tempo changes as they
/// arrive in the merged stream.
#[derive(Debug, Clone)]
pub struct TempoClock {
    division: Division,
    sample_rate: f64,
    /// Tick at which the current tempo took effect.
    base_tick: u64,
    /// Frame position of `base_tick`.
    base_frame: f64,
    frames_per_tick: f64,
}

impl TempoClock {
    pub fn new(division: Division, sample_rate: u32) -> Self {
        let mut c = TempoClock {
            division,
            sample_rate: sample_rate as f64,
            base_tick: 0,
            base_frame: 0.0,
            frames_per_tick: 0.0,
        };
        c.set_tempo_raw(500_000); // 120 bpm until told otherwise
        c
    }

    fn set_tempo_raw(&mut self, us_per_qn: u32) {
        self.frames_per_tick = match self.division {
            Division::Ppq(ppq) => {
                let ppq = ppq.max(1) as f64;
                (us_per_qn as f64 * 1e-6) * self.sample_rate / ppq
            }
            Division::Smpte { fps, ticks_per_frame } => {
                let tps = fps.max(1) as f64 * ticks_per_frame.max(1) as f64;
                self.sample_rate / tps
            }
        };
    }

    /// Apply a tempo change that takes effect at `tick`.
    pub fn set_tempo(&mut self, tick: u64, us_per_qn: u32) {
        if matches!(self.division, Division::Smpte { .. }) {
            return; // SMPTE division ignores tempo meta events
        }
        self.base_frame = self.frame_at(tick);
        self.base_tick = tick;
        self.set_tempo_raw(us_per_qn);
    }

    #[inline]
    pub fn frame_at(&self, tick: u64) -> f64 {
        self.base_frame + (tick - self.base_tick) as f64 * self.frames_per_tick
    }
}

// ---------------------------------------------------------------------------
// writer, for generating test material
// ---------------------------------------------------------------------------

/// Minimal SMF writer. Only used by tests and the `gen-test-midi` CLI command,
/// but it lives here so the format constants stay in one file.
pub struct MidiWriter {
    tracks: Vec<Vec<u8>>,
    ppq: u16,
}

impl MidiWriter {
    pub fn new(ppq: u16) -> Self {
        MidiWriter { tracks: Vec::new(), ppq }
    }

    pub fn track(&mut self, events: Vec<(u64, [u8; 3], usize)>) {
        // events: (absolute tick, message bytes, message length)
        let mut sorted = events;
        sorted.sort_by_key(|e| e.0);
        let mut buf = Vec::new();
        let mut last = 0u64;
        for (tick, msg, len) in sorted {
            write_varlen(&mut buf, tick - last);
            last = tick;
            buf.extend_from_slice(&msg[..len]);
        }
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        self.tracks.push(buf);
    }

    pub fn tempo_track(&mut self, us_per_qn: u32) {
        let mut buf = Vec::new();
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x51, 0x03]);
        buf.extend_from_slice(&us_per_qn.to_be_bytes()[1..4]);
        write_varlen(&mut buf, 0);
        buf.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        self.tracks.push(buf);
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(File::create(path)?);
        f.write_all(b"MThd")?;
        f.write_all(&6u32.to_be_bytes())?;
        f.write_all(&1u16.to_be_bytes())?;
        f.write_all(&(self.tracks.len() as u16).to_be_bytes())?;
        f.write_all(&self.ppq.to_be_bytes())?;
        for t in &self.tracks {
            f.write_all(b"MTrk")?;
            f.write_all(&(t.len() as u32).to_be_bytes())?;
            f.write_all(t)?;
        }
        f.flush()?;
        Ok(())
    }
}

fn write_varlen(buf: &mut Vec<u8>, mut v: u64) {
    let mut stack = [0u8; 10];
    let mut n = 0;
    stack[n] = (v & 0x7F) as u8;
    n += 1;
    v >>= 7;
    while v > 0 {
        stack[n] = ((v & 0x7F) as u8) | 0x80;
        n += 1;
        v >>= 7;
    }
    for i in (0..n).rev() {
        buf.push(stack[i]);
    }
}

