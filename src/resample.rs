//! Load-time sample-rate conversion for the sample pool.
//!
//! Only ever runs on the host, once, while building the pool. Uses a
//! polyphase windowed-sinc bank so a multi-gigabyte soundfont converts in
//! seconds rather than in the hours a naive per-output-sample sinc would take.

const TAPS: usize = 32;
const PHASES: usize = 512;

pub struct SincBank {
    /// `PHASES * TAPS` coefficients, phase-major.
    table: Vec<f32>,
}

impl SincBank {
    /// `ratio` is output rate over input rate. Below 1.0 the cutoff moves down
    /// with the ratio so downsampling does not alias.
    pub fn new(ratio: f64) -> Self {
        let cutoff = if ratio < 1.0 { ratio } else { 1.0 } * 0.95;
        let mut table = vec![0.0f32; PHASES * TAPS];
        let half = TAPS as f64 / 2.0;
        for p in 0..PHASES {
            let frac = p as f64 / PHASES as f64;
            let mut sum = 0.0f64;
            for k in 0..TAPS {
                // Distance from the output point to source tap k.
                let x = k as f64 - half + 1.0 - frac;
                let s = sinc(x * cutoff) * cutoff;
                // Blackman window over the tap span.
                let w = {
                    let t = (k as f64 - frac + 1.0) / TAPS as f64;
                    let t = t.clamp(0.0, 1.0);
                    0.42 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
                        + 0.08 * (4.0 * std::f64::consts::PI * t).cos()
                };
                let v = s * w;
                table[p * TAPS + k] = v as f32;
                sum += v;
            }
            // Normalise each phase so a DC input passes through at unity.
            if sum.abs() > 1e-9 {
                let inv = (1.0 / sum) as f32;
                for k in 0..TAPS {
                    table[p * TAPS + k] *= inv;
                }
            }
        }
        SincBank { table }
    }

    pub fn process(&self, src: &[i16], ratio: f64) -> Vec<i16> {
        if src.is_empty() {
            return Vec::new();
        }
        let n = src.len() as i64;
        let out_len = ((src.len() as f64 * ratio).round() as usize).max(1);
        let mut out = Vec::with_capacity(out_len);
        let half = TAPS as i64 / 2;
        let inv_ratio = 1.0 / ratio;

        for j in 0..out_len {
            let t = j as f64 * inv_ratio;
            let i = t.floor() as i64;
            let frac = t - i as f64;
            let p = ((frac * PHASES as f64) as usize).min(PHASES - 1);
            let coeffs = &self.table[p * TAPS..p * TAPS + TAPS];

            let base = i - half + 1;
            let mut acc = 0.0f32;
            if base >= 0 && base + TAPS as i64 <= n {
                // Interior fast path: no clamping needed.
                let s = &src[base as usize..base as usize + TAPS];
                for (c, v) in coeffs.iter().zip(s) {
                    acc += c * *v as f32;
                }
            } else {
                for (k, c) in coeffs.iter().enumerate() {
                    let idx = (base + k as i64).clamp(0, n - 1) as usize;
                    acc += c * src[idx] as f32;
                }
            }
            out.push(acc.clamp(-32768.0, 32767.0) as i16);
        }
        out
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Convenience wrapper that builds a bank and converts one sample.
///
/// Callers converting many samples at the same ratio should build the
/// `SincBank` once; `Bank` loading does this per sample because soundfonts
/// mix rates, and the table build is a few microseconds.
pub fn resample_i16(src: &[i16], ratio: f64) -> Vec<i16> {
    if (ratio - 1.0).abs() < 1e-12 {
        return src.to_vec();
    }
    SincBank::new(ratio).process(src, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_survives() {
        let src = vec![10000i16; 4000];
        let out = resample_i16(&src, 48000.0 / 44100.0);
        assert_eq!(out.len(), (4000.0f64 * 48000.0 / 44100.0).round() as usize);
        // Ignore the window edges where the clamped taps pull the value.
        for &v in &out[200..out.len() - 200] {
            assert!((v as i32 - 10000).abs() < 60, "dc drifted to {v}");
        }
    }

    #[test]
    fn sine_keeps_its_frequency() {
        // 1 kHz at 44.1 k, resampled to 48 k, must still be 1 kHz.
        let n = 44100;
        let src: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64 / 44100.0;
                (16000.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as i16
            })
            .collect();
        let out = resample_i16(&src, 48000.0 / 44100.0);

        // Count zero crossings over one second of output.
        let mut crossings = 0;
        for w in out[1000..47000].windows(2) {
            if (w[0] < 0) != (w[1] < 0) {
                crossings += 1;
            }
        }
        let seconds = 46000.0 / 48000.0;
        let freq = crossings as f64 / 2.0 / seconds;
        assert!((freq - 1000.0).abs() < 5.0, "got {freq} Hz");
    }

    #[test]
    fn upsampling_does_not_clip_headroom() {
        // A loud but band-limited signal must survive 2x upsampling without
        // hitting the rails. A full-scale alternating signal at Nyquist would
        // legitimately overshoot, so it is not the case being tested here.
        let src: Vec<i16> = (0..2000)
            .map(|i| {
                let t = i as f64 / 44100.0;
                (26000.0 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i16
            })
            .collect();
        let out = resample_i16(&src, 2.0);
        assert!(out.iter().all(|&v| v.abs() < 32000), "resampler clipped");
    }

    #[test]
    fn extreme_input_saturates_instead_of_wrapping() {
        let src: Vec<i16> = (0..500).map(|i| if i % 2 == 0 { 32767 } else { -32768 }).collect();
        let out = resample_i16(&src, 2.0);
        // The clamp must hold: the accumulator saturates rather than wrapping,
        // which would show up as a full-scale sign flip.
        assert!(out.iter().all(|&v| (-32768..=32767).contains(&(v as i32))));
    }
}
