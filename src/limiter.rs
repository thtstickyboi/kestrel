//! Soft limiter, ported from the one OmniConverter already ships.
//!
//! `OmniConverter/Extensions/Audio/Limiter.cs`, originally from Kiva by
//! Arduano. Black MIDI mixes clip constantly, and matching the curve the
//! existing pipeline uses means a render through this backend sits at the same
//! level as one through BASS or XSynth.
//!
//! Runs on the mixed stereo block on the host, not per voice, so it costs
//! nothing at a million voices. State carries across blocks, and the update is
//! a pure function of the previous state and the input, so it does not disturb
//! bit-exact reproducibility.

#[derive(Debug, Clone)]
pub struct Limiter {
    loudness_l: f64,
    loudness_r: f64,
    velocity_l: f64,
    velocity_r: f64,
    attack: f64,
    falloff: f64,
    min_thresh: f64,
    /// 0 disables the effect, 1 is full strength.
    pub strength: f64,
    /// Optional high-frequency energy limiter, off by default.
    pub reduce_high_pitch: bool,
    velocity_thresh: f64,
    first_sample: bool,
}

impl Limiter {
    pub fn new(sample_rate: u32) -> Self {
        Limiter {
            loudness_l: 1.0,
            loudness_r: 1.0,
            velocity_l: 0.0,
            velocity_r: 0.0,
            attack: 100.0,
            falloff: sample_rate as f64 / 3.0,
            min_thresh: 0.4,
            strength: 1.0,
            reduce_high_pitch: false,
            velocity_thresh: 1.0,
            first_sample: true,
        }
    }

    pub fn with_frequency_reduce(mut self, frequency_reduce: f64) -> Self {
        self.reduce_high_pitch = true;
        self.velocity_thresh = 1.0 / frequency_reduce;
        self
    }

    /// Process one interleaved stereo block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len() % 2, 0);
        let attack = self.attack;
        let falloff = self.falloff;

        for i in (0..buf.len()).step_by(2) {
            let in_l = buf[i] as f64;
            let in_r = buf[i + 1] as f64;
            let l_abs = in_l.abs();
            let r_abs = in_r.abs();

            self.loudness_l = if self.loudness_l > l_abs {
                (self.loudness_l * falloff + l_abs) / (falloff + 1.0)
            } else {
                (self.loudness_l * attack + l_abs) / (attack + 1.0)
            };
            self.loudness_r = if self.loudness_r > r_abs {
                (self.loudness_r * falloff + r_abs) / (falloff + 1.0)
            } else {
                (self.loudness_r * attack + r_abs) / (attack + 1.0)
            };

            if self.loudness_l < self.min_thresh {
                self.loudness_l = self.min_thresh;
            }
            if self.loudness_r < self.min_thresh {
                self.loudness_r = self.min_thresh;
            }

            let mut l = in_l / (self.loudness_l * self.strength + 2.0 * (1.0 - self.strength)) / 2.0;
            let mut r = in_r / (self.loudness_r * self.strength + 2.0 * (1.0 - self.strength)) / 2.0;

            if !self.first_sample {
                let dl = (in_l - l).abs();
                let dr = (in_r - r).abs();
                self.velocity_l = if self.velocity_l > dl {
                    (self.velocity_l * falloff + dl) / (falloff + 1.0)
                } else {
                    (self.velocity_l * attack + dl) / (attack + 1.0)
                };
                self.velocity_r = if self.velocity_r > dr {
                    (self.velocity_r * falloff + dr) / (falloff + 1.0)
                } else {
                    (self.velocity_r * attack + dr) / (attack + 1.0)
                };
            }
            self.first_sample = false;

            if self.reduce_high_pitch {
                if self.velocity_l > self.velocity_thresh {
                    l = l / self.velocity_l * self.velocity_thresh;
                }
                if self.velocity_r > self.velocity_thresh {
                    r = r / self.velocity_r * self.velocity_thresh;
                }
            }

            buf[i] = l as f32;
            buf[i + 1] = r as f32;
        }
    }
}

/// Hard clamp, always applied last so nothing leaves the renderer out of range.
/// Returns how many samples it had to move, which is the number of samples that
/// would otherwise have left the renderer clipped.
pub fn clamp_block(buf: &mut [f32]) -> u64 {
    let mut n = 0u64;
    for v in buf.iter_mut() {
        if *v > 1.0 || *v < -1.0 {
            n += 1;
            *v = v.clamp(-1.0, 1.0);
        }
    }
    n
}

// ---------------------------------------------------------------------------
// brickwall true-peak limiter
// ---------------------------------------------------------------------------

/// Which limiter runs on the mixed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterMode {
    /// No limiting. `clamp_block` still runs, so loud material hard-clips.
    Off,
    /// The port of the realtime limiter OmniConverter ships, above.
    ///
    /// **Deprecated for rendering.** It does not bound its own output, so
    /// `clamp_block` hard-clips behind it, and its 1/3 s release drags the
    /// level down after every loud moment. Kept only for level-matching a
    /// render against BASS or XSynth, which is the one thing it is better at.
    Omni,
    /// Lookahead true-peak brickwall. Guarantees the output never exceeds the
    /// ceiling, and only pulls the gain down around the peak that needs it.
    Brickwall,
}

impl LimiterMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" => Some(LimiterMode::Off),
            "omni" | "kiva" | "legacy" => Some(LimiterMode::Omni),
            "brickwall" | "brick" | "peak" => Some(LimiterMode::Brickwall),
            _ => None,
        }
    }
}

/// Phases of the true-peak oversampler, and taps per phase.
const TP_PHASES: usize = 4;
const TP_TAPS: usize = 8;

/// Lookahead true-peak brickwall limiter.
///
/// The trouble with the follower above is that it is a *feedback* design: it
/// sees a peak only once the peak has gone past, so it needs a long release to
/// avoid distorting, and that release is what drags the level down for a third
/// of a second after every loud moment. This one delays the audio and computes
/// the gain from what is about to arrive, so a 1 ms transient costs about a
/// millisecond of gain reduction and the material after it is left alone.
///
/// # Why the ceiling cannot be exceeded
///
/// Write `d[s]` for the detected true peak of input frame `s`, `L` for the
/// lookahead, and let the output at time `t` be input frame `t - L`. Then:
///
/// * `env[t]` is the largest `d[s]` for `s` in `[t-L, t]`, a sliding maximum.
/// * `gr[t]` is `ceiling / env[t]`, clamped to at most 1.
/// * `ma[t]` is the mean of `gr[s]` over the same window, and the gain that
///   gets applied is never above `ma[t]`.
///
/// For every `s` in that window, the window of `env[s]` is `[s-L, s]`, which
/// contains `t-L`. So `env[s]` is at least `d[t-L]`, and every `gr[s]` in the
/// average is therefore at most `ceiling / d[t-L]`. A mean of values that are
/// all at most that is at most that, so the emitted sample cannot exceed the
/// ceiling. The averaging is what keeps the gain continuous when a peak enters
/// the window instead of stepping, and the argument says it costs nothing in
/// safety to have it.
///
/// The detector is a symmetric FIR with its own group delay `g`, so what it
/// reports at `s` describes input `s - g`. The audio is therefore delayed by
/// `L + g` rather than `L`, which is exactly what makes `t - L - g` fall inside
/// every window in the average. Getting that wrong does not fail quietly: with
/// the delay left at `L` this let a spike through at 4.6x the ceiling.
///
/// The release is program dependent, in two stages; see `process`.
///
/// Costs `L + g` frames of latency; the render comes out delayed by that much.
pub struct Brickwall {
    ceiling: f64,
    look: usize,
    release_coef: f64,
    /// Attack and release coefficients of the sustained stage. Both zero when
    /// the stage is disabled, which makes it a plain single-stage limiter.
    sustain_atk: f64,
    sustain_rel: f64,
    /// Gain the sustained stage is holding: a slow envelope of the
    /// requirement, in both directions, so brief peaks barely move it.
    g_slow: f64,
    /// Gain the fast stage is holding, on top of the sustained one.
    g_fast: f64,
    /// Interleaved stereo delay line, `look + group` frames.
    delay: Vec<f32>,
    dpos: usize,
    /// Monotonic deque for the sliding maximum, decreasing, of (value, index).
    dq: std::collections::VecDeque<(f64, u64)>,
    /// Ring of `gr` values with a running sum, for the moving average.
    gr_ring: Vec<f64>,
    gr_pos: usize,
    gr_sum: f64,
    /// Release-only smoother state.
    gain: f64,
    /// Oversampler history, one per channel.
    hist: [[f64; TP_TAPS]; 2],
    /// Polyphase coefficients, indexed by phase then tap.
    poly: [[f64; TP_TAPS]; TP_PHASES],
    /// Frames the audio is delayed by: the lookahead plus the detector's own
    /// group delay.
    delay_frames: usize,
    true_peak: bool,
    idx: u64,
    /// Largest true peak seen at the input, for reporting.
    pub peak_in: f64,
    /// Smallest gain the limiter had to apply, for reporting.
    pub min_gain: f64,
}

impl Brickwall {
    pub fn new(
        sample_rate: u32,
        ceiling: f64,
        lookahead_ms: f64,
        release_ms: f64,
        sustain_ms: f64,
        true_peak: bool,
    ) -> Self {
        let sr = sample_rate as f64;
        let look = ((lookahead_ms * 1e-3 * sr).round() as usize).max(1);
        let rel = (release_ms * 1e-3 * sr).max(1.0);
        // The sustained stage engages over its own time constant and lets go
        // over four times that, so it settles onto a steady loud passage and
        // then takes its time coming back rather than breathing with the
        // material. Zero disables it and leaves a plain single-stage limiter.
        let (sustain_atk, sustain_rel) = if sustain_ms > 0.0 {
            let a = (sustain_ms * 1e-3 * sr).max(1.0);
            (
                1.0 - (-1.0f64 / a).exp(),
                1.0 - (-1.0f64 / (a * 4.0)).exp(),
            )
        } else {
            (0.0, 0.0)
        };
        // One-pole release. Expressed as a time constant so that a given
        // --limiter-release means the same thing at any sample rate.
        let release_coef = 1.0 - (-1.0 / rel).exp();
        // The oversampler is a symmetric FIR, so its output describes the input
        // this many samples back. The audio has to be delayed by the lookahead
        // *plus* that, or the first few entries of the moving average describe
        // inputs that do not include the sample being emitted -- and when the
        // required gain is tiny, a handful of stray 1.0s in the mean dominate
        // it. That is not a small error: it measured 4.6x over the ceiling.
        let group = if true_peak { TP_TAPS / 2 - 1 } else { 0 };
        let delay_frames = look + group;
        Brickwall {
            ceiling,
            look,
            release_coef,
            sustain_atk,
            sustain_rel,
            g_slow: 1.0,
            g_fast: 1.0,
            delay: vec![0.0; delay_frames * 2],
            dpos: 0,
            dq: std::collections::VecDeque::with_capacity(look + 2),
            gr_ring: vec![1.0; look + 1],
            gr_pos: 0,
            gr_sum: (look + 1) as f64,
            gain: 1.0,
            hist: [[0.0; TP_TAPS]; 2],
            poly: design_polyphase(),
            delay_frames,
            true_peak,
            idx: 0,
            peak_in: 0.0,
            min_gain: 1.0,
        }
    }

    /// Frames of latency this adds to the render.
    pub fn latency(&self) -> usize {
        self.delay_frames
    }

    /// True peak of one frame: the largest magnitude of the 4x oversampled
    /// signal, taken across both channels so the stereo image is preserved.
    /// With `true_peak` off this is the plain sample magnitude, which misses
    /// inter-sample overshoot but costs nothing.
    fn detect(&mut self, l: f64, r: f64) -> f64 {
        if !self.true_peak {
            return l.abs().max(r.abs());
        }
        let mut peak = 0.0f64;
        for (ch, v) in [l, r].into_iter().enumerate() {
            let h = &mut self.hist[ch];
            h.copy_within(0..TP_TAPS - 1, 1);
            h[0] = v;
            for ph in &self.poly {
                let mut acc = 0.0;
                for (c, x) in ph.iter().zip(h.iter()) {
                    acc += c * x;
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
        peak
    }

    /// Process one interleaved stereo block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        debug_assert_eq!(buf.len() % 2, 0);
        for i in (0..buf.len()).step_by(2) {
            let in_l = buf[i] as f64;
            let in_r = buf[i + 1] as f64;

            // Pull the delayed frame out, push the new one in.
            let d = self.dpos * 2;
            let out_l = self.delay[d] as f64;
            let out_r = self.delay[d + 1] as f64;
            self.delay[d] = buf[i];
            self.delay[d + 1] = buf[i + 1];
            self.dpos = (self.dpos + 1) % self.delay_frames;

            // Sliding maximum of the detected true peak.
            let det = self.detect(in_l, in_r);
            if det > self.peak_in {
                self.peak_in = det;
            }
            while let Some(&(v, _)) = self.dq.back() {
                if v <= det {
                    self.dq.pop_back();
                } else {
                    break;
                }
            }
            self.dq.push_back((det, self.idx));
            let oldest = self.idx.saturating_sub(self.look as u64);
            while let Some(&(_, j)) = self.dq.front() {
                if j < oldest {
                    self.dq.pop_front();
                } else {
                    break;
                }
            }
            let env = self.dq.front().map(|&(v, _)| v).unwrap_or(0.0);

            // The gain that envelope needs, then the moving average of it.
            let gr = if env > self.ceiling {
                self.ceiling / env
            } else {
                1.0
            };
            self.gr_sum += gr - self.gr_ring[self.gr_pos];
            self.gr_ring[self.gr_pos] = gr;
            self.gr_pos = (self.gr_pos + 1) % self.gr_ring.len();
            let ma = self.gr_sum / self.gr_ring.len() as f64;

            // Program-dependent release, as two stages.
            //
            // A single release time has to choose. Long, and the level stays
            // down for a third of a second after every transient, which is
            // pumping. Short, and on sustained loud material the gain
            // re-attacks on every peak and lets go milliseconds later, which
            // is reshaping the waveform rather than limiting it.
            //
            // The slow stage moves at its own pace in both directions, so a
            // brief transient barely touches it and a sustained loud passage
            // settles it. The fast stage then handles only what is left over
            // -- the transient part, which is brief, so it can let go quickly
            // without the gain moving audibly. As the slow stage catches up on
            // a sustained passage, `req` climbs back towards 1 and the fast
            // stage stops doing anything.
            //
            // What this does *not* do is stop the gain dipping at every peak
            // over the ceiling; nothing can, and that is the attack, not the
            // release. It makes the depth the fast stage works over small.
            //
            // Safety is unaffected: the fast stage is clamped to `req` on the
            // way down, so `g_fast <= ma / g_slow` and the product is at most
            // `ma`, whatever the slow stage is doing.
            if self.sustain_atk > 0.0 {
                let c = if ma < self.g_slow {
                    self.sustain_atk
                } else {
                    self.sustain_rel
                };
                self.g_slow += (ma - self.g_slow) * c;
                self.g_slow = self.g_slow.clamp(1e-9, 1.0);
            }
            let req = (ma / self.g_slow).min(1.0);
            if req < self.g_fast {
                self.g_fast = req;
            } else {
                self.g_fast += (req - self.g_fast) * self.release_coef;
                if self.g_fast > req {
                    self.g_fast = req;
                }
            }
            self.gain = self.g_slow * self.g_fast;
            // The product can only drift above `ma` through rounding; pin it.
            if self.gain > ma {
                self.gain = ma;
            }

            buf[i] = (out_l * self.gain) as f32;
            buf[i + 1] = (out_r * self.gain) as f32;
            self.idx += 1;
        }
    }
}

/// A 4x polyphase interpolator for true-peak detection: a Blackman-windowed
/// sinc, split by phase. BS.1770-4 specifies a longer filter than this; eight
/// taps a phase catches the great majority of inter-sample overshoot at a
/// fraction of the cost, and this is a detector, never in the signal path.
/// Computed rather than tabulated so the window and the cutoff stay visible.
fn design_polyphase() -> [[f64; TP_TAPS]; TP_PHASES] {
    let mut out = [[0.0f64; TP_TAPS]; TP_PHASES];
    let n = (TP_TAPS * TP_PHASES) as f64;
    for (p, phase) in out.iter_mut().enumerate() {
        for (t, c) in phase.iter_mut().enumerate() {
            // Where this tap sits in the prototype, in input samples.
            let x = t as f64 - (TP_TAPS as f64 / 2.0 - 1.0) - p as f64 / TP_PHASES as f64;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let k = (t * TP_PHASES + p) as f64;
            let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * k / (n - 1.0)).cos()
                + 0.08 * (4.0 * std::f64::consts::PI * k / (n - 1.0)).cos();
            *c = sinc * w;
        }
        // Normalise each phase to unity gain. Without this the window pulls
        // every phase down -- phase zero, which should be a plain delay, comes
        // out at 0.81 -- so the detector under-reports the peak it is there to
        // find and the limiter quietly under-corrects.
        let sum: f64 = phase.iter().sum();
        if sum.abs() > 1e-12 {
            for c in phase.iter_mut() {
                *c /= sum;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_signal_passes_through_scaled() {
        let mut lim = Limiter::new(48000);
        // The loudness follower falls with a 1/3-second time constant, so it
        // needs a couple of seconds to reach the 0.4 floor. Once there, quiet
        // material is divided by 0.4 * 2 = 0.8.
        let mut buf = vec![0.1f32; 48000 * 6];
        lim.process(&mut buf);
        for &v in &buf[buf.len() - 1000..] {
            assert!((v - 0.125).abs() < 0.005, "got {v}");
        }
    }

    #[test]
    fn loud_signal_is_pulled_down() {
        let mut lim = Limiter::new(48000);
        let mut buf = vec![8.0f32; 48000 * 2];
        lim.process(&mut buf);
        let tail = &buf[buf.len() - 100..];
        for &v in tail {
            assert!(v.abs() <= 1.0, "limiter left {v} above full scale");
        }
    }

    #[test]
    fn is_deterministic_across_identical_input() {
        let make = || {
            let mut lim = Limiter::new(48000);
            let mut buf: Vec<f32> = (0..8192)
                .map(|i| ((i as f32) * 0.01).sin() * 3.0)
                .collect();
            lim.process(&mut buf);
            buf
        };
        assert_eq!(make(), make());
    }

    /// The whole point of a brickwall: whatever goes in, nothing comes out
    /// above the ceiling. If this ever fails, `clamp_block` starts hard-clipping
    /// and hard clipping is what a click is.
    #[test]
    fn brickwall_never_exceeds_the_ceiling() {
        for ceiling in [1.0f64, 0.5] {
            let mut bw = Brickwall::new(48000, ceiling, 2.0, 60.0, 400.0, true);
            // Loud sustained tone with violent transients on top, of the kind
            // a saturated black MIDI mix produces.
            let mut buf: Vec<f32> = (0..48000 * 2)
                .map(|i| {
                    let t = i / 2;
                    let base = ((t as f64) * 0.05).sin() * 40.0;
                    let spike = if t % 977 == 0 { 300.0 } else { 0.0 };
                    (base + spike) as f32
                })
                .collect();
            bw.process(&mut buf);
            let peak = buf.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            assert!(
                peak as f64 <= ceiling + 1e-4,
                "brickwall at ceiling {ceiling} let {peak} through"
            );
            assert_eq!(
                clamp_block(&mut buf),
                0,
                "brickwall output still needed hard clipping at ceiling {ceiling}"
            );
        }
    }

    /// The reason it exists. A brief transient must not pull down the material
    /// after it: that is the pumping the follower above is prone to, because it
    /// only sees a peak once the peak has gone past and needs a third of a
    /// second to recover.
    #[test]
    fn brickwall_recovers_quickly_after_a_transient() {
        let quiet = 0.5f32;
        let make = |limit: bool| {
            let mut buf: Vec<f32> = vec![0.0; 48000 * 2];
            for (i, v) in buf.iter_mut().enumerate() {
                let t = i / 2;
                // One 1 ms burst at 0.1 s, quiet steady tone either side.
                *v = if (4800..4848).contains(&t) { 60.0 } else { quiet };
            }
            if limit {
                Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
            }
            buf
        };
        let out = make(true);
        let level = |sec: f64| {
            let a = (sec * 48000.0) as usize * 2;
            let b = a + 4800;
            out[a..b].iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        };
        // Well before the burst the signal is under the ceiling and untouched.
        assert!(
            (level(0.02) - quiet).abs() < 0.02,
            "quiet material before the transient was attenuated: {}",
            level(0.02)
        );
        // 250 ms after it, the follower above would still be recovering. This
        // one has to be back.
        assert!(
            (level(0.35) - quiet).abs() < 0.02,
            "still ducking 250 ms after a 1 ms transient: {}",
            level(0.35)
        );
    }

    /// True-peak detection has to catch overshoot that sample-peak detection
    /// misses, or the name means nothing.
    #[test]
    fn true_peak_detection_catches_intersample_overshoot() {
        // A half-Nyquist tone whose samples sit exactly at full scale.
        let make = || -> Vec<f32> {
            (0..48000 * 2)
                .map(|i| {
                    let t = (i / 2) as f64;
                    // Samples land on +/-1.0 while the waveform between them
                    // reaches sqrt(2). A sample-peak limiter sees nothing to
                    // do; a true-peak one has 3 dB to take off.
                    ((t * std::f64::consts::PI / 2.0 + std::f64::consts::PI / 4.0).sin()
                        * std::f64::consts::SQRT_2) as f32
                })
                .collect()
        };
        let mut tp = make();
        let mut sp = make();
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut tp);
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, false).process(&mut sp);
        let peak = |b: &[f32]| b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            peak(&tp) < peak(&sp) - 0.05,
            "true-peak mode did not pull further than sample-peak mode: {} against {}",
            peak(&tp),
            peak(&sp)
        );
    }

    #[test]
    fn brickwall_is_deterministic() {
        let run = || {
            let mut buf: Vec<f32> = (0..8192)
                .map(|i| ((i as f32) * 0.017).sin() * 9.0)
                .collect();
            Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
            buf
        };
        assert_eq!(run(), run());
    }

    /// The point of the sustained stage.
    ///
    /// A short release is what keeps the material after a transient at full
    /// level, but on *sustained* loud material it makes the gain re-attack on
    /// every peak and let go again milliseconds later. A gain moving that fast
    /// is not limiting the waveform, it is reshaping it. The slow stage is
    /// supposed to absorb the sustained part of the reduction so the fast one
    /// idles near unity and the gain holds still.
    ///
    /// Measured as the steadiness of the gain the limiter actually applied,
    /// recovered by dividing output by input, which is the direct statement of
    /// the property and does not conflate the source's own dynamics with it.
    #[test]
    fn the_sustained_stage_holds_the_gain_steady_on_a_loud_passage() {
        let wobble = |sustain_ms: f64| -> f64 {
            // Two tones a 50 Hz beat apart, far above the ceiling. Nothing
            // here is a transient: all of the reduction is sustained.
            let src: Vec<f32> = (0..48000 * 2 * 2)
                .map(|i| {
                    let t = (i / 2) as f64 / 48000.0;
                    let a = (t * 1000.0 * std::f64::consts::TAU).sin();
                    let b = (t * 1050.0 * std::f64::consts::TAU).sin();
                    ((a + b) * 20.0) as f32
                })
                .collect();
            let mut buf = src.clone();
            // Deliberately short fast release: the setting that misbehaves.
            let mut bw = Brickwall::new(48000, 1.0, 2.0, 5.0, sustain_ms, true);
            let d = bw.latency() * 2;
            bw.process(&mut buf);
            // Recover the applied gain, over the back half so the slow stage
            // has settled, and only where the input is big enough that the
            // division is not dominated by rounding near a zero crossing.
            let start = src.len() / 2;
            let g: Vec<f64> = (start..src.len() - d)
                .filter(|&k| src[k].abs() > 4.0)
                .map(|k| (buf[k + d] as f64) / (src[k] as f64))
                .collect();
            assert!(g.len() > 10000, "not enough usable samples at {sustain_ms}");
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            let var = g.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / g.len() as f64;
            var.sqrt() / mean
        };

        let single = wobble(0.0);
        let staged = wobble(400.0);
        // A fifth steadier, not an order of magnitude: the gain still has to
        // dip at every peak over the ceiling, and that is the attack. What the
        // stage removes is the depth the fast release works over.
        assert!(
            staged < single * 0.85,
            "the sustained stage did not steady the gain: it wobbles by              {staged:.4} with the stage and {single:.4} without"
        );
    }

    /// And it must not have cost the thing the fast release was for: a brief
    /// transient still has to let go quickly.
    #[test]
    fn the_sustained_stage_does_not_reintroduce_pumping() {
        let quiet = 0.5f32;
        let mut buf: Vec<f32> = vec![0.0; 48000 * 2];
        for (i, v) in buf.iter_mut().enumerate() {
            let t = i / 2;
            *v = if (4800..4848).contains(&t) { 60.0 } else { quiet };
        }
        Brickwall::new(48000, 1.0, 2.0, 60.0, 400.0, true).process(&mut buf);
        let level = |sec: f64| {
            let a = (sec * 48000.0) as usize * 2;
            buf[a..a + 4800].iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        };
        assert!(
            (level(0.35) - quiet).abs() < 0.02,
            "still ducking 250 ms after a 1 ms transient with the sustained \
             stage on: {}",
            level(0.35)
        );
    }
}
