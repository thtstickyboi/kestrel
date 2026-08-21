//! 32.32 fixed-point phase arithmetic.
//!
//! Not f32, which loses precision on long samples, and not f64, which costs
//! throughput on consumer NVIDIA parts. The GPU stores the same value as two
//! u32 lanes in separate SoA arrays; these helpers keep the host in sync.

/// A 32.32 fixed-point number. The upper 32 bits are a sample index, the lower
/// 32 bits are the fraction of the way to the next sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Fixed(pub u64);

pub const FRAC_BITS: u32 = 32;
pub const FRAC_ONE: u64 = 1u64 << FRAC_BITS;
/// Exactly the constant the shader multiplies by, so both sides round the same.
pub const FRAC_SCALE_F32: f32 = 1.0 / 4294967296.0;

impl Fixed {
    #[inline]
    pub const fn from_parts(hi: u32, lo: u32) -> Self {
        Fixed(((hi as u64) << 32) | lo as u64)
    }

    /// Convert a positive ratio to 32.32. Computed in f64, then frozen to
    /// fixed point so the device never sees an f64.
    #[inline]
    pub fn from_f64(v: f64) -> Self {
        debug_assert!(v >= 0.0, "phase step must be non-negative");
        let scaled = v * FRAC_ONE as f64;
        // Saturate rather than wrap: an absurd pitch ratio should render as a
        // very fast sweep, not as silence from a wrapped step.
        if scaled.is_nan() || scaled <= 0.0 {
            Fixed(0)
        } else if scaled >= u64::MAX as f64 {
            Fixed(u64::MAX)
        } else {
            Fixed(scaled as u64)
        }
    }

    #[inline]
    pub const fn from_int(i: u32) -> Self {
        Fixed((i as u64) << 32)
    }

    #[inline]
    pub const fn hi(self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline]
    pub const fn lo(self) -> u32 {
        self.0 as u32
    }

    /// Fractional part as an f32 in [0, 1). This is exactly what the shader
    /// computes with `f32(lo) * (1.0 / 4294967296.0)`.
    #[inline]
    pub fn frac_f32(self) -> f32 {
        self.lo() as f32 * FRAC_SCALE_F32
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / FRAC_ONE as f64
    }

    #[inline]
    pub fn wrapping_add(self, rhs: Fixed) -> Fixed {
        Fixed(self.0.wrapping_add(rhs.0))
    }

    /// Scale by an 8.24 fixed-point factor, truncating, saturating rather than
    /// wrapping.
    ///
    /// This is how pitch bend reaches a voice: the spawned step is the note's
    /// unbent ratio and the effective step is this product, recomputed as the
    /// bend moves. It has to be exact integer arithmetic rather than a float
    /// multiply, because the device computes the same product in 32-bit limbs
    /// and the two must agree bit for bit. A step that differs in its lowest
    /// bits does not show up as an error in the null test, it shows up as the
    /// two renders slowly drifting out of phase with each other.
    #[inline]
    pub fn scale(self, factor: u32) -> Fixed {
        let p = ((self.0 as u128) * (factor as u128)) >> BEND_FRAC_BITS;
        if p > u64::MAX as u128 {
            Fixed(u64::MAX)
        } else {
            Fixed(p as u64)
        }
    }
}

/// Pitch bend factors are 8.24 fixed point: a multiplier in [0, 256) with 24
/// fractional bits, which is about a ten-thousandth of a cent of resolution
/// and enough headroom for the widest bend range RPN 0 can ask for.
pub const BEND_FRAC_BITS: u32 = 24;
pub const BEND_ONE: u32 = 1 << BEND_FRAC_BITS;

/// The 8.24 factor for a bend of `semitones`, which is what the host freezes
/// into the bend table so the device never computes a power.
pub fn bend_factor(semitones: f64) -> u32 {
    if !semitones.is_finite() {
        return BEND_ONE;
    }
    let f = (semitones / 12.0).exp2() * BEND_ONE as f64;
    if f <= 0.0 {
        1
    } else if f >= u32::MAX as f64 {
        u32::MAX
    } else {
        f.round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_unity() {
        let f = Fixed::from_f64(1.0);
        assert_eq!(f.hi(), 1);
        assert_eq!(f.lo(), 0);
        assert_eq!(f.frac_f32(), 0.0);
    }

    #[test]
    fn accumulates_without_drift() {
        // 44100 -> 48000 is the worst common ratio. The step is exact and the
        // accumulation is linear, so the total after N frames is checkable in
        // closed form without actually stepping a billion times.
        let ratio = 44100.0f64 / 48000.0;
        let step = Fixed::from_f64(ratio);

        let mut p = Fixed::default();
        for _ in 0..1000 {
            p = p.wrapping_add(step);
        }
        assert!((p.to_f64() - ratio * 1000.0).abs() < 1e-6);

        let n = 1_000_000_000u64;
        let total = (step.0 as u128) * n as u128;
        let err = (total as f64 / FRAC_ONE as f64) - ratio * n as f64;
        assert!(err.abs() < 1.0, "drift over 1e9 frames was {err} samples");
    }

    #[test]
    fn frac_matches_shader_expression() {
        for lo in [0u32, 1, 0x4000_0000, 0x8000_0000, 0xFFFF_FFFF] {
            let f = Fixed::from_parts(3, lo);
            assert_eq!(f.frac_f32(), lo as f32 * (1.0 / 4294967296.0));
        }
    }

    #[test]
    fn f32_phase_would_have_failed() {
        // Motivation check for the whole module: an f32 accumulator loses the
        // ability to represent a fractional step past 2^24 samples.
        let mut p: f32 = 0.0;
        let step: f32 = 44100.0 / 48000.0;
        for _ in 0..(1 << 24) {
            p += step;
        }
        let exact = (1u64 << 24) as f64 * (44100.0 / 48000.0);
        assert!((p as f64 - exact).abs() > 100.0);
    }
}
