//! Per-shape image features targeted at the reject criteria reviewers use:
//! content cutoff, surrounding interference, and text-flow alignment.
//!
//! For each shape's axis-aligned bounding box (in image pixel space):
//!
//! - `cutoff`       — mean gradient magnitude in a 2 px band along the
//!   *interior* edges of the box, normalised by the image-wide gradient
//!   baseline. High when content (typically text) is crossing the
//!   boundary — i.e. the box is clipping content.
//! - `isolation`    — mean gradient magnitude in a 6 px ring *just outside*
//!   the box, same normalisation. High when surroundings are noisy. This
//!   is the user's "OK to cut content if the alternative is including too
//!   much outside interference" exception.
//! - `misalign`     — angle between the dominant text-line orientation
//!   inside the box (estimated via the structure tensor of the gradient
//!   field) and the shape's `direction`. `0` = aligned, `1` = perpendicular.
//! - `ink_contrast` — `(mean_outside_ring − mean_inside) / 255`. Cheap
//!   sanity feature distinguishing "actual content inside a clean
//!   background" from "empty box on noisy background" and similar.

use crate::annotation::Shape;
use anyhow::{Context, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::path::Path;

/// Width (px) of the band along the interior edge used for `cutoff`.
const INTERIOR_BAND_PX: u32 = 2;

/// Width (px) of the ring just outside the box used for `isolation`.
const EXTERIOR_RING_PX: u32 = 6;

/// Minimum value used as the divisor when normalising gradient means. Keeps
/// images with almost no edges (mostly-blank pages) from blowing up.
const GRAD_NORM_FLOOR: f64 = 8.0;

type GradImg = ImageBuffer<Luma<i16>, Vec<i16>>;

/// Half-open pixel rectangle `[x0, x1) × [y0, y1)`. Shared by the inner-box
/// and exterior-ring computations to keep coordinate quartets out of long
/// argument lists.
#[derive(Debug, Clone, Copy)]
struct Rect {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl Rect {
    #[inline]
    fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

#[derive(Debug, Clone, Default)]
pub struct ShapeImageFeats {
    pub cutoff: f64,
    pub isolation: f64,
    pub misalign: f64,
    pub ink_contrast: f64,
}

impl ShapeImageFeats {
    pub fn to_array(&self) -> [f64; 4] {
        [self.cutoff, self.isolation, self.misalign, self.ink_contrast]
    }
}

pub fn extract_for_file(jpg: &Path, shapes: &[Shape]) -> Result<Vec<ShapeImageFeats>> {
    let img = image::open(jpg).with_context(|| format!("opening JPEG {:?}", jpg))?;
    let gray = img.to_luma8();
    let gx = imageproc::gradients::horizontal_sobel(&gray);
    let gy = imageproc::gradients::vertical_sobel(&gray);

    // Pre-compute per-pixel gradient magnitude into a flat buffer.
    // This avoids redundant sqrt calls when multiple features scan
    // overlapping pixel regions (cutoff border, isolation ring, etc.).
    let (img_w, img_h) = (gray.width(), gray.height());
    let grad_mag = build_gradient_magnitude(&gx, &gy, img_w, img_h);

    // Whole-image mean gradient is the per-image normalisation baseline.
    let img_grad_mean = gradient_mean_buf(&grad_mag, img_w, Rect { x0: 0, y0: 0, x1: img_w, y1: img_h });
    let img_grad_norm = img_grad_mean.max(GRAD_NORM_FLOOR);

    let ctx = ExtractCtx {
        gray: &gray,
        grad_mag: &grad_mag,
        gx: &gx,
        gy: &gy,
        img_w,
        img_h,
        img_grad_norm,
    };

    let mut out = Vec::with_capacity(shapes.len());
    for shape in shapes {
        out.push(extract_one(shape, &ctx));
    }
    Ok(out)
}

/// Build a flat `Vec<f64>` of per-pixel gradient magnitudes.
/// Layout: row-major, index = y * width + x.
fn build_gradient_magnitude(gx: &GradImg, gy: &GradImg, w: u32, h: u32) -> Vec<f64> {
    let mut buf = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let gxv = gx.get_pixel(x, y)[0] as f64;
            let gyv = gy.get_pixel(x, y)[0] as f64;
            buf.push((gxv * gxv + gyv * gyv).sqrt());
        }
    }
    buf
}

/// Mean gradient magnitude over a rectangular region, reading from the
/// pre-computed flat buffer.
fn gradient_mean_buf(buf: &[f64], img_w: u32, r: Rect) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut n = 0u64;
    for y in r.y0..r.y1 {
        let row_start = (y * img_w) as usize;
        for x in r.x0..r.x1 {
            sum += buf[row_start + x as usize];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Mean gradient magnitude in `outer` minus the pixels inside `inner`,
/// reading from the pre-computed flat buffer.
fn ring_gradient_mean_buf(buf: &[f64], img_w: u32, outer: Rect, inner: Rect) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    for y in outer.y0..outer.y1 {
        let row_start = (y * img_w) as usize;
        for x in outer.x0..outer.x1 {
            if x >= inner.x0 && x < inner.x1 && y >= inner.y0 && y < inner.y1 {
                continue;
            }
            sum += buf[row_start + x as usize];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Border-band mean from the pre-computed flat buffer.
fn border_band_gradient_mean_buf(buf: &[f64], img_w: u32, r: Rect, band: u32) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    let top_y1 = (r.y0 + band).min(r.y1);
    for y in r.y0..top_y1 {
        let row_start = (y * img_w) as usize;
        for x in r.x0..r.x1 {
            sum += buf[row_start + x as usize];
            n += 1;
        }
    }
    let bot_y0 = r.y1.saturating_sub(band).max(top_y1);
    for y in bot_y0..r.y1 {
        let row_start = (y * img_w) as usize;
        for x in r.x0..r.x1 {
            sum += buf[row_start + x as usize];
            n += 1;
        }
    }
    let left_x1 = (r.x0 + band).min(r.x1);
    for x in r.x0..left_x1 {
        for y in top_y1..bot_y0 {
            sum += buf[(y * img_w + x) as usize];
            n += 1;
        }
    }
    let right_x0 = r.x1.saturating_sub(band).max(left_x1);
    for x in right_x0..r.x1 {
        for y in top_y1..bot_y0 {
            sum += buf[(y * img_w + x) as usize];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

/// Context for feature extraction — shared per-image to avoid passing 8+ args.
struct ExtractCtx<'a> {
    gray: &'a GrayImage,
    grad_mag: &'a [f64],
    gx: &'a GradImg,
    gy: &'a GradImg,
    img_w: u32,
    img_h: u32,
    img_grad_norm: f64,
}

fn extract_one(shape: &Shape, ctx: &ExtractCtx<'_>) -> ShapeImageFeats {
    let (x0_i, y0_i, w, h) = axis_aligned_bbox(&shape.points);
    let inner = Rect {
        x0: x0_i.max(0) as u32,
        y0: y0_i.max(0) as u32,
        x1: ((x0_i + w as i32).max(0) as u32).min(ctx.img_w),
        y1: ((y0_i + h as i32).max(0) as u32).min(ctx.img_h),
    };
    if inner.is_empty() {
        return ShapeImageFeats::default();
    }

    // cutoff — interior border band.
    let band = INTERIOR_BAND_PX.clamp(1, ((inner.x1 - inner.x0).min(inner.y1 - inner.y0) / 2).max(1));
    let border_mean = border_band_gradient_mean_buf(ctx.grad_mag, ctx.img_w, inner, band);
    // Saturate at 3× the image-wide baseline → scale to [0, 1].
    let cutoff = (border_mean / ctx.img_grad_norm).min(3.0) / 3.0;

    // isolation — exterior ring around the inner box.
    let r = EXTERIOR_RING_PX;
    let outer = Rect {
        x0: inner.x0.saturating_sub(r),
        y0: inner.y0.saturating_sub(r),
        x1: (inner.x1 + r).min(ctx.img_w),
        y1: (inner.y1 + r).min(ctx.img_h),
    };
    let ring_mean = ring_gradient_mean_buf(ctx.grad_mag, ctx.img_w, outer, inner);
    let isolation = (ring_mean / ctx.img_grad_norm).min(3.0) / 3.0;

    // misalign — structure-tensor estimate vs box direction.
    let misalign = compute_misalign(ctx.gx, ctx.gy, inner, shape.direction.unwrap_or(0.0));

    // ink_contrast.
    let inside_mean = intensity_mean(ctx.gray, inner);
    let outside_mean = ring_intensity_mean(ctx.gray, outer, inner);
    let ink_contrast = ((outside_mean - inside_mean) / 255.0).clamp(-1.0, 1.0);

    ShapeImageFeats {
        cutoff,
        isolation,
        misalign,
        ink_contrast,
    }
}

fn intensity_mean(gray: &GrayImage, r: Rect) -> f64 {
    if r.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut n = 0u64;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            sum += gray.get_pixel(x, y)[0] as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Mean intensity in `outer` minus the pixels inside `inner`.
fn ring_intensity_mean(gray: &GrayImage, outer: Rect, inner: Rect) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    for y in outer.y0..outer.y1 {
        for x in outer.x0..outer.x1 {
            if x >= inner.x0 && x < inner.x1 && y >= inner.y0 && y < inner.y1 {
                continue;
            }
            sum += gray.get_pixel(x, y)[0] as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn compute_misalign(gx: &GradImg, gy: &GradImg, r: Rect, box_direction: f64) -> f64 {
    // Structure tensor over the box interior.
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    let mut samples = 0u64;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let gxv = gx.get_pixel(x, y)[0] as f64;
            let gyv = gy.get_pixel(x, y)[0] as f64;
            sxx += gxv * gxv;
            syy += gyv * gyv;
            sxy += gxv * gyv;
            samples += 1;
        }
    }
    if samples == 0 || (sxx + syy) < 1.0 {
        return 0.0;
    }
    // Dominant gradient orientation. Gradient is perpendicular to text rows,
    // so text orientation = grad orientation + π/2.
    let theta_grad = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let theta_text = theta_grad + std::f64::consts::FRAC_PI_2;
    angular_distance_normalized(theta_text, box_direction)
}

/// Returns the angular distance between two orientations in `[0, 1]` where
/// `0` = parallel, `1` = perpendicular. Orientation is mod π (lines, not
/// vectors).
fn angular_distance_normalized(a: f64, b: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let mut d = (a - b).abs() % pi;
    if d > pi * 0.5 {
        d = pi - d;
    }
    d / (pi * 0.5)
}

fn axis_aligned_bbox(points: &[(f64, f64)]) -> (i32, i32, f64, f64) {
    if points.is_empty() {
        return (0, 0, 0.0, 0.0);
    }
    let (min_x, max_x) = points
        .iter()
        .map(|p| p.0)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), x| {
            (mn.min(x), mx.max(x))
        });
    let (min_y, max_y) = points
        .iter()
        .map(|p| p.1)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), y| {
            (mn.min(y), mx.max(y))
        });
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);
    (min_x as i32, min_y as i32, w, h)
}

/// Aggregate packed per-shape feature arrays (4 dims each) into the 6-dim
/// vector the LR consumes:
/// `[cutoff_mean, cutoff_max, isolation_mean, misalign_mean, misalign_max, ink_contrast_mean]`.
/// `*_max` captures the "worst shape" — a single badly clipped or misaligned
/// box is often the reject signal even when most shapes look fine.
pub fn aggregate_arrays(per_shape: &[[f64; 4]]) -> [f64; 6] {
    if per_shape.is_empty() {
        return [0.0; 6];
    }
    let n = per_shape.len() as f64;
    let mut sum_cutoff = 0.0;
    let mut max_cutoff = f64::NEG_INFINITY;
    let mut sum_iso = 0.0;
    let mut sum_mis = 0.0;
    let mut max_mis = f64::NEG_INFINITY;
    let mut sum_ink = 0.0;
    for a in per_shape {
        sum_cutoff += a[0];
        if a[0] > max_cutoff {
            max_cutoff = a[0];
        }
        sum_iso += a[1];
        sum_mis += a[2];
        if a[2] > max_mis {
            max_mis = a[2];
        }
        sum_ink += a[3];
    }
    [
        sum_cutoff / n,
        max_cutoff,
        sum_iso / n,
        sum_mis / n,
        max_mis,
        sum_ink / n,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_empty() {
        let r = aggregate_arrays(&[]);
        assert_eq!(r, [0.0; 6]);
    }

    #[test]
    fn test_aggregate_mean_and_max() {
        let r = aggregate_arrays(&[[0.2, 0.5, 0.1, 0.3], [0.8, 0.3, 0.9, 0.5]]);
        // cutoff_mean=0.5, cutoff_max=0.8, iso_mean=0.4, mis_mean=0.5, mis_max=0.9, ink_mean=0.4
        assert!((r[0] - 0.5).abs() < 1e-9);
        assert!((r[1] - 0.8).abs() < 1e-9);
        assert!((r[2] - 0.4).abs() < 1e-9);
        assert!((r[3] - 0.5).abs() < 1e-9);
        assert!((r[4] - 0.9).abs() < 1e-9);
        assert!((r[5] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_angular_distance() {
        let pi = std::f64::consts::PI;
        assert!(angular_distance_normalized(0.0, 0.0) < 1e-9);
        assert!((angular_distance_normalized(pi * 0.5, 0.0) - 1.0).abs() < 1e-9);
        assert!((angular_distance_normalized(pi, 0.0) - 0.0).abs() < 1e-9);
        assert!((angular_distance_normalized(pi * 0.25, 0.0) - 0.5).abs() < 1e-9);
    }
}
