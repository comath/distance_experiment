use packed_simd::*;
use std::cmp::{max, min};

pub trait Metric: Send + Sync {
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32;
    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32;
    fn norm(x: &[f32]) -> f32;
}

pub struct L2 {}

impl Metric for L2 {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            let y_simd = f32x16::from_slice_unaligned(y);
            let diff = x_simd - y_simd;
            d_acc_16 += diff * diff;
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            let y_simd = f32x8::from_slice_unaligned(y);
            let diff = x_simd - y_simd;
            d_acc_8 += diff * diff;
            y = &y[8..];
            x = &x[8..];
        }
        let leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| (xi - yi) * (xi - yi))
            .fold(0.0, |acc, y| acc + y);
        (leftover + d_acc_8.sum() + d_acc_16.sum()).sqrt()
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while x.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 += x_simd * x_simd;
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if x.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 += x_simd * x_simd;
            x = &x[8..];
        }
        let leftover = x.iter().map(|xi| xi * xi).fold(0.0, |acc, xi| acc + xi);
        (leftover + d_acc_8.sum() + d_acc_16.sum()).sqrt()
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.len() == 0 || y_val.len() == 0 {
            if x_val.len() == 0 && y_val.len() == 0 {
                return 0.0;
            }
            if x_val.len() > 0 && y_val.len() == 0 {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut total = 0.0;
            let mut long_iter;
            let short_iter;
            if x_ind.len() > y_ind.len() {
                long_iter = x_ind.iter().zip(x_val);
                short_iter = y_ind.iter().zip(y_val);
            } else {
                long_iter = y_ind.iter().zip(y_val);
                short_iter = x_ind.iter().zip(x_val);
            }

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        total += lv * lv;
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        total += val * val;
                        l_tr = long_iter.next();
                    } else {
                        total += sv * sv;
                    }
                } else {
                    total += sv * sv;
                }
            }
            while let Some((_li, lv)) = l_tr {
                total += lv * lv;
                l_tr = long_iter.next();
            }
            total.sqrt()
        }
    }
}

pub struct Linfty {}

impl Metric for Linfty {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            let diff = (x_simd - y_simd).abs();
            d_acc_16 = d_acc_16.max(diff);
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            let diff = (x_simd - y_simd).abs();
            d_acc_8 = d_acc_8.max(diff);
            y = &y[8..];
            x = &x[8..];
        }
        let leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| (xi - yi).abs())
            .fold(0.0, |acc: f32, y| acc.max(y));
        leftover.max(d_acc_8.max_element().max(d_acc_16.max_element()))
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while x.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 = d_acc_16.max(x_simd);
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if x.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 = d_acc_8.max(x_simd);
            x = &x[8..];
        }
        let leftover = x.iter().map(|xi| xi.abs()).fold(0.0, |acc:f32, xi| acc.max(xi));
        leftover.max(d_acc_8.max_element().max(d_acc_16.max_element()))
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.len() == 0 || y_val.len() == 0 {
            if x_val.len() == 0 && y_val.len() == 0 {
                return 0.0;
            }
            if x_val.len() > 0 && y_val.len() == 0 {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut max_val:f32 = 0.0;
            let mut long_iter;
            let short_iter;
            if x_ind.len() > y_ind.len() {
                long_iter = x_ind.iter().zip(x_val);
                short_iter = y_ind.iter().zip(y_val);
            } else {
                long_iter = y_ind.iter().zip(y_val);
                short_iter = x_ind.iter().zip(x_val);
            }

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        max_val = max_val.max(lv * lv);
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        max_val = max_val.max(val * val);
                        l_tr = long_iter.next();
                    } else {
                        max_val = max_val.max(sv * sv);
                    }
                } else {
                    max_val = max_val.max(sv * sv);
                }
            }
            while let Some((_li, lv)) = l_tr {
                max_val = max_val.max(lv * lv);
                l_tr = long_iter.next();
            }
            max_val.sqrt()
        }
    }
}


pub struct L1 {}

impl Metric for L1 {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            let diff = x_simd - y_simd;
            d_acc_16 += diff.abs();
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            let diff = x_simd - y_simd;
            d_acc_8 += diff.abs();
            y = &y[8..];
            x = &x[8..];
        }
        let leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| (xi - yi).abs())
            .fold(0.0, |acc, y| acc + y);
        (leftover + d_acc_8.sum() + d_acc_16.sum())
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while x.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 += x_simd.abs();
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if x.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 += x_simd.abs();
            x = &x[8..];
        }
        let leftover = x.iter().map(|xi| xi.abs()).fold(0.0, |acc, xi| acc+xi);
        leftover + d_acc_8.sum() + d_acc_16.sum()
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.len() == 0 || y_val.len() == 0 {
            if x_val.len() == 0 && y_val.len() == 0 {
                return 0.0;
            }
            if x_val.len() > 0 && y_val.len() == 0 {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut total = 0.0;
            let mut long_iter;
            let short_iter;
            if x_ind.len() > y_ind.len() {
                long_iter = x_ind.iter().zip(x_val);
                short_iter = y_ind.iter().zip(y_val);
            } else {
                long_iter = y_ind.iter().zip(y_val);
                short_iter = x_ind.iter().zip(x_val);
            }

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        total += lv.abs();
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        total += val.abs();
                        l_tr = long_iter.next();
                    } else {
                        total += sv.abs();
                    }
                } else {
                    total += sv.abs();
                }
            }
            while let Some((_li, lv)) = l_tr {
                total += lv.abs();
                l_tr = long_iter.next();
            }
            total
        }
    }
}

pub struct CosineSim {}

impl Metric for CosineSim {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        let mut x_acc_16 = f32x16::splat(0.0);
        let mut y_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 += x_simd * y_simd;
            x_acc_16 += x_simd * x_simd;
            y_acc_16 += y_simd * y_simd;
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        let mut x_acc_8 = f32x8::splat(0.0);
        let mut y_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 += x_simd * y_simd;
            x_acc_8 += x_simd * x_simd;
            y_acc_8 += y_simd * y_simd;
            y = &y[8..];
            x = &x[8..];
        }
        let acc_leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| xi * yi)
            .fold(0.0, |acc, y| acc + y);
        let y_leftover = y.iter().map(|yi| yi * yi).fold(0.0, |acc, yi| acc + yi);
        let x_leftover = x.iter().map(|xi| xi * xi).fold(0.0, |acc, xi| acc + xi);
        let acc = acc_leftover + d_acc_8.sum() + d_acc_16.sum();
        let xnm = (x_leftover + x_acc_8.sum() + x_acc_16.sum()).sqrt();
        let ynm = (y_leftover + y_acc_8.sum() + y_acc_16.sum()).sqrt();
        acc / (xnm * ynm).max(0.00001)
    }

    fn norm(mut x: &[f32]) -> f32 {
        0.0
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.len() == 0 || y_val.len() == 0 {
            0.0
        } else {
            let mut dotprod = 0.0;
            let mut long_iter;
            let short_iter;
            if x_ind.len() > y_ind.len() {
                long_iter = x_ind.iter().zip(x_val);
                short_iter = y_ind.iter().zip(y_val);
            } else {
                long_iter = y_ind.iter().zip(y_val);
                short_iter = x_ind.iter().zip(x_val);
            }

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    l_tr = long_iter.next();
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        dotprod += sv * lv;
                        l_tr = long_iter.next();
                    }
                }
            }
            let xnm = L2::norm(x_val);
            let ynm = L2::norm(y_val);
            dotprod / (xnm * ynm).max(0.00001)
        }
    }
}