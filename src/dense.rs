use packed_simd::*;
use rand::Rng;
use rayon::prelude::*;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::{max,min};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct PointCloud<F: Fn(&[f32], &[f32]) -> f32 + std::marker::Sync + std::marker::Sync> {
    dim: usize,
    data: Vec<f32>,
    dist_fn: F,
    chunk: usize,
}

// To bypass the borrow checker and do bad things
struct MyBox {
    p: *mut f32,
}
unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}

const DIM: usize = 100*4;
const COUNT: usize = 1000;

#[inline]
pub fn sumsq_dense_fixed64(x: &[f32;64], y: &[f32;64]) -> f32 {
    let leftover = y
        .iter()
        .zip(x.iter())
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .fold(0.0, |acc, y| acc + y);
    leftover
}

#[inline]
pub fn sumsq_dense_fixed32(x: &[f32;32], y: &[f32;32]) -> f32 {
    let leftover = y
        .iter()
        .zip(x.iter())
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .fold(0.0, |acc, y| acc + y);
    leftover
}

#[inline]
pub fn sumsq_dense(x: &[f32], y: &[f32]) -> f32 {
    let leftover = y
        .iter()
        .zip(x)
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .fold(0.0, |acc, y| acc + y);
    leftover
}

#[inline]
pub fn l2_dense(mut x: &[f32], mut y: &[f32]) -> f32 {
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
pub fn l1_dense(mut x: &[f32], mut y: &[f32]) -> f32 {
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
pub fn linfty_dense(mut x: &[f32], mut y: &[f32]) -> f32 {
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
        d_acc_8 += d_acc_8.max(diff);
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
fn cosine_dense(mut x: &[f32], mut y: &[f32]) -> f32 {
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
    (acc.cos()) / (xnm * ynm)
}

impl<F: Fn(&[f32], &[f32]) -> f32 + std::marker::Sync + std::marker::Sync> PointCloud<F> {
    pub fn new_random(dim: usize, count: usize, dist_fn: F) -> PointCloud<F> {
        let mut rng = rand::thread_rng();
        let data = (0..(dim * count)).map(|_i| rng.gen::<f32>()).collect();
        let chunk = max(15000 / dim, 20);
        PointCloud {
            data,
            dim,
            dist_fn,
            chunk,
        }
    }

    pub fn new_zeros(dim: usize, count: usize, dist_fn: F) -> PointCloud<F> {
        let data = vec![0.0; dim * count];
        let chunk = max(15000 / dim, 20);
        PointCloud {
            data,
            dim,
            dist_fn,
            chunk,
        }
    }

    pub fn len(&self) -> usize {
        assert!(self.data.len() % self.dim == 0);
        self.data.len() / self.dim
    }

    pub fn get(&self, i: usize) -> Result<&[f32], &str> {
        Ok(&self.data[(i * self.dim)..((i + 1) * self.dim)])
    }

    pub fn dists(&self, x: &[f32], indexes: &[usize]) -> Result<Vec<f32>, &str> {
        let len = indexes.len();
        if len > self.chunk * 3 {
            let mut dists: Vec<f32> = Vec::with_capacity(len);
            let dists_ptr1: MyBox = MyBox {
                p: dists.as_mut_ptr(),
            };
            let error: Arc<Mutex<Result<(), &str>>> = Arc::new(Mutex::new(Ok(())));
            rayon::scope(|s| {
                let mut start = 0;
                while start + self.chunk * 2 < len {
                    let range = start..(start + self.chunk);
                    s.spawn(|_| unsafe {
                        for i in range {
                            match self.get(indexes[i]) {
                                Ok(y) => *dists_ptr1.p.add(i) = (self.dist_fn)(x, y),
                                Err(e) => {
                                    *dists_ptr1.p.add(i) = 0.0;
                                    *error.lock().unwrap() = Err(e);
                                }
                            }
                        }
                    });
                    start += self.chunk;
                }
                let range = start..len;
                s.spawn(|_| unsafe {
                    for i in range {
                        match self.get(indexes[i]) {
                            Ok(y) => *dists_ptr1.p.add(i) = (self.dist_fn)(x, y),
                            Err(e) => {
                                *dists_ptr1.p.add(i) = 0.0;
                                *error.lock().unwrap() = Err(e);
                            }
                        }
                    }
                });
            });
            unsafe {
                dists.set_len(len);
            }
            (*error.lock().unwrap())?;
            Ok(dists)
        } else {
            indexes
                .iter()
                .map(|i| {
                    let y = self.get(*i)?;
                    Ok((self.dist_fn)(x, y))
                })
                .collect()
        }
    }

    pub fn l2_simd(&self, x: &[f32], indexes: &[usize]) -> Vec<f32> {
        indexes
            .iter()
            .map(|i| (self.dist_fn)(x, self.get(*i).unwrap()))
            .collect()
    }

    pub fn l2_rayon_simd(&self, x: &[f32], indexes: &[usize]) -> Vec<f32> {
        indexes
            .into_par_iter()
            .map(|i| (self.dist_fn)(x, self.get(*i).unwrap()))
            .collect()
    }

    pub fn l2_rayon_shortcut(&self, x: &[f32], indexes: &[usize]) -> Vec<f32> {
        let len = self.len();
        if indexes.len() > 500 {
            let mut res = Vec::with_capacity(len);
            indexes
                .into_par_iter()
                .map(|i| (self.dist_fn)(x, self.get(*i).unwrap()))
                .collect_into_vec(&mut res);
            res
        } else {
            indexes
                .iter()
                .map(|i| (self.dist_fn)(x, self.get(*i).unwrap()))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn it_works() {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        let dists = zero_data.l2_simd(&zero_vec[..], &indexes[..COUNT / 2]);

        assert_eq!(dists[0], 0.0);
    }

    #[bench]
    fn bench_l2_simd(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_simd(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_l2_rayon_smid(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_rayon_simd(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_dists(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_l2_rayon_shortcut(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_rayon_shortcut(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_simd(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, linfty_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_simd(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_rayon_smid(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, linfty_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_rayon_simd(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_rayon_custom(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, linfty_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_rayon_shortcut(b: &mut Bencher) {
        let zero_data = PointCloud::new_zeros(DIM, COUNT, linfty_dense);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.l2_rayon_shortcut(&zero_vec[..], &indexes[..COUNT / 2]));
    }
}
