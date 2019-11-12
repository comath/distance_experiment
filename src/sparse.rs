use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use std::cmp::{max, min};
use std::sync::{Arc, Mutex};
use std::u32;
use packed_simd::*;
use std::marker::PhantomData;

const DIM: usize = 1000 * 2;
const DDIM: usize = 1000;
const COUNT: usize = 1000;
const SP_COEF: f32 = 0.1;
const SETSEED: [u8;32] = [1;32];
const VECSEED: [u8;32] = [8;32];

use super::distances::*;

// To bypass the borrow checker and do bad things
struct MyBox {
    p: *mut f32,
}
unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}

pub struct SparsePointRef<'a> {
    ja: &'a [u32],
    a: &'a [f32],
}

pub struct SparsePointVec {
    ja: Vec<u32>,
    a: Vec<f32>,
}

impl SparsePointVec {
    pub fn zero() -> SparsePointVec {
        SparsePointVec {
            ja: Vec::with_capacity(0),
            a: Vec::with_capacity(0),
        }
    }

    pub fn random(dim: usize, sparse_coef: f32) -> SparsePointVec {
        let mut rng: StdRng = rand::SeedableRng::from_seed(VECSEED);
        let non_zero_count = ((dim as f32) * sparse_coef) as usize;
        let a = (0..non_zero_count).map(|_i| rng.gen::<f32>()).collect();
        let mut ja: Vec<u32> = rand::seq::index::sample(&mut rng, dim, non_zero_count)
            .iter()
            .map(|i| i as u32)
            .collect();
        (&mut ja[..]).sort();
        SparsePointVec { ja, a }
    }

    pub fn random_dense(sparse_dim: usize, sparse_coef: f32, dense_dim: usize) -> SparsePointVec {
        let mut rng: StdRng = rand::SeedableRng::from_seed(VECSEED);
        let non_zero_count = ((sparse_dim as f32) * sparse_coef) as usize;
        let mut a: Vec<f32> = (0..non_zero_count).map(|_i| rng.gen::<f32>()).collect();
        let mut ja: Vec<u32> = rand::seq::index::sample(&mut rng, sparse_dim, non_zero_count)
            .iter()
            .map(|i| i as u32)
            .collect();
        (&mut ja[..]).sort();
        a.extend((0..dense_dim).map(|_i| rng.gen::<f32>()));
        ja.extend(sparse_dim as u32..(sparse_dim + dense_dim) as u32);
        SparsePointVec { ja, a }
    }
}

impl<'a> From<SparsePointRef<'a>> for SparsePointVec {
    fn from(spr: SparsePointRef) -> Self {
        let ja = Vec::from(spr.ja);
        let a = Vec::from(spr.a);
        SparsePointVec { ja, a }
    }
}

pub trait SparsePoint {
    fn indexes(&self) -> &[u32];
    fn values(&self) -> &[f32];

    fn dense(&self, dim: usize) -> Vec<f32> {
        match self.indexes().last() {
            Some(i) => assert!((*i as usize) < dim),
            None => return vec![0.0; dim],
        }
        let mut d = Vec::with_capacity(dim);
        for (tr, val) in self.indexes().iter().zip(self.values()) {
            while d.len() < *tr as usize {
                d.push(0.0);
            }
            d.push(*val);
        }
        while d.len() < dim {
            d.push(0.0);
        }
        d
    }
    fn len(&self) -> usize {
        self.indexes().len()
    }

    fn spref<'b, 'c>(&'b self) -> SparsePointRef<'c>
    where
        'b: 'c;
}

impl<'a> SparsePoint for SparsePointRef<'a> {
    fn indexes(&self) -> &[u32] {
        &self.ja
    }

    fn values(&self) -> &[f32] {
        &self.a
    }

    fn spref<'b, 'c>(&'b self) -> SparsePointRef<'c>
    where
        'b: 'c,
    {
        SparsePointRef {
            ja: &self.ja[..],
            a: &self.a[..],
        }
    }
}

impl SparsePoint for SparsePointVec {
    fn indexes(&self) -> &[u32] {
        &self.ja[..]
    }

    fn values(&self) -> &[f32] {
        &self.a[..]
    }

    fn spref<'a, 'b>(&'a self) -> SparsePointRef<'b>
    where
        'a: 'b,
    {
        SparsePointRef {
            ja: &self.ja[..],
            a: &self.a[..],
        }
    }
}

pub fn l2_sparse_simd(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
    // For some degree of accuracy we store the running total in 32 accumluators
    
    if x_val.len() == 0 || y_val.len() == 0 {
        if x_val.len() == 0 && y_val.len() == 0 {
            return 0.0;
        }
        if x_val.len() > 0 && y_val.len() == 0 {
            L2::norm(x_val)
        } else {
            L2::norm(y_val)
        }
    } else {
        let mut total: [f32; 32] = [0.0; 32];
        let mut total_i = 0;
        // The major loop needs to be done as few times as possible, this gets the shortest slice
        let (l_ind, l_val, s_ind, s_val);
        if x_ind.len() > y_ind.len() {
            l_ind = x_ind;
            l_val = x_val;
            s_ind = y_ind;
            s_val = y_val;
        } else {
            s_ind = x_ind;
            s_val = x_val;
            l_ind = y_ind;
            l_val = y_val;
        }

        let mut d_acc_8 = f32x8::splat(0.0);
        let mut s_tr = 0;
        let mut l_tr = 0;
        unsafe {
            // We first loop over the vector with less support as the outer loop branches more than the inner
            while s_ind.len() > s_tr + 8 {
                // This is the whole of the inner loop, 2 conditions with a single branch.
                while l_ind.len() > l_tr + 1 && *l_ind.get_unchecked(l_tr) < *s_ind.get_unchecked(s_tr) {
                    total[total_i] += l_val.get_unchecked(l_tr) * l_val.get_unchecked(l_tr);
                    total_i = (total_i + 1) % 32;
                    l_tr += 1;
                }
                if l_ind.len() > l_tr && *s_ind.get_unchecked(s_tr) == *l_ind.get_unchecked(l_tr) {
                    // We have end up with points whose support overlaps a lot. This shortcuts those with an SIMD instruction
                    // We check far to close as it's more likely to be different far away from the current instruction.
                    if l_ind.len() > l_tr + 7
                        && s_ind.get_unchecked(s_tr + 7) == l_ind.get_unchecked(l_tr + 7)
                        && s_ind.get_unchecked(s_tr + 6) == l_ind.get_unchecked(l_tr + 6)
                        && s_ind.get_unchecked(s_tr + 5) == l_ind.get_unchecked(l_tr + 5)
                        && s_ind.get_unchecked(s_tr + 4) == l_ind.get_unchecked(l_tr + 4)
                        && s_ind.get_unchecked(s_tr + 3) == l_ind.get_unchecked(l_tr + 3)
                        && s_ind.get_unchecked(s_tr + 2) == l_ind.get_unchecked(l_tr + 2)
                        && s_ind.get_unchecked(s_tr + 1) == l_ind.get_unchecked(l_tr + 1)
                    {
                        let y_simd = f32x8::from_slice_unaligned(&l_val[l_tr..]);
                        let x_simd = f32x8::from_slice_unaligned(&s_val[s_tr..]);
                        let diff = x_simd - y_simd;
                        d_acc_8 += diff * diff;
                        s_tr += 8;
                        l_tr += 8;
                    } else {
                        let a = l_val.get_unchecked(l_tr) - s_val.get_unchecked(s_tr);
                        total[total_i] += a * a;
                        total_i = (total_i + 1) % 32;
                        s_tr += 1;
                        l_tr += 1;
                    }
                } else {
                    total[total_i] += s_val.get_unchecked(s_tr) * s_val.get_unchecked(s_tr);
                    total_i = (total_i + 1) % 32;
                    s_tr += 1;
                }
            }

            while s_ind.len() > s_tr {
                while l_ind.len() > l_tr && l_ind.get_unchecked(l_tr) < s_ind.get_unchecked(s_tr) {
                    total[total_i] += l_val.get_unchecked(l_tr) * l_val.get_unchecked(l_tr);
                    total_i = (total_i + 1) % 32;
                    l_tr += 1;
                }
                if l_ind.len() > l_tr && s_ind.get_unchecked(s_tr) == l_ind.get_unchecked(l_tr) {
                    let a = l_val.get_unchecked(l_tr) - s_val.get_unchecked(s_tr);
                    total[total_i] += a * a;
                    total_i = (total_i + 1) % 32;
                    s_tr += 1;
                    l_tr += 1;
                } else {
                    total[total_i] += s_val.get_unchecked(s_tr) * s_val.get_unchecked(s_tr);
                    total_i = (total_i + 1) % 32;
                    s_tr += 1;
                }
            }

            while l_val.len() > l_tr {
                total[total_i] += l_val.get_unchecked(l_tr) * l_val.get_unchecked(l_tr);
                total_i = (total_i + 1) % 32;
                l_tr += 1;
            }

            total[0] += d_acc_8.sum();
        }
        total.iter().fold(0.0, |a, x| a + x).sqrt()
    }
}

/// CSR matrix format sparse data cloud
#[derive(Debug)]
struct SparsePointCloud<F: Metric> {
    dim: usize,
    count: usize,
    chunk: usize,
    a: Vec<f32>,
    ia: Vec<u32>,
    ja: Vec<u32>, // the traditional names
    metric:std::marker::PhantomData<F>,
}

impl<F: Metric> SparsePointCloud<F> {
    fn new_random(dim: usize, count: usize, sparse_coef: f32) -> SparsePointCloud<F> {
        let mut rng: StdRng = rand::SeedableRng::from_seed(SETSEED);
        let non_zero_count = (((dim * count) as f32) * sparse_coef) as usize;
        let mut a = Vec::with_capacity(non_zero_count);
        let mut ia = Vec::with_capacity(count + 1);
        let mut ja = Vec::with_capacity(non_zero_count);
        let mut consumed_nz: usize = 0;
        ia.push(0);
        while non_zero_count > consumed_nz && ia.len() < count + 1 {
            let non_zero_row = min(non_zero_count - consumed_nz, rng.gen_range(0, dim));
            ja.extend(
                rand::seq::index::sample(&mut rng, dim, non_zero_row)
                    .iter()
                    .map(|i| i as u32),
            );
            a.extend((0..non_zero_row).map(|_i| rng.gen::<f32>()));
            (&mut ja[consumed_nz..(consumed_nz + non_zero_row)]).sort();
            consumed_nz += non_zero_row;
            ia.push(consumed_nz as u32);
        }
        while ia.len() < count + 1 {
            ia.push(consumed_nz as u32);
        }
        let chunk = max((2000.0 / (dim as f32 * sparse_coef)) as usize, 20);

        SparsePointCloud {
            dim,
            count,
            chunk,
            a,
            ia,
            ja,
            metric: PhantomData,
        }
    }

    fn new_random_dense(
        sparse_dim: usize,
        count: usize,
        sparse_coef: f32,
        dense_dim: usize,
    ) -> SparsePointCloud<F> {
        let mut rng: StdRng = rand::SeedableRng::from_seed(SETSEED);
        let non_zero_count = (((sparse_dim * count) as f32) * sparse_coef) as usize;
        let mut a = Vec::with_capacity(non_zero_count + dense_dim * count);
        let mut ia = Vec::with_capacity(count + 1);
        let mut ja = Vec::with_capacity(non_zero_count + dense_dim * count);
        let mut consumed_nz: usize = 0;
        let mut true_nz: usize = 0;

        ia.push(0);
        while non_zero_count > consumed_nz && ia.len() < count + 1 {
            let non_zero_row = min(non_zero_count - consumed_nz, rng.gen_range(0, sparse_dim));
            ja.extend(
                rand::seq::index::sample(&mut rng, sparse_dim, non_zero_row)
                    .iter()
                    .map(|i| i as u32),
            );
            a.extend((0..non_zero_row).map(|_i| rng.gen::<f32>()));
            (&mut ja[true_nz..(true_nz + non_zero_row)]).sort();

            a.extend((0..dense_dim).map(|_i| rng.gen::<f32>()));
            ja.extend(sparse_dim as u32..(sparse_dim + dense_dim) as u32);
            consumed_nz += non_zero_row;
            true_nz += non_zero_row + dense_dim;
            ia.push(true_nz as u32);
        }
        while ia.len() < count + 1 {
            ia.push(true_nz as u32);
        }
        let chunk = max(
            (2000.0 / ((sparse_dim + dense_dim) as f32 * sparse_coef)) as usize,
            20,
        );

        SparsePointCloud {
            dim: sparse_dim + dense_dim,
            count,
            chunk,
            a,
            ia,
            ja,
            metric: PhantomData,
        }
    }

    fn new_zeros(dim: usize, count: usize) -> SparsePointCloud<F> {
        let a = Vec::new();
        let ia = vec![0; count + 1];
        let ja = Vec::new();
        let chunk = min(15000 / dim, 20);
        SparsePointCloud {
            dim,
            count,
            chunk,
            a,
            ia,
            ja,
            metric: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn get<'a, 'b>(&'a self, i: usize) -> Result<SparsePointRef<'b>, &str>
    where
        'a: 'b,
    {
        let (ja, a) = self._get_(i)?;
        Ok(SparsePointRef { ja, a })
    }

    fn _get_(&self, i: usize) -> Result<(&[u32], &[f32]), &str> {
        let s;
        let e;
        if i + 1 < self.ia.len() {
            s = self.ia[i] as usize;
            e = self.ia[i + 1] as usize;
            let ja: &[u32] = &self.ja[s..e];
            let a: &[f32] = &self.a[s..e];
            Ok((ja, a))
        } else {
            let ja: &[u32] = &self.ja[0..0];
            let a: &[f32] = &self.a[0..0];
            Ok((ja, a))
        }
    }

    pub fn simple_dists<T>(&self, x: &T, indexes: &[usize]) -> Vec<f32>
    where
        T: SparsePoint,
    {
        let x_ind = x.indexes();
        let x_val = x.values();
        indexes
            .iter()
            .map(|i| {
                let (yja, ya) = &self._get_(*i).unwrap();
                F::sparse(&x_ind, &x_val, yja, ya)
            })
            .collect()
    }

    pub fn dists<T>(&self, x: &T, indexes: &[usize]) -> Result<Vec<f32>, &str>
    where
        T: SparsePoint,
    {
        let x_ind = x.indexes();
        let x_val = x.values();
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
                    s.spawn(|_| {
                        for i in range {
                            match self._get_(indexes[i]) {
                                Ok((yja, ya)) => unsafe {
                                    *dists_ptr1.p.add(i) = F::sparse(&x_ind, &x_val, yja, ya)
                                },
                                Err(e) => {
                                    unsafe {
                                        *dists_ptr1.p.add(i) = 0.0;
                                    }
                                    *error.lock().unwrap() = Err(e);
                                }
                            }
                        }
                    });
                    start += self.chunk;
                }
                let range = start..len;
                s.spawn(|_| {
                    for i in range {
                        match self._get_(indexes[i]) {
                            Ok((yja, ya)) => unsafe {
                                *dists_ptr1.p.add(i) = F::sparse(&x_ind, &x_val, yja, ya)
                            },
                            Err(e) => {
                                unsafe {
                                    *dists_ptr1.p.add(i) = 0.0;
                                }
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
                    let (yja, ya) = self._get_(*i)?;
                    Ok(F::sparse(&x_ind, &x_val, yja, ya))
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn structured_get_test() {
        let data = SparsePointCloud::<L2> {
            dim: 5,
            count: 5,
            chunk: 1,
            a: vec![1.0, 2.0, 3.0],
            ia: vec![0, 2, 3, 3, 3, 3],
            ja: vec![1, 2, 4],
            metric: PhantomData,
        };

        let first = data.get(0).unwrap();

        assert_eq!(first.a, [1.0, 2.0]);
        assert_eq!(first.ja, [1, 2]);
        assert_eq!(first.dense(5), vec![0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn structured_dist_0_test() {
        let data = SparsePointCloud::<L2> {
            dim: 5,
            count: 5,
            chunk: 1,
            a: vec![3.0, 4.0, 3.0],
            ia: vec![0, 2, 3, 3, 3, 3],
            ja: vec![1, 2, 4],
            metric: PhantomData,
        };

        let zero_vec = SparsePointVec::zero();
        let indexes: Vec<usize> = (0..COUNT).collect();
        let dists = data.simple_dists(&zero_vec, &indexes[..]);
        let correct: Vec<f32> = vec![5.0, 3.0, 0.0, 0.0, 0.0];
        for (a, b) in dists.iter().zip(correct) {
            assert_approx_eq!(a, b, 0.0001);
        }
    }

    #[test]
    fn structured_dist_nz_test_2() {
        let data = SparsePointCloud::<L2> {
            dim: 5,
            count: 5,
            chunk: 1,
            a: vec![3.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            ia: vec![0, 2, 6, 6, 6, 6],
            ja: vec![1, 2, 0, 1, 3, 4],
            metric: PhantomData,
        };

        let nonzero_vec = SparsePointVec {
            a: vec![-2.0],
            ja: vec![2],
        };

        let indexes: Vec<usize> = (0..COUNT).collect();
        let dists = data.simple_dists(&nonzero_vec, &indexes[..]);
        let correct: Vec<f32> = vec![5.0, 4.0, 2.0, 2.0, 2.0];
        for (a, b) in dists.iter().zip(correct) {
            assert_approx_eq!(a, b, 0.0001);
        }
    }

    #[test]
    fn random_ia_len() {
        let random_data = SparsePointCloud::<L2>::new_random(5, 6, 0.5);
        assert_eq!(random_data.ia.len(), 7);
    }

    #[test]
    fn random_ja_a_eq() {
        let random_data = SparsePointCloud::<L2>::new_random(5, 6, 0.5);
        assert_eq!(random_data.ja.len(), random_data.a.len());
    }

    #[test]
    fn random_ja_a_len() {
        let random_data = SparsePointCloud::<L2>::new_random(5, 6, 0.5);
        assert!(random_data.a.len() <= 15);
    }

    #[test]
    fn random_ja_dim() {
        let random_data = SparsePointCloud::<L2>::new_random(5, 6, 0.5);
        for j in random_data.ja {
            assert!(j < (random_data.dim as u32));
        }
    }

    #[test]
    fn random_dense_ia_len() {
        let random_dense_data = SparsePointCloud::<L2>::new_random_dense(5, 6, 0.5, 3);
        assert_eq!(random_dense_data.ia.len(), 7);
    }

    #[test]
    fn random_dense_ja_a_eq() {
        let random_dense_data = SparsePointCloud::<L2>::new_random_dense(5, 6, 0.5, 3);
        assert_eq!(random_dense_data.ja.len(), random_dense_data.a.len());
    }

    #[test]
    fn random_dense_ja_a_len() {
        let random_dense_data = SparsePointCloud::<L2>::new_random_dense(5, 6, 0.5, 3);
        assert!(random_dense_data.a.len() <= 15 + 6 * 3);
    }

    #[test]
    fn random_dense_ja_dim() {
        let random_dense_data = SparsePointCloud::<L2>::new_random_dense(5, 6, 0.5, 3);
        for j in random_dense_data.ja {
            assert!(j < (random_dense_data.dim as u32));
        }
    }

    #[bench]
    fn bench_sparse_dence_a_simple(b: &mut Bencher) {
        let v1 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
        let v2 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
        b.iter(|| L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()));
        assert_approx_eq!(
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            0.0001
        );
    }

    #[bench]
    fn bench_sparse_dence_a_simple_simd(b: &mut Bencher) {
        let v1 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
        let v2 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
        b.iter(|| l2_sparse_simd(v1.indexes(), v1.values(), v2.indexes(), v2.values()));
        assert_approx_eq!(
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            l2_sparse_simd(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            0.0001
        );
    }

    #[bench]
    fn bench_sparse_a_simple(b: &mut Bencher) {
        let v1 = SparsePointVec::random(DIM + DDIM, SP_COEF);
        let v2 = SparsePointVec::random(DIM + DDIM, SP_COEF);
        b.iter(|| L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()));
        assert_approx_eq!(
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            0.0001
        );
    }

    #[bench]
    fn bench_sparse_a_simple_simd(b: &mut Bencher) {
        let v1 = SparsePointVec::random(DIM + DDIM, SP_COEF);
        let v2 = SparsePointVec::random(DIM + DDIM, SP_COEF);
        b.iter(|| l2_sparse_simd(v1.indexes(), v1.values(), v2.indexes(), v2.values()));
        assert_approx_eq!(
            L2::sparse(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            l2_sparse_simd(v1.indexes(), v1.values(), v2.indexes(), v2.values()),
            0.0001
        );
    }

    #[bench]
    fn bench_simple_dists(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random(DIM + DDIM, COUNT, SP_COEF);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random(DIM + DDIM, SP_COEF);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.simple_dists(&zero_vec, i);
        });
    }

    #[bench]
    fn bench_simple_dists_simd(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random(DIM + DDIM, COUNT, SP_COEF);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random(DIM + DDIM, SP_COEF);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.simple_dists(&zero_vec, i);
        });
    }

    #[bench]
    fn bench_dists(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random(DIM + DDIM, COUNT, SP_COEF);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random(DIM + DDIM, SP_COEF);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.dists(&zero_vec, i);
        });
    }


    #[bench]
    fn bench_dists_simd(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random(DIM + DDIM, COUNT, SP_COEF);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random(DIM + DDIM, SP_COEF);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.dists(&zero_vec, i);
        });
    }

    #[bench]
    fn bench_sparse_dence_simple_dists(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random_dense(DIM, COUNT, SP_COEF, DDIM);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.simple_dists(&zero_vec, i);
        });
    }

    #[bench]
    fn bench_sparse_dence_simple_dists_simd(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random_dense(DIM, COUNT, SP_COEF, DDIM);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.simple_dists(&zero_vec, i);
        });
    }

    #[bench]
    fn bench_sparse_dence_dists(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random_dense(DIM, COUNT, SP_COEF, DDIM);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.dists(&zero_vec, i);
        });
    }


    #[bench]
    fn bench_sparse_dence_dists_simd(b: &mut Bencher) {
        let zero_data = SparsePointCloud::<L2>::new_random_dense(DIM, COUNT, SP_COEF, DDIM);
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| {
            let zero_vec = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
            let i = test::black_box(&indexes[..COUNT / 2]);
            zero_data.dists(&zero_vec, i);
        });
    }
}
