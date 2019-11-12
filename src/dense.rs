use rand::Rng;
use rayon::prelude::*;
use std::marker::PhantomData;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::{max, min};
use std::sync::{Arc, Mutex};

use super::distances::*;


#[derive(Debug)]
pub struct PointCloud<F: Metric> {
    dim: usize,
    data: Vec<f32>,
    chunk: usize,
    metric:std::marker::PhantomData<F>,
}

// To bypass the borrow checker and do bad things
struct MyBox {
    p: *mut f32,
}
unsafe impl Send for MyBox {}
unsafe impl Sync for MyBox {}

const DIM: usize = 1000 * 3;
const COUNT: usize = 1000;

impl<F:Metric> PointCloud<F> {
    pub fn new_random(dim: usize, count: usize) -> PointCloud<F> {
        let mut rng = rand::thread_rng();
        let data = (0..(dim * count)).map(|_i| rng.gen::<f32>()).collect();
        let chunk = max(15000 / dim, 20);
        PointCloud::<F> {
            data,
            dim,
            chunk,
            metric: PhantomData,
        }
    }

    pub fn new_zeros(dim: usize, count: usize) -> PointCloud<F> {
        let data = vec![0.0; dim * count];
        let chunk = max(15000 / dim, 20);
        PointCloud::<F> {
            data,
            dim,
            chunk,
            metric: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        assert!(self.data.len() % self.dim == 0);
        self.data.len() / self.dim
    }

    pub fn get(&self, i: usize) -> Result<&[f32], &str> {
        Ok(&self.data[(i * self.dim)..((i + 1) * self.dim)])
    }

    pub fn chunk_dists(&self, x: &[f32], indexes: &[usize]) -> Result<Vec<f32>, &str> {
        let len = indexes.len();
        let mut dists: Vec<f32> = vec![0.0;len];
        let dist_iter = dists.par_chunks_mut(self.chunk);
        let indexes_iter = indexes.par_chunks(self.chunk);
        let error: Arc<Mutex<Result<(), &str>>> = Arc::new(Mutex::new(Ok(())));
        dist_iter.zip(indexes_iter).for_each(|(chunk_dists,chunk_indexes)| {
            for (d,i) in chunk_dists.iter_mut().zip(chunk_indexes) {
                match self.get(*i) {
                    Ok(y) => *d = (F::dense)(x, y),
                    Err(e) => {
                        *error.lock().unwrap() = Err(e);
                    }
                }
            }
        });
        (*error.lock().unwrap())?;
        Ok(dists)
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
                                Ok(y) => *dists_ptr1.p.add(i) = (F::dense)(x, y),
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
                            Ok(y) => *dists_ptr1.p.add(i) = (F::dense)(x, y),
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
                    Ok((F::dense)(x, y))
                })
                .collect()
        }
    }

    pub fn simple_dist(&self, x: &[f32], indexes: &[usize]) -> Vec<f32> {
        indexes
            .iter()
            .map(|i| (F::dense)(x, self.get(*i).unwrap()))
            .collect()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn it_works() {
        let zero_data = PointCloud::<L2>::new_zeros(DIM, COUNT);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        let dists = zero_data.simple_dist(&zero_vec[..], &indexes[..COUNT / 2]);

        assert_eq!(dists[0], 0.0);
    }

    #[bench]
    fn bench_l2_chunk(b: &mut Bencher) {
        let zero_data = PointCloud::<L2>::new_zeros(DIM, COUNT);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.chunk_dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_l2_dists(b: &mut Bencher) {
        let zero_data = PointCloud::<L2>::new_zeros(DIM, COUNT);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_chunk(b: &mut Bencher) {
        let zero_data = PointCloud::<Linfty>::new_zeros(DIM, COUNT);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.chunk_dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }

    #[bench]
    fn bench_linfty_dists(b: &mut Bencher) {
        let zero_data = PointCloud::<Linfty>::new_zeros(DIM, COUNT);
        let zero_vec = vec![0.0; DIM];
        let mut indexes: Vec<usize> = (0..COUNT).collect();
        indexes.shuffle(&mut thread_rng());
        b.iter(|| zero_data.dists(&zero_vec[..], &indexes[..COUNT / 2]));
    }
}
