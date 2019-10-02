#![feature(test)]
#![allow(dead_code)]
//#![deny(warnings)]
#[macro_use]
extern crate assert_approx_eq;
extern crate rand;
extern crate test;

use rand::seq::SliceRandom;
use rand::thread_rng;   

mod dense;
use dense::*;

mod sparse;
use sparse::*;

const DIM: usize = 20;
const DDIM: usize = 10;
const COUNT: usize = 2;
const SP_COEF: f32 = 0.2;

fn main() {
    let zero_data = PointCloud::new_zeros(DIM, COUNT, l2_dense);
    let zero_vec = vec![0.0; DIM];
    let mut indexes: Vec<usize> = (0..COUNT).collect();
    indexes.shuffle(&mut thread_rng());
    let dists = zero_data
        .dists(&zero_vec[..], &indexes[..COUNT / 2])
        .unwrap();

    assert_eq!(dists[0], 0.0);

    let v1 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
    let v2 = SparsePointVec::random_dense(DIM, SP_COEF, DDIM);
    println!("================================================");
    println!(
        "sparse {}",
        l2_sparse_simd(v1.indexes(), v1.values(), v2.indexes(), v2.values())
    );
}
