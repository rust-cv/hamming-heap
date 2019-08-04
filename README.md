# hamming-heap

Rust crate providing amortized constant time min heaps for binary features using hamming distance

This is used in the [HNSW](https://github.com/rust-photogrammetry/hnsw) crate.
It was originally developed for the [HWT](https://github.com/vadixidav/hwt) crate and continues to be useful
for doing fast nearest-neighbor searches. It currently has no tests itself, but it has been tested to varying degrees
in integration with the two crates.

Contributions are welcome! Currently benchmarks and tests would be appreciated.
