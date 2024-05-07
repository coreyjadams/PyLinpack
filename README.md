# PyLinpack

This is a collection of scripts that implement the LINPACK benchmark in JAX, numpy, numba, etc.

They aren't writing the benchmark from scratch, for the most part - instead coming at this from user space and leveraging
the libraries pulled in by jax, numpy, etc.

NOTE: I (Corey Adams) did not write the original code, it is available here:
https://www.mjr19.org.uk/linpack/source.html

I did apply some updates to various pieces but really, in general, don't attribute any of this code or algorithms to the owner of this github!

# Performance Results

I am measuring performance on a laptop with an Nvidia gpu, GeForce RTX 3050.  Not all benchmarks have GPU implementations.  fp64 only, here.

## Original linpack (numpy)

Measured over a variety of sizes, 5 measurements each.  Reporting the best performance of each size, which of course has some noise so these are approximate numbers only.


|  N   | MFlops |
| :---: | :---: |
| 16   | 41.9   |
| 32   | 170.1  |
| 64   | 666.7  |
| 128  | 3395   |
| 256  | 12043  |
| 512  | 30233  |
| 1024 | 69982  |
| 2048 | 67222  |

# JAX Linpack (jax.numpy)

With Jax, we can use it as a drop-in replacement for Numpy.  Here are the CPU results:

|  N   | MFlops |
| :---: | :---: |
| 16   | 12.9   |
| 32   | 97.1  |
| 64   | 610.5  |
| 128  | 2546 |
| 256  | 9875  |
| 512  | 25175  |
| 1024 | 41100 |
| 2048 | 91000 |


And, here are the GPU results:

|  N   | MFlops |
| :---: | :---: |
| 16   | 144.7   |
| 32   | 911  |
| 64   | 3780  |
| 128  | 3995 |
| 256  | 4773 |
| 512  | 5118  |
| 1024 | 15681 |
| 2048 | 21112 |

## Julia (hand-written LU decomp w/ pivoting)

For Julia, on CPU, I see these results:

|  N   | MFlops |
| :---: | :---: |
| 16   | 388   |
| 32   | 1246  |
| 64   | 2604  |
| 128  | 3539  |
| 256  | 3888  |
| 512  | 4006  |
| 1024 | 3562  |
| 2048 | 3140  |