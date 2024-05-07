
#!/usr/bin/python3

# Linpack written in python3 / numpy
#
# MJ Rutter Nov 2015
#
# Apr 2020 updated to use a.item() instead of np.asscalar(a)
#
# C Adams Update to JAX Jan 2024

import numpy
import numba
from time import time
from jax import scipy
from datetime import datetime
import sys



import argparse

parser = argparse.ArgumentParser("Linpack with JAX")

parser.add_argument("-N", type=int, help="Matrix size (NxN)", default=2000)
parser.add_argument("-s", "--seed", type=int, help="Random number seed", default=int(datetime.now().microsecond))
parser.add_argument("-d", "--dtype", type=str, help="Random number seed", 
                    default="float64",
                    choices=["float64", "float32", "float16", "bfloat16"])

args = parser.parse_args()

N = args.N

if args.dtype == "float16":
    eps = 9.77e-4
elif args.dtype == "float32":
    eps = 1.19e-7
if args.dtype == "float64":
    from jax import config
    config.update("jax_enable_x64", True)
    eps=2.22e-16

print("Running Linpack NUMBA ",N,"x",N,"\n")
print("Arguments: ")
for arg in vars(args):
    print("  ", arg, getattr(args,arg))

ops=(2.0*N)*N*N/3.0+(2.0*N)*N


# def initialize_matrix(random_key, shape, dtype):
#     A = random.uniform(random_key, shape, dtype=dtype)-0.5
#     return A

# def compute_B(A):
#     return A.sum(axis=1).reshape((-1,1))

# @jit
# def solve(A, B):
#     return numpy.linalg.solve(A, B)

# @jit
# def residual(A, B, X):
#     return numpy.matmul(A,X) - B

@numba.njit(parallel=False)
def pivot(_A, _k, _kp):
    '''
    Perform a pivot operation on rows if needed

    This is row-pivoting only (no columns)

    Therefore, This only needs to be done on columns > _k
    (assuming _k < _kp but should check maybe!)

    '''

    N = _A.shape[0]
    for idx in numba.prange(_k, N):
        # Get the value in row _kp:
        t = _A[_kp][idx]
        # Write row _k to row _kp
        _A[_kp][idx] = _A[_k][idx]
        # Write the original value from _kp to _k:
        _A[_k][idx] = t

    return _A

@numba.njit(parallel=False)
def pivot_full(_A, _k, _kp):
    '''
    Perform a pivot operation on rows if needed

    This is row-pivoting only (no columns)

    Therefore, This only needs to be done on columns > _k
    (assuming _k < _kp but should check maybe!)

    '''

    N = _A.shape[0]
    for idx in numba.prange(0, N):
        # Get the value in row _kp:
        t = _A[_kp][idx]
        # Write row _k to row _kp
        _A[_kp][idx] = _A[_k][idx]
        # Write the original value from _kp to _k:
        _A[_k][idx] = t

    return _A


@numba.njit(parallel=False)
def form_gauss_vector(_A, _k):
    """
    Form the gauss vector and update the matrix value as the return

    The vector, at each stage of the algorithm, needs less and less of a column/matrix



    """

    # Start with the values at every point in the target column
    N = _A.shape[0]

    # gauss_vector = numpy.zeros(shape=( N - _k, ))
    # target_row   = numpy.zeros(shape=( N - _k, ))
    # # gauss_vector = _A[:,_k:] 
    # # target_row = _A[_k:,:]
    # for idx in range(_k, N):
    #     gauss_vector[idx - _k] = _A[idx, _k]
    #     target_row[idx - _k]   = _A[_k, idx]

    gauss_vector = _A[_k:, _k]
    target_row   = _A[_k, _k:]
    # return target_row

    a_nn = _A[_k,_k]
    # print("a_nn", a_nn)
    gauss_vector = (gauss_vector / a_nn).copy()
    # This should be the biggest value at or below the diagonal 
    # in this column, from the pivoting:

    # Scale the gauss vector 
    # gauss_vector = _A[:,_k] / a_nn
    # print("gv_final: ", gauss_vector)
    # Update the matrix by subtracting off the gauss vector.
    # use shaping and broadcasting to get the whole thing:

    # print("gauss_vector: ", gauss_vector)

    # print(numpy.outer(gauss_vector, target_row))

    # Now, subtract off the low right corner from the outer product:
    outer = numpy.outer(gauss_vector, target_row)
    # Skip the first row:
    for idx in range(_k+1, N):
        for idy in numba.prange(_k, N):
            # _A[idx, idy] = _A[idx, idy] - outer[idx  - _k, idy - _k]
            _A[idx, idy] = _A[idx, idy] - gauss_vector[idx - _k]*target_row[idy - _k]

    return gauss_vector


A = numpy.random.uniform(size=(N,N)) - 0.5
scipy_A = A.copy()
# k = 0

# A_update, g = form_gauss_vector(A, k)
# print(g.shape)
# print(g)

# print(A_update.shape)
# print(A_update[0])

# @numba.njit
def LU_partial_pivoting(_A):
    """LU_partial_pivoting(_A, overwrite_a=False)

    Factor a matrix _A into it's LU decomposition with partial pivoting
    """

    # U is going to be what is left of _A when we're all done

    N = _A.shape[0]
    _A = _A.copy()
    # Intial list of pivots (aka none, everything in order)
    perms = numpy.arange(0, N)
    L     = numpy.zeros_like(_A)



    # Iterate over the rows:

    # Don't do the last row
    for k in range(N-1):

        # First, find the largest entry in _A[k:,k] and
        # permute its row to _A[k,k]

        # Keep track of this permutation too!  It is the pivot matrix


        # Pick out the current matrix from the carry object:


        # First, decide if we are going to pivot:
        # Looking only in column k, from k down along rows
        column = _A[k:,k]
        max_index = numpy.argmax(numpy.abs(column)) # Take the whole column from k down
        max_index = max_index + k
        if max_index != k:
            # Do the Pivot:
            _A = pivot(_A, k, max_index)
            L  = pivot_full(L, k, max_index)
            t = perms[k]
            perms[k] = perms[max_index]
            perms[max_index] = t

        # Next, for the gauss vector:
        # Again, an inplace update:
        gauss_vector = form_gauss_vector(_A, k)

        # Write the gauss vector into L:
        L[k:,k] = gauss_vector


    # Create the permutation matrix:
    P = numpy.zeros_like(_A)
    for idx in numba.prange(N):
        P[perms[idx],idx] = 1.0

    # Fill in the last L:
    L[N-1,N-1] = 1

    return P, L, _A
    # print("full_region: ",  full_region)




P, L, U = LU_partial_pivoting(A)

import scipy


P, L, U = scipy.linalg.lu(scipy_A, permute_l=False)
t = time()
# Convert to matrices
P, L, U = scipy.linalg.lu(scipy_A, permute_l=False)
t = time() - t
print(f"Scipy time: {t:.3f}")

# print("Scipy P: \n", P)
# print("Scipy L: \n", L)
# print("Scipy U: \n", U)
# print(numpy.matmul(numpy.matmul(P, L),U) - A)

t = time()
P, L, U = LU_partial_pivoting(A)
t = time() - t
print(f"Custom time: {t:.3f}")

# print("custom P: \n", P)
# print("custom L: \n", L)
# print("custom U: \n", U)

# print(numpy.matmul(numpy.matmul(P, L),U) - A)
# na=numpy.amax(abs(A), axis=(0,1))

# X = solve(A,B)

# t=time()
# X=solve(A,B)
# X.block_until_ready()
# t=time()-t
