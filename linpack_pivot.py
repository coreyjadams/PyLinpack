
#!/usr/bin/python3

# Linpack written in python3 / numpy
#
# MJ Rutter Nov 2015
#
# Apr 2020 updated to use a.item() instead of np.asscalar(a)
#
# C Adams Update to JAX Jan 2024

import jax
from jax import jit
from jax import numpy, lax
from jax.numpy import  matmul
from jax import random
from time import time
from jax import scipy
from datetime import datetime
import sys

import argparse




def initialize_matrix(random_key, shape, dtype):
    A = random.uniform(random_key, shape, dtype=dtype)-0.5
    return A

def compute_B(A):
    return A.sum(axis=1).reshape((-1,1))

@jit
def solve(A, B):
    return numpy.linalg.solve(A, B)

@jit
def residual(A, B, X):
    return numpy.matmul(A,X) - B


def pivot(_A, _k, _kp):
    '''
    Perform a pivot operation on rows if needed
    '''
    temp = _A[_k]
    _A = _A.at[_k].set(_A[_kp])
    _A = _A.at[_kp].set(temp)

    return _A


def no_pivot(_A, _k, _kp):
    '''
    Dummy function for the no-pivot case to make the conditional easy
    '''
    return _A


def form_gauss_vector(_A, _k):
    """
    Form the gauss vector and update the matrix value as the return

    """

    # Start with the values at every point in the target column
    gauss_vector = _A[:,_k] 
    target_row = _A[_k,:]

    a_nn = _A[_k,_k]
    # Scale the gauss vector 
    gauss_vector = gauss_vector / a_nn

    # This should be the biggest value at or below the diagonal 
    # in this column, from the pivoting:

    # Set to 0 the entries above and including the diagonal:
    mask = numpy.arange(gauss_vector.shape[0]) > _k
    gauss_vector = numpy.where(mask,  gauss_vector, 0.0)

    # Update the matrix by subtracting off the gauss vector.
    # use outer product shaping to get the whole thing:

    _A = _A - numpy.outer(gauss_vector, target_row)

    return _A, gauss_vector


pivot = jit(pivot, donate_argnums=0)
no_pivot = jit(no_pivot, donate_argnums=0)
form_gauss_vector = jit(form_gauss_vector, donate_argnums=0)



@jit
def lu_iteration(carry, _k):
    # This needs to become a function we can iterate over:

    # First, find the largest entry in A[k:,k] and
    # permute its row to A[k,k]

    # Keep track of this permutation too!  It is the pivot matrix

    # Doing this with a full region and we can use a statically-sized
    # mask and a where operation to dynamically mask based on index.

    # Surely it's faster to only look over the "real" region of interest
    # but that will trigger a recompile every time which isn't ideal

    # Pick out the current matrix from the carry object:
    _A, _L = carry[0], carry[1]

    # Current pivots
    pivots = carry[2]

    # First, decide if we are going to pivot:
    full_region = _A[:,_k] # Take the whole column
    # print("full_region: ",  full_region)

    # Only look at the area below the diagonal:
    mask = numpy.arange(full_region.shape[0]) >= _k
    full_region = numpy.where(mask, full_region, 0.0)

    # print("fulll_region: ", full_region)

    # Do we need to pivot?
    _kp = numpy.abs(full_region).argmax()

    # TODO: Switch the pivots here!
    temp = pivots[_k]
    pivots = pivots.at[_k].set(pivots[_kp])
    pivots = pivots.at[_kp].set(temp)

    

    # Apply a pivot if needed, and do both A and L:
    _A  = lax.cond(_kp != _k, pivot, no_pivot, _A, _k, _kp)
    # print("Pivoted _A: ", _A)
    _L  = lax.cond(_kp != _k, pivot, no_pivot, _L, _k, _kp)
    
    # get the update and form the gauss vectors if needed:
    _A, gauss_vector = form_gauss_vector(_A, _k)

    # Update _L:
    _L = _L.at[:,_k].set(gauss_vector)
    # Need to put ones on the diagonal of L:
    _L = _L.at[_k,_k].set(1.0)

    return (_A, _L, pivots), None

lu_iteration = jit(lu_iteration)


@jit
def LU_partial_pivoting(A):
    """LU_partial_pivoting(A, overwrite_a=False)

    Factor a matrix A into it's LU decomposition with partial pivoting
    """


    N = A.shape[0]
    # Intial list of pivots (aka none, everything in order)
    perms = numpy.arange(0, N)
    L = numpy.zeros_like(A)

    # Initial L matrix (all zeros):
    carry = (
        A,
        L,
        perms
    )

    for k in range(N):
        carry, _ = lu_iteration(carry, k)
    # carry, _ = lax.scan(lu_iteration, carry, perms)

    U = carry[0]
    L = carry[1]


    # Create the permutation matrix:
    P = numpy.zeros_like(A)
    P = P.at[carry[2], perms].add(1.0)

    return P, L, U

def main(args):

    # Create a random number seed:
    key = random.PRNGKey(args.seed)

    # Create AxA array of random numbers -0.5 to 0.5
    key, subkey = random.split(key)

    A = initialize_matrix(subkey, (N,N), args.dtype)

    # A = numpy.asarray([[2, 1, 1, 0], [4, 3, 3, 1], [8,7,9,5], [6, 7, 9, 8]], dtype=args.dtype)

    B = compute_B(A)

    print("B.shape: ", B.shape)

    P, L, U = scipy.linalg.lu(A, permute_l=False)
    jax.profiler.start_trace("./prof/scipy-lu/")
    t = time()
    # Convert to matrices
    P, L, U = scipy.linalg.lu(A, permute_l=False)
    U.block_until_ready()
    t = time() - t
    jax.profiler.stop_trace()
    print(f"Scipy time: {t:.5f}")

    # print("Scipy P: \n", P)
    # print("Scipy L: \n", L)
    # print("Scipy U: \n", U)
    # print(matmul(matmul(P, L),U) - A)
    P, L, U = LU_partial_pivoting(A)

    jax.profiler.start_trace("./prof/custom-lu/")
    t = time()
    P, L, U = LU_partial_pivoting(A)
    U.block_until_ready()
    t = time() - t
    jax.profiler.stop_trace()
    print(f"Custom time: {t:.5f}")


if __name__ == "__main__":

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

    print("Running Linpack ",N,"x",N,"\n")
    print("Arguments: ")
    for arg in vars(args):
        print("  ", arg, getattr(args,arg))

    ops=(2.0*N)*N*N/3.0+(2.0*N)*N

    main(args)