
#!/usr/bin/python3

# Linpack written in python3 / numpy
#
# MJ Rutter Nov 2015
#
# Apr 2020 updated to use a.item() instead of np.asscalar(a)
#
# C Adams Update to JAX Jan 2024

from jax import jit
from jax.numpy import linalg, amax, matmul
from jax import random
from time import time
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

print("Running Linpack ",N,"x",N,"\n")
print("Arguments: ")
for arg in vars(args):
    print("  ", arg, getattr(args,arg))

ops=(2.0*N)*N*N/3.0+(2.0*N)*N


def initialize_matrix(random_key, shape, dtype):
    A = random.uniform(random_key, shape, dtype=dtype)-0.5
    return A

def compute_B(A):
    return A.sum(axis=1).reshape((-1,1))

@jit
def solve(A, B):
    return linalg.solve(A, B)

def residual(A, B, X):
    return matmul(A,X) - B

# Create a random number seed:
key = random.PRNGKey(args.seed)

# Create AxA array of random numbers -0.5 to 0.5
key, subkey = random.split(key)

A = initialize_matrix(subkey, (N,N), args.dtype)

B = compute_B(A)

print("B.shape: ", B.shape)

# Convert to matrices



na=amax(abs(A), axis=(0,1))

X = solve(A,B)

t=time()
X=solve(A,B)
X.block_until_ready()
t=time()-t


R = residual(A, B, X)

Rs=max(abs(R))

nx=max(abs(X)).item()

print("Residual is ",Rs)
print("Normalised residual is ",Rs/(N*na*nx*eps))
print("Machine epsilon is ",eps)
print("x[0]-1 is ",X[0].item()-1)
print("x[n-1]-1 is ",X[N-1].item()-1)

print("Time is ",t)
print("MFLOPS: ",ops*1e-6/t)
print("GFLOPS: ",ops*1e-9/t)
print("TFLOPS: ",ops*1e-12/t)