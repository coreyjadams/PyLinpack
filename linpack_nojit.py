
#!/usr/bin/python3

# Linpack written in python3 / numpy
#
# MJ Rutter Nov 2015
#
# Apr 2020 updated to use a.item() instead of np.asscalar(a)
#
# C Adams Update to JAX Jan 2024

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

eps=2.22e-16
N = args.N

if args.dtype == "float64":
    from jax import config
    config.update("jax_enable_x64", True)

print("Running Linpack ",N,"x",N,"\n")
print("Arguments: ")
for arg in vars(args):
    print("  ", arg, getattr(args,arg))

ops=(2.0*N)*N*N/3.0+(2.0*N)*N

# Create a random number seed:
key = random.PRNGKey(args.seed)

# Create AxA array of random numbers -0.5 to 0.5
key, subkey = random.split(key)

A=random.uniform(subkey, (N,N), dtype=args.dtype)-0.5
print("A.shape: ", A.shape)
B=A.sum(axis=1)
B = B.reshape((N,1))

print("B.shape: ", B.shape)

# Convert to matrices



na=amax(abs(A), axis=(0,1))

X = linalg.solve(A,B)

t=time()
X=linalg.solve(A,B)
X.block_until_ready()
t=time()-t


R=matmul(A,X)-B

Rs=max(abs(R),)

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