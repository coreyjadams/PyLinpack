#!/usr/bin/python3

# Linpack written in python3 / numpy
#
# MJ Rutter Nov 2015
#
# Apr 2020 updated to use a.item() instead of np.asscalar(a)

# NOTE: I (Corey Adams) did not write this code, it is available here:
# https://www.mjr19.org.uk/linpack/source.html

from numpy import matrix, array, linalg, random, amax
from time import time
import sys

try:
  N=int(sys.argv[1])
except:
  N=2000

eps=2.22e-16

ops=(2.0*N)*N*N/3.0+(2.0*N)*N

print("Running Linpack ",N,"x",N,"\n")

# Create AxA array of random numbers -0.5 to 0.5

A=random.random_sample((N,N))-0.5

B=A.sum(axis=1)

# Convert to matrices

A=matrix(A)

B=matrix(B.reshape((N,1)))

na=amax(abs(A.A))

print("Na:", na)

t=time()
X=linalg.solve(A,B)
t=time()-t

R=A*X-B

Rs=max(abs(R.A)).item()

nx=max(abs(X)).item()

print("Residual is ",Rs)
print("Normalised residual is ",Rs/(N*na*nx*eps))
print("Machine epsilon is ",eps)
print("x[0]-1 is ",X[0].item()-1)
print("x[n-1]-1 is ",X[N-1].item()-1)

print("Time is ",t)
print("MFLOPS: ",ops*1e-6/t)