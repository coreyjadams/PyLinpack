#/usr/bin/env julia

# LINPACK on a single device in JULIA
using Profile

import Random
Random.seed!(1234)

using Random
using ArgParse
using LinearAlgebra
using BenchmarkTools


function initialize_matrix(shape::Integer, dtype::DataType)
    A = rand(dtype, (shape, shape)) .- 0.5
    return A
end

function compute_B(A::Matrix)::Vector
    N = size(A,1)
    return reshape(sum(A, dims=1), (N,))
end


@views function pivot!(_A::AbstractArray, _k::Integer, _kp::Integer)
    """
    Perform a pivot operation on rows if needed

    This is row-pivoting only (no columns)

    Therefore, This only needs to be done on columns > _k
    (assuming _k < _kp but should check maybe!)

    """

    N = size(_A, 1)
    for idx in _k:N
        # Get the value in row _kp:
        t = _A[_kp, idx]
        # Write row _k to row _kp
        _A[_kp, idx] = _A[_k, idx]
        # _A[_kp][idx] = _A[_k][idx]
        # Write the original value from _kp to _k:
        _A[_k, idx] = t
    end
    return
end

@views function pivot_full!(_A::AbstractArray, _k::Integer, _kp::Integer)
    """
    Perform a pivot operation on rows if needed

    This is row-pivoting only (no columns)

    Therefore, This only needs to be done on columns > _k
    (assuming _k < _kp but should check maybe!)

    """

    N = size(_A, 1)
    for idx in 1:N
        # Get the value in row _kp:
        t = _A[_kp, idx]
        # Write row _k to row _kp
        _A[_kp, idx] = _A[_k, idx]
        # Write the original value from _kp to _k:
        _A[_k, idx] = t
    end

    return

end

function form_gauss_vector!(_A::Matrix{T}, _k::Integer)::Vector{T} where {T<:Real}
    """
    Form the gauss vector and update the matrix value as the return

    The vector, at each stage of the algorithm, needs less and less of a column/matrix



    """

    # Start with the values at every point in the target column
    N = size(_A, 1)

    gauss_vector = copy(_A[_k:N, _k] / _A[_k,_k])
    @views target_row   = _A[_k, _k:N]


    # Scale the gauss vector 
    # gauss_vector = (gauss_vector / a_nn) # Copy?

    # This should be the biggest value at or below the diagonal 
    # in this column, from the pivoting:

    # Update the matrix by subtracting off the gauss vector.
    # use shaping and broadcasting to get the whole thing:

    # print(numpy.outer(gauss_vector, target_row))

    # Now, subtract off the low right corner from the outer product:


    # This is how to form the outer product vector:
    # outer1 = gauss_vector .* target_row'


    # Skip the first row:

    @inbounds for idy in _k:N
        t_y = idy - _k + 1
        @inbounds for idx in _k+1:N
            # Using outer product vector:
            # _A[idx, idy] = _A[idx, idy] - outer[idx  - _k + 1, idy - _k + 1]
            _A[idx, idy] -= gauss_vector[idx - _k + 1] * target_row[t_y]
        end
    end

    return gauss_vector

end

function LU_partial_pivoting(_A::AbstractArray)
    """LU_partial_pivoting(_A, overwrite_a=False)

    Factor a matrix _A into it's LU decomposition with partial pivoting
    """

    # U is going to be what is left of _A when we're all done

    N = size(_A, 1)
    # _A = _A.copy()
    # Intial list of pivots (aka none, everything in order)
    perms = collect(1:N)
    L     = zero(_A)



    # Iterate over the rows:

    # Don't do the last row

    @inbounds for k in 1:N-1

        # First, find the largest entry in _A[k:,k] and
        # permute its row to _A[k,k]

        # Keep track of this permutation too!  It is the pivot matrix


        # Pick out the current matrix from the carry object:


        # First, decide if we are going to pivot:
        # Looking only in column k, from k down along rows
        # column = _A[k:N,k]

        # Take the whole column from k down
        max_index = argmax(abs.(_A[k:N,k])) 
        max_index = max_index + k - 1

        if max_index != k
            # Do the Pivot:
            pivot!(_A, k, max_index)
            pivot_full!(L, k, max_index)
            t = perms[k]
            perms[k] = perms[max_index]
            perms[max_index] = t
        end

        # Next, for the gauss vector:
        # Again, an inplace update:
        gauss_vector = form_gauss_vector!(_A, k)
        # Write the gauss vector into L:
        L[k:N, k] = gauss_vector

    end

    # Create the permutation matrix:
    P = zero(_A)
    for idx in 1:N
        P[perms[idx],idx] = 1.0
    end

    # Fill in the last L:
    L[N,N] = 1

    return P, L, _A

end

function forward_sub(L::Matrix, B::Vector)

    N = size(L, 1)

    # Forward substitution:
    Y = zero(B)
    for i in 1:N
        tmp = B[i]
        for j in 1:i-1
            tmp -= L[i,j]*Y[j]
        end
        Y[i] = tmp / L[i,i]
    end

    return Y
end

function back_sub(U::Matrix, B::Vector)

    N = size(U, 1)
    X = zero(B)

    for i in reverse(1:N)
        tmp = B[i]
        for j in i+1:N
            tmp -= U[i,j] * X[j]
        end
        X[i] = tmp / U[i,i]
    end

    return X
end

function solve(M::Matrix, B::Vector)
    
    # Decompose M into P, L, U:
    
    P, L, U = LU_partial_pivoting(M)

    # P^-1 = P^T, so we can get to LU quickly:

    B = P' * B

    # First, let U X = Y
    # Then,  L Y = B
    # Solve this for Y:

    N = size(M, 1)


    # Forward substitution:
    Y = forward_sub(L, B)

    # Back Substitution to solve U X = Y
    X = back_sub(U, Y)

    return X

end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-N"
            help = "Matrix Size (NxN)"
            default = 2000
            arg_type = Int
        "--dtype", "-d"
            help = "Data Type"
            arg_type = DataType
            default = Float64
    end

    return parse_args(s)
end

function main()

    parsed_args = parse_commandline()
    display("Parsed args:")
    for (arg,val) in parsed_args
        display("  $arg  =>  $val")
    end
    display(parsed_args)
    
    dtype = parsed_args["dtype"]
    M = initialize_matrix(parsed_args["N"], dtype)
    N = size(M, 1)

    B = compute_B(M)
    println(size(B))
    println(typeof(B))

    # Warm up:
    Mc = copy(M)

    X = solve(Mc, B)


    # # P, L, U = LU_partial_pivoting(Mc)

    # # @btime P, L, U = LU_partial_pivoting($Mc)

    # @profile (for i in 1:10000; form_gauss_vector!(Mc, 50); end)
    # @profile (for i in 1:100; solve(Mc, B); end)

    # Profile.print()
    # @btime form_gauss_vector!($Mc, 50)

    b = @timed X = solve(Mc, B)
    t = b.time

    ops=(2.0*N)*N*N/3.0+(2.0*N)*N

    R = M*X - B
    # println("R: ", R)

    Rs = maximum(abs.(R.*M))

    println("Time is ", t)
    println("MFLOPS: ", ops * 1e-6 / t)

    # # # Rs=max(abs(R.A)).item()
    
    # # # nx=max(abs(X)).item()
    
    println("Residual is ",Rs)

end

main()

