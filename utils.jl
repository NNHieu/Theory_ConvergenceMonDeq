using LinearAlgebra
using SparseArrays

identity_matrix(m) = Diagonal(ones(m))

function commutation_matrix(m, n)
    row = range(1, m*n)
    col = vec(reshape(row, m, n)')
    data = ones(Int64, m*n)
    return sparse(row, col, data)
end