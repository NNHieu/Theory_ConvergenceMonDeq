include("utils.jl")
include("mondeq.jl")

using LinearAlgebra
using TensorCast
import .MonDeq
using Symbolics

norm_row(X) = sqrt.(sum(abs2, X, dims=2))

num_sample = 1
d = 2
k = 3
alpha = 0.1

Id_k = identity_matrix(k)
Id_kk = identity_matrix(k*k)
K_k = commutation_matrix(k, k)
P_BP_B = 2(Id_kk - K_k)
doubN_k = (Id_kk + K_k)

