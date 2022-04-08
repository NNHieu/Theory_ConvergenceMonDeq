include("utils.jl")
include("mondeq.jl")

using LinearAlgebra
using TensorCast
import .MonDeq
using JLD2

norm_row(X) = sqrt.(sum(abs2, X, dims=2))

num_sample = 5
d = 3
k = 100
alpha = 0.1

Id_k = identity_matrix(k)
Id_kk = identity_matrix(k*k)
K_k = commutation_matrix(k, k)
P_BP_B = 2(Id_kk - K_k)
doubN_k = (Id_kk + K_k)

Id_kk = nothing

eigmin_H1 = []
eigmin_H2 = []
eigmin_H3 = []
eigmin_H = []

eigmin_H1_H2 = []
eigmin_H1_H3 = []
eigmin_H2_H3 = []

for i in 1:1000
    X = randn(num_sample, d)
    X = X ./ norm_row(X)

    A = randn(k, k) / k
    B = randn(k, k) / k
    W = MonDeq.W(A, B, alpha, k)
    U = randn(k, d) / k
    u = randn(k) 

    P_AP_A = doubN_k*kron(Id_k, A'A)*doubN_k

    Z_star = MonDeq.z_star(X, zeros(Float64, (num_sample, k)), W, U)
    S = MonDeq.relu_diff.(Z_star)
    Q = MonDeq.fixedpoint_bwd(zeros(size(S)), repeat(u, 1, num_sample)', S, W)

    @cast Z_Q[i, (k, j)] := Z_star[i, j] * Q[i, k] 

    H1 = Z_Q * (P_BP_B - P_AP_A) * Z_Q'
    H2 = (X * X') .* (Q * Q')
    H3 = Z_star * Z_star'
    H = H1 + H2 + H3

    push!(eigmin_H1, eigvals(H1)[1])
    push!(eigmin_H2, eigvals(H2)[1])
    push!(eigmin_H3, eigvals(H3)[1])
    push!(eigmin_H, eigvals(H)[1])

    push!(eigmin_H1_H2, eigvals(H1 + H2)[1])
    push!(eigmin_H1_H3, eigvals(H1 + H3)[1])
    push!(eigmin_H2_H3, eigvals(H2 + H3)[1])

end
jldopen("data/kernel_$(num_sample)_$(d)_$(k)_$(alpha).jld2", "w") do file
    write(file, "eigmin_H1", eigmin_H1)
    write(file, "eigmin_H2", eigmin_H2)
    write(file, "eigmin_H3", eigmin_H3)
    write(file, "eigmin_H", eigmin_H)
    write(file, "eigmin_H1_H2", eigmin_H1_H2)
    write(file, "eigmin_H1_H3", eigmin_H1_H3)
    write(file, "eigmin_H2_H3", eigmin_H2_H3)
end

close(f)