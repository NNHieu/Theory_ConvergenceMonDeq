include("mondeq.jl")

using LinearAlgebra
using ForwardDiff
using SparseArrays
import .MonDeq

k = 4
d = 3
Id = MonDeq.Id
Id_k = Id(k)
Id_kk = MonDeq.Id(k*k)

function commutation_matrix(m, n)
    row = range(1, m*n)
    col = vec(reshape(row, m, n)')
    data = ones(Int64, m*n)
    return sparse(row, col, data)
end

K_k = commutation_matrix(k, k)

W_mon = (A, B) -> MonDeq.W(A, B, 0.1, k) 

for i in range(1, 10)
    x = randn(d)
    z = randn(k)
    A = randn(k, k)
    B = randn(k, k)
    U = randn(k, d)

    W = W_mon(A, B)

    true_df_dz = ForwardDiff.jacobian(z -> MonDeq.g(z, x, W, U), z)
    true_df_dA = ForwardDiff.jacobian(A -> MonDeq.g(z, x, W_mon(A, B), U), A)
    true_df_dB = ForwardDiff.jacobian(B -> MonDeq.g(z, x, W_mon(A, B), U), B)
    true_df_dU = ForwardDiff.jacobian(U -> MonDeq.g(z, x, W, U), U)
    
    J = Diagonal(MonDeq.relu_diff.(W*z + U*x))
    P_A = (Id_kk + K_k)*kron(Id_k, A')
    P_B = Id_kk - K_k
    z_T_kron_J = kron(z', J)

    df_dz = (Id_k - J*W)
    df_dA = z_T_kron_J*P_A
    df_dB = -z_T_kron_J*P_B
    df_dU = -kron(x', J)

    if !isapprox(df_dz, true_df_dz)
        println("df_dz")
        println(df_dz)
        println(true_df_dz)
    end
    if !isapprox(df_dA, true_df_dA)
        println("df_dA")
        println(df_dA)
        println(true_df_dA)
        break
    end
    if  !isapprox(df_dB, true_df_dB)
        println("df_dB")
        println(df_dB)
        println(true_df_dB)
    end
    if  !isapprox(df_dU, true_df_dU)
        println("df_dU")
        println(df_dU)
        println(true_df_dU)
    end

    # println(df_dz == true_df_dz, df_dA == true_df_dA, df_dB == true_df_dB)
end

# println('Finish')