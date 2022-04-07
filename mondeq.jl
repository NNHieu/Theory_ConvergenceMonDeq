module MonDeq
    using LinearAlgebra
    using FixedPointAcceleration
    using NLsolve

    # export relu, relu_diff
    # export W, f, g, z_star, y

    # Activation function
    relu = x -> ifelse(x<0, zero(x), x)
    relu_diff = x -> ifelse(x<0, zero(x), one(x))

    W(A, B, m, k) = (1 - m)*I - A'*A + B - B'
    f(z::Vector, x::Vector, W, U) = relu.(W*z + U*x)
    f(z::Matrix, x::Matrix, W, U) = relu.(z*W' + x*U')
    
    function f!(Z::Matrix, X::Matrix, W, U)
        Z = relu.(Z*W' + X*U')
    end

    g(z::Vector, x::Vector, W, U) = z - f(z, x, W, U)

    z_star(x::SubArray, z0::SubArray, W, U) = fixed_point(z -> f(z, x, W, U), z0, Algorithm = :Anderson).FixedPoint_
    z_star(x::Vector, z0::Vector, W, U) = fixed_point(z -> f(z, x, W, U), z0, Algorithm = :Anderson).FixedPoint_
    function z_star(X::Matrix, Z0::Matrix, W::Matrix, U::Matrix)
        N = size(Z0, 1)
        # function _f!(z)
        #     z = f(z, x, W, U)
        # end
        Z_star = Z0
        @inbounds @views for n = 1:N
            Z_star[n, :] .= z_star(Vector(X[n, :]), Vector(Z_star[n, :]), W, U)
        end
        return Z_star
    end
    # function z_star(X::Matrix, Z0::Matrix, W, U)
    #     @cast Z_star[i, ]
    # end

    y(x::Vector, z0::Vector, W, U, u) = u'*z_star(x, z0, W, U)

    function fixedpoint_bwd(V0::Matrix, U, S::Matrix, W::Matrix)
        function _f!(V)
            V = U + (V.*S)W
        end
        fixedpoint(_f!, V0, method = :anderson).zero
    end
end