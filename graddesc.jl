using LinearAlgebra: norm

function gd(∇f::Function, x0::Vector{<:Real}; η::Real=0.01, xtol::Real=1e-6, maxiter::Integer=1000)
    x = x0
    for k in 1:maxiter-1
        Δx = η * ∇f(x[:, k])
        x -= Δx
        norm(Δx, Inf) < xtol && break
    end
    return x
end

function gd!(∇f::Function, x::Matrix{<:Real}; η::Real=0.01, xtol::Real=0.0)
    maxiter = size(x,2)
    iter = 0
    for k in 1:maxiter-1
        Δx = η*∇f(x[:, k])
        x[:, k+1] = x[:, k] - Δx
        if norm(Δx, Inf) < xtol
            iter = k + 1
            break
        end
    end
    return iter
end
