# * Sigmoid activation function (logistic function)
σ(z::Real) = 1.0 / (1.0 + exp(-z))
∇σ(z::Real) = σ(z) * (1.0 - σ(z))
# ** Squared error
# *** 1 target
mse(ŷ::Real, y::Real) = (ŷ - y)^2 / 2.0
# *** n targets (no mean)
mse(ŷ::Vector{<:Real}, y::Vector{<:Real}) = sum((ŷ - y) .^ 2) / 2.0
# * 1 experiment
# ** 1/m features to 1 target
σ(x::T, w::T, b::Real) where {T<:Union{Real,Vector{<:Real}}} = σ(w' * x + b)
∇σ(x::T, w::T, b::Real) where {T<:Union{Real,Vector{<:Real}}} = ∇σ(w' * x + b) * [x; 1.0]
mse(x::T, w::T, b::Real, y::Real) where {T<:Union{Real,Vector{<:Real}}} = mse(σ(x, w, b), y)
function ∇mse(x::T, w::T, b::Real, y::Real) where {T<:Union{Real,Vector{<:Real}}}
    σz = σ(x, w, b)
    return ((σz - y) * σz * (1.0 - σz)) * [x; 1.0]
end
# ** 1 feature to n targets
σ(x::Real, w::Vector{<:Real}, b::Vector{<:Real}) = σ.(w * x + b)
# Non-zero elements of ∇σ. Should be an n×2n matrix with ∇σ in its "diagonal".
∇σ(x::Real, w::Vector{<:Real}, b::Vector{<:Real}) = reduce(hcat, [∇σ(x, w[j], b[j]) for j = 1:length(b)])'
mse(x::Real, w::Vector{<:Real}, b::Vector{<:Real}, y::Vector{<:Real}) = mse(σ(x, w, b), y)
function ∇mse(x::Real, w::Vector{<:Real}, b::Vector{<:Real}, y::Vector{<:Real})
    return reduce(hcat, [∇mse(x, w[j], b[j], y[j]) for j = 1:length(b)])'
end
# ** m features to n targets
σ(x::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}) = σ.(w * x + b)
# Non-zero elements of ∇σ. Should be an n×(m+1)n matrix with ∇σ in its "diagonal".
∇σ(x::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}) = reduce(hcat, [∇σ(x, w[j, :], b[j]) for j = 1:length(b)])'
mse(x::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}, y::Vector{<:Real}) = mse(σ(x, w, b), y)
function ∇mse(x::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}, y::Vector{<:Real})
    return reduce(hcat, [∇mse(x, w[j, :], b[j], y[j]) for j = 1:length(b)])'
end
# * p experiments
# ** 1 feature to 1 target
 mse(x::Vector{<:Real}, w::Real, b::Real, y::Vector{<:Real}) = sum(k ->  mse(x[k], w, b, y[k]), 1:length(y)) / length(y)
∇mse(x::Vector{<:Real}, w::Real, b::Real, y::Vector{<:Real}) = sum(k -> ∇mse(x[k], w, b, y[k]), 1:length(y)) / length(y)
 mse(x::Vector{<:Real}, y::Vector{<:Real}) = wb::Vector{<:Real} ->  mse(x, wb[1], wb[2], y)
∇mse(x::Vector{<:Real}, y::Vector{<:Real}) = wb::Vector{<:Real} -> ∇mse(x, wb[1], wb[2], y)
# ** m feature to 1 target
 mse(x::Matrix{<:Real}, y::Vector{<:Real}) = wb::Vector{<:Real} ->  mse(x, wb[1:(end-1)], wb[end], y)
∇mse(x::Matrix{<:Real}, y::Vector{<:Real}) = wb::Vector{<:Real} -> ∇mse(x, wb[1:(end-1)], wb[end], y)
# ** m features to n targets
 mse(x::Matrix{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}, y::Matrix{<:Real}) = sum(k ->  mse(x[:, k], w, b, y[:, k]), 1:size(y, 2)) / size(y, 2)
∇mse(x::Matrix{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}, y::Matrix{<:Real}) = sum(k -> ∇mse(x[:, k], w, b, y[:, k]), 1:size(y, 2)) / size(y, 2)
 mse(x::Matrix{<:Real}, y::Matrix{<:Real}) = wb::Matrix{<:Real} ->  mse(x, wb[:, 1:(end-1)], wb[:, end], y)
∇mse(x::Matrix{<:Real}, y::Matrix{<:Real}) = wb::Matrix{<:Real} -> ∇mse(x, wb[:, 1:(end-1)], wb[:, end], y)

# * 1 experiment
# ** 1 feature to 1 target
function ∇2mse(x::Real, w::Real, b::Real, y::Real)
    ŷ = σ(w * x + b)
    ∇2 = [((2.0 * (1.0 + y) - 3.0 * ŷ) * ŷ - y) * ŷ * (1.0 - ŷ) * [x^2; x; 1.0]; 0.0]
    ∇2[4] = ∇2[3]
    ∇2[3] = ∇2[2]
    return reshape(∇2, 2, 2)
end
∇2mse(x::Real, y::Real) = wb::Vector{<:Real} -> ∇2mse(x, wb[1], wb[2], y)
# * p experiments
# ** 1 feature to 1 target
function ∇2mse(x::Vector{<:Real}, w::Real, b::Real, y::Vector{<:Real})
    ŷ = σ.(w * x .+ b)
    ŷ = ((2.0 * (1.0 .+ y) - 3.0 * ŷ) .* ŷ - y) .* ŷ .* (1.0 .- ŷ)
    ∇2 = Matrix{Float64}(undef, 2, 2)
    ∇2[1, 1] = ŷ' * (x .^ 2)
    ∇2[2, 1] = ŷ' * x
    ∇2[1, 2] = ∇2[2, 1]
    ∇2[2, 2] = sum(ŷ)
    return ∇2
end
∇2mse(x::Vector{<:Real}, y::Vector{<:Real}) = wb::Vector{<:Real} -> ∇2mse(x, wb[1], wb[2], y)
