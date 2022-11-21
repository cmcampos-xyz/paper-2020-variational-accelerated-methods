"""
    LossFunctions

Collection of standard functions (losses, classifiers, and such) used in
Machine Learning.
"""
module LossFunctions

using LinearAlgebra: Diagonal, I

export
    logistic, ∇logistic, ∇2logistic,
    softmax, ∇softmax, ∇2softmax,
    mse, ∇mse,
    logloss, ∇logloss, ∇2logloss

# * Affine transformation
affine(x::T, w::T, b::Real) where {T<:Union{Real,Vector{<:Real}}} = b + w' * x
affine(x::Real, w::Vector{<:Real}, b::Vector{<:Real}) = b + w * x
affine(x::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}) = b + w * x
affine(x::Union{Real,Vector{<:Real}}, wb::Vector{<:Real}) = affine(x, length(wb) > 2 ? wb[1:(end-1)] : wb[1], wb[end])
affine(x::Vector{<:Real}, wb::Matrix{<:Real}) = affine(x, wb[:, 1:(end-1)], wb[:, end])

# * Classifiers

# Logistic function
# - Unimodal or standard
logistic(z::Real) = 1.0 / (1.0 + exp(-z))
function ∇logistic(z::Real)
    σ = logistic(z)
    return σ * (1.0 - σ)
end
function ∇2logistic(z::Real)
    σ = logistic(z)
    return σ*(1.0-σ)*(1.0-2.0*σ)
end
# - Multimodal or SoftMax
function logistic(z::Vector{<:Real}; reduce::Bool=true)
    y = exp.(z)
    return y / (reduce + sum(y))
end
function ∇logistic(z::Vector{<:Real}; reduce::Bool=true)
    σ = logistic(z;reduce=reduce)
    return Diagonal(σ)+(-σ)*(σ')
end
function ∇2logistic(z::Vector{<:Real}; reduce::Bool=true)
    σ = logistic(z;reduce=reduce)
    σ2 = σ*σ'
    dim = length(z)
    ∇2 = Array{Float64}(undef, dim, dim, dim)
    for k = 1:dim
        ∇2[:,:,k] = 2.0*σ[k]*σ2
    end
    for k = 1:dim
        ∇2[:,k,k] -= σ2[:,k]
        ∇2[k,:,k] -= σ2[:,k]
        ∇2[k,k,:] -= σ2[:,k]
    end
    for k = 1:dim
        ∇2[k,k,k] += σ[k]

    end
    return ∇2
end
for D in (Symbol(""),:∇,:∇2)
    Dsoftmax = Symbol(D, :softmax)
    Dlogistic = Symbol(D, :logistic)
    @eval $Dsoftmax(z::Vector{<:Real}) = $Dlogistic(z; reduce=false)
end

# * Loss Functions

# (Mean) Squarred Error
mse(ŷ::Real, y::Real) = (ŷ-y)^2 / 2.0
mse(ŷ::Vector{<:Real}, y::Vector{<:Real}) = sum((ŷ-y).^2) / 2.0
∇mse(ŷ::T, y::T) where {T<:Union{Real,Vector{<:Real}}} = ŷ-y
∇2mse(ŷ::T, y::T) where {T<:Union{Real,Vector{<:Real}}} = I(length(ŷ))

# Logistic cross entropy (log-loss)
logloss(ŷ::Vector{<:Real}, y::Vector{<:Real}) = -(y'*log.(ŷ))
∇logloss(ŷ::Vector{<:Real}, y::Vector{<:Real}) = -(y ./ ŷ)
∇2logloss(ŷ::Vector{<:Real}, y::Vector{<:Real}) = Diagonal(y ./ ŷ.^2)
logloss(ŷ::Vector{<:Real}, j::Integer) = -log(ŷ[j])
function ∇logloss(ŷ::Vector{<:Real}, j::Integer)
    ∇ = zeros(length(ŷ))
    ∇[j] = -1.0/ŷ[j]
    return ∇
end
function ∇2logloss(ŷ::Vector{<:Real}, j::Integer)
    ∇2 = Diagonal(zeros(length(ŷ)))
    ∇2[j,j] = 1.0/ŷ[j]^2
    return ∇2
end
for ∇ in (Symbol(""),:∇,:∇2)
    ∇logloss = Symbol(∇, :logloss)
    @eval $∇logloss(ŷ::Vector{<:Real}, y::Vector{<:Bool}) = $∇logloss(ŷ, findfirst(y))
end





function lossmodobjective(loss::Function, class::Function)
    ∇loss, ∇2loss = eval(Symbol(:∇, loss)), eval(Symbol(:∇2, loss))
    ∇class, ∇2class = eval(Symbol(:∇, class)), eval(Symbol(:∇2, class))
    objective(in::Vector{<:Real}, out::Vector{<:Real}, wb::Matrix{<:Real}) = loss(class(affine(in, wb)), out)
    function ∇objective(in::Vector{<:Real}, out::Vector{<:Real}, wb::Matrix{<:Real})
        x, y = in, out
        z = affine(x, wb)
        ŷ = class(z)
        return (∇class(z) * ∇loss(ŷ, y)) .* repeat([x; 1.0]', outer=(size(wb, 1), 1))
    end
    function ∇2objective(in::Vector{<:Real}, out::Vector{<:Real}, wb::Matrix{<:Real})
        x, y = in, out
        dimout, dimin = size(wb)
        z = affine(x, wb)
        ŷ = class(z)
        ∇ŷ = ∇class(z)
        ∇2ŷ = ∇2class(z)
        ∇l = ∇loss(ŷ, y)
        ∇2lc = ∇ŷ * ∇2loss(ŷ, y) * ∇ŷ + hcat((∇2ŷ[:, :, k] * ∇l for k = 1:dimout)...)
        ∇z2 = [x; 1.0] * [x; 1.0]' #reshape(kron([x; 1.0], [x; 1.0]), dimin, dimin)
        ∇2 = Array{Float64}(undef, dimout, dimin, dimout, dimin)
        for i = 1:dimout, j = 1:dimout
            ∇2[i, :, j, :] = ∇2lc[i, j] * ∇z2
        end
        return ∇2
    end
    for f in (:objective, :∇objective, :∇2objective)
        @eval $f(in::Matrix{<:Real}, out::Matrix{<:Real}, wb::Matrix{<:Real}) = sum(k -> $f(in[:, k], out[:, k], wb), 1:size(out, 2)) / size(out, 2)
    end
    return objective, ∇objective, ∇2objective
end

for func_name in (:mse, :logloss)
    for func in (func_name, Symbol(:∇, func_name))
        @eval begin
            ($func)(ŷ::Matrix{<:Real}, y::Matrix{<:Real}) = sum(k -> ($func)(ŷ[:, k], y[:, k]), 1:size(y, 2)) / size(y, 2)
        end
    end
end



# * Compositions

# Affine |> Identity |> MSE
mse_id_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}) = mse(b + w * x, y)
function ∇mse_id_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real})
    ŷ = b + w * x
    return ∇mse(ŷ, y) .* repeat([x; 1.0]', outer=(length(y), 1))
end

# Affine |> Logistic |> MSE
mse_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}) = mse(logistic(b + w * x), y)
function ∇mse_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real})
    z = b + w * x
    ŷ = logistic(z)
    return (∇logistic(z) * ∇mse(ŷ, y)) .* repeat([x; 1.0]', outer=(length(y), 1))
end
function ∇2mse_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real})
    dimout, dimin = size(w)
    dimin += 1
    z = b + w * x
    ŷ = logistic(z)
    ∇ŷ = ∇logistic(z)
    ∇2ŷ = ∇2logistic(z)
    ∇err = ∇mse(ŷ, y)
    ∇2err_log = ∇ŷ^2 + hcat((∇2ŷ[:, :, k] * ∇err for k = 1:dimout)...)
    ∇z2 = [x; 1.0] * [x; 1.0]' #reshape(kron([x; 1.0], [x; 1.0]), dimin, dimin)
    ∇2 = Array{Float64}(undef, dimout, dimin, dimout, dimin)
    for i = 1:dimout, j = 1:dimout
        ∇2[i, :, j, :] = ∇2err_log[i, j] * ∇z2
    end
    return ∇2
end

# Affine |> Logistic |> LogLoss
function ll_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}; reduce::Bool=true)
    z = w*x + b
    j = findfirst(y .== 1.0)
    return log(reduce+sum(exp.(z))) - (isnothing(j) ? 0.0 : z[j]) # better Bool
end
function ∇ll_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}; reduce::Bool=true)
    z = w*x + b
    return (logistic(z,reduce=reduce)-y) .* repeat([x;1.0]', outer=(size(w,1), 1))
end
function ∇2ll_log_aff(x::Vector{<:Real}, y::Vector{<:Real}, w::Matrix{<:Real}, b::Vector{<:Real}; reduce::Bool=true)
    dimout, dimin = size(w)
    dimin += 1
    z = w*x + b
    ∇2ll_log = ∇logistic(z,reduce=reduce)
    ∇z2 = [x;1.0] * [x; 1.0]'
    ∇2 = Array{Float64}(undef, dimout, dimin, dimout, dimin)
    for i = 1:dimout, j = 1:dimout
        ∇2[i,:,j,:] = ∇2ll_log[i,j] * ∇z2
    end
    return ∇2
end
for ∇ in (Symbol(""), :∇, :∇2)
    ∇in = Symbol(∇, :ll_log_aff)
    ∇out = Symbol(∇, :ll_sm_aff)
    @eval $∇out(args...) = $∇in(args...; reduce=false)
end


# Params as vectors

for func_name in (:mse_id_aff, :mse_log_aff, :ll_log_aff, :ll_sm_aff)
    func = func_name
    @eval begin
        ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Matrix{<:Real}; kwds...) = sum(k -> ($func)(x[:, k], y[:, k], wb[:, 1:(end-1)], wb[:, end]), 1:size(y, 2); kwds...) / size(y, 2)
        ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Vector{<:Real}; kwds...) = ($func)(x, y, reshape(wb, size(y, 1), size(x, 1) + 1); kwds...)
    end
    func = Symbol(:∇, func_name)
    @eval begin
        ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Matrix{<:Real}) = sum(k -> ($func)(x[:, k], y[:, k], wb[:, 1:(end-1)], wb[:, end]), 1:size(y, 2)) / size(y, 2)
        ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Vector{<:Real}) = vec(($func)(x, y, reshape(wb, size(y, 1), size(x, 1) + 1)))
    end
    func = Symbol(:∇2, func_name)
    @eval begin
        function ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Matrix{<:Real})
            samples = min(size(x, 2), size(y, 2))
            return sum(($func)(x[:, k], y[:, k], wb[:, 1:(end-1)], wb[:, end]) for k in 1:samples) / samples
        end
        function ($func)(x::Matrix{<:Real}, y::Matrix{<:Real}, wb::Vector{<:Real})
            dimin, dimout = size(x, 1), size(y, 1)
            dimin += 1 # bias
            dim = dimin * dimout
            reshape(($func)(x, y, reshape(wb, dimout, dimin)), dim, dim)
        end
    end
end


end
