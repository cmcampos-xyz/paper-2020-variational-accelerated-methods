using Base: reduce_empty_iter
using DataFrames, MLDatasets
using Flux: Dense, Chain, normalise, onecold, onehotbatch
using Statistics: mean, std
using StatsBase: sample

include("LossFunctions.jl")
using .LossFunctions: ∇mse_log_aff, ∇ll_log_aff, ∇2ll_log_aff

include("MomentumDescent.jl")
using .MomentumDescent: gmd, wwj

function getdata()
    ## (down)load data
    features = MLDatasets.Iris(as_df=false).features
    targets = vec(MLDatasets.Iris(as_df=false).targets)
    ## normalise features and compute targets
    μ = vec(mean(features, dims=2))
    σ = vec(std(features, mean=μ, dims=2))
    features = (features .- μ) ./ σ
    labels = sort(unique(targets))
    targets = onehotbatch(targets, labels)
    targets = 1.0 * targets
    return features, targets
end

function randsplit(data)
    ## split data randomly between training and testing data
    features, targets = data
    samples = size(features, 2)
    totrain = samples > 150 ? 100 : Integer(round(2 * samples / 3))
    train_indices = sort(sample(1:samples, totrain, replace=false))
    test_indices = setdiff(1:samples, train_indices)
    train = (features[:, train_indices], targets[:, train_indices])
    test = (features[:, test_indices], targets[:, test_indices])
    return train, test
end

function randweights(sz::Tuple{Int64,Int64})
    ## random weights
    weights = 10 * randn(sz)
    ## null biases
    weights[:, end] .= 0.0
    return weights
end

function train(data; optim::Function, loss::Symbol,
    init::Union{Nothing,Matrix{Float64}}=nothing,
    reduce::Bool=false)
    ## last class as discard class
    reduce && (data = (data[1], data[2][1:(end-1),:]))
    ## set sizes
    model_sz = (size(data[2],1), size(data[1],1)+1)
    ## set initial guess if needed
    isnothing(init) && (init = randweights(model_sz))
    ## set gradient loss
    ∇loss = eval(Symbol(:∇, loss))
    ∇f(WB) = vec(∇loss(data..., reshape(WB, model_sz)))
    ## optimize
    weights = reshape(optim(∇f, vec(init)), model_sz)
    ## null weights for discard class
    reduce && (weights = [weights; zeros(model_sz[2])'])
    return weights
end

accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

function confusion(x, y, model)
    ## [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
    ŷ = onehotbatch(onecold(model(x)), 1:size(y, 1))
    return Int64.(y * transpose(ŷ))
end

function test(model, data)
    accuracy_score = accuracy(data..., model)
    confusion_matrix = confusion(data..., model)
    return accuracy_score, confusion_matrix
end

function coefficients(s::Symbol=:constant,n::Integer=3)
    if s ∉ (:bounded, :unbounded)
        h = 0.9451430311574998
        μ₀ = (1.0+exp(-h))/(1.0+exp(h))
        η₀ = 2.0/(1.0+exp(h))*h^2 # == 0.5
        μ = _ -> μ₀
        η = _ -> η₀
    else
        n != 4 && (n = 3)
        n½ = n/2
        h = sqrt(0.5)
        if s == :bounded
            h² = h^2
            μ = k -> k/(k+n)
            η = k -> (k+n½)/(k+n)*h^2 #  → 0.5 (k→∞)
        elseif s == :unbounded
            p = n-1
            p_1 = n-2
            hᵖ = h^p
            λ = n == 3 ? 4/2^4 : 9/2^8
            μ = k -> k/(k+n)
            η = k -> λ*(k+n½)^p_1/(k+n)*hᵖ
        end
    end
    return μ, η
end

## experiments
numtest = 2^10
methods = [:constant, :bounded, :unbounded, :wwj]
nummeth = length(methods)
((@isdefined n) && n ∈ (3,4)) || (n = 4)
reduce = true
epochs = [25, 50, 75, 100, 150, 250]
data = getdata()
data_sz = (size(data[1], 1), size(data[2], 1))

acc_μ = Matrix{Float64}(undef, nummeth, length(epochs))
acc_σ = Matrix{Float64}(undef, nummeth, length(epochs))
conf_μ = Array{Float64,4}(undef, data_sz[2], data_sz[2], nummeth, length(epochs))
conf_σ = Array{Float64,4}(undef, data_sz[2], data_sz[2], nummeth, length(epochs))
accuracies = Matrix{Float64}(undef, nummeth, numtest)
confusions = Array{Float64,4}(undef, data_sz[2], data_sz[2], nummeth, numtest)
model_sz = (data_sz[2] - reduce, data_sz[1] + 1)
model_dim = prod(model_sz)
for (e, epochs) in enumerate(epochs)
    Threads.@threads for t in 1:numtest
        data_test, data_train = randsplit(data)
        randw = randweights(model_sz)
        for (i, method) in enumerate(methods)
            if method != :wwj
                μ, η = coefficients(method,n)
                optim = (∇f, x0) -> gmd(∇f, x0, μ=μ, η=η, epochs=epochs)
            else
                ∇2f(WB) = reshape(∇2ll_log_aff(data_train..., reshape(WB, model_sz)), model_dim, model_dim)
                if n != 4
                    optim = (∇f, x0) -> wwj(∇f, x0, 0.5, epochs)
                else
                    optim = (∇f, x0) -> wwj(∇f, ∇2f, x0, 0.5, epochs)
                end
            end
            weights = train(data_train, loss=:ll_log_aff, init=randw, optim=optim, reduce=reduce)
            model = Chain(Dense(weights[:, 1:(end-1)], weights[:, end]))
            accuracies[i, t], confusions[:, :, i, t] = test(model, data_test)
        end
    end
    acc_μ[:, e] = vec(mean(accuracies, dims=2))
    acc_σ[:, e] = vec(std(accuracies, mean=acc_μ[:, e], dims=2))
    conf_μ[:, :, :, e] = reshape(mean(confusions, dims=4), 3, 3, nummeth)
    conf_σ[:, :, :, e] = reshape(std(confusions, mean=conf_μ[:, :, :, e], dims=4), 3, 3, nummeth)
end

# display(round.(100*acc_μ,digits=1))
# display(round.(100*acc_σ,digits=1))
# display(round.(conf_μ,digits=1))
# display(round.(conf_σ,digits=1))

using Plots;# pgfplotsx()
plot(
    100*acc_μ',
    xticks=(1:length(epochs), epochs),
    xlabel="epochs",
    ylabel="accuracy (%)",
    label=["NAG/constant" "NAG/bounded (n = $n)" "NAG/unbounded (n = $n)" "WWJ (n = $n)"],
    legend=:bottomright,
    seriescolor=[3 1 4 7],
    markershape=:circle,markerstrokecolor=:auto)
