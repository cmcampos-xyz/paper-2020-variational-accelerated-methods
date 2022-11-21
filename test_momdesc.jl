## includes
include("genmomdesc.jl")
include("rosenbrock.jl")
using LinearAlgebra: SymTridiagonal, norm
include("yatf.jl")
include("mse.jl")
include("LossFunctions.jl")
using .LossFunctions:
    ll_log_aff, ∇ll_log_aff, ∇2ll_log_aff,
    mse_log_aff, ∇mse_log_aff, ∇2mse_log_aff

## Plotting (default backend: GR)
using Plots
## uncomment below to change default backend
#pgfplots()  ##PGFPlots  <-- does not work!!! BUGGED???
#pgfplotsx() ##PGFPlotsX <-- recommended for .tex output
#plotlyjs()  ##PlotlyJS  <-- interactive, bugged subplots

## Defaults
(@isdefined problem) || (problem = 1)
plotdisplay = false
gentrajectoryplots = false
saveplots = false
plotfolder = "../figures/"
plotextensions = []
backend_name() in (:pgfplots, :pgfplotsx) && push!(plotextensions, ".tex")

## Set options HERE !!!
#problem = 2 ## (see problem list below)
#plotdisplay = true
#gentrajectoryplots = true
#saveplots = true
#append!(plotextensions,[".png", ".pdf", ".svg"])

Base.:*(a::Symbol,b::Symbol) = Symbol(a,b)
Base.:*(a::Symbol,b::Integer) = a*Symbol(b)

struct Simulation
    trajectory::Matrix{Float64}
    values::Vector{Float64}
    method::Symbol
    coefficients::Symbol
end

Simulation(f::Function,y::Matrix{Float64},m,c) = Simulation(copy(y),dropdims(mapslices(f, y; dims=1); dims=1),m,c)

simulations = Vector{Simulation}()
trajectory = Dict{Union{Symbol,Tuple},Matrix{Float64}}()
 objvalues = Dict{Union{Symbol,Tuple},Vector{Float64}}()
plots = Dict{Union{Symbol,Tuple},Plots.Plot}()

# problems
problem_names = [
    "Rosenbrock",          #1
    "YATF",                #2
    "High-dim quadratic",  #3
    "Logistic regression", #4
    "Cross-entropy loss"   #5
]
problem_nicks = [
    "ros",
    "yatf",
    "hdquad",
    "logreg",
    "logloss"
]
problem_ids = collect(1:5)
lookup_name = Dict(problem_ids .=> problem_names)
lookup_nick = Dict(problem_ids .=> problem_nicks)

print("\nSimulations for $(lookup_name[problem])\n\n")

## maximum order for polinomial dilation
n_max = 4 # 2 (none), 3 or 4

## problem setup
if lookup_nick[problem] == "ros"
    # dimensions
    dim = 30
    # function setup
    ∇0f =   rosenbrock
    ∇1f =  ∇rosenbrock
    ∇2f = ∇2rosenbrock
    # initial condition
    # for global minimum
    y0 = zeros(dim)
    # for local minimum (uncomment below)
    #y0 = [-1;ones(dim-1)]
    # simulations setup
    h = 0.01
    epochs = 20000
    # graph options
    xlim = [8e2, epochs]
    ylim = [1e-14, 1e2]
elseif lookup_nick[problem] == "yatf"
    # dimensions
    dim = 2
    # function setup
    ∇0f = ((x,y),) ->   yatf(x,y) + 0.6574000294758535
    ∇1f = ((x,y),) ->  ∇yatf(x,y)
    ∇2f = ((x,y),) -> ∇2yatf(x,y)
    # initial condition
    y0 = [-0.25, 0.35]
    # simulations setup
    h = 0.01
    epochs = 3800
    # graph options
    xlim = [8e2, epochs]
    ylim = [1e-16, 0.9]
elseif lookup_nick[problem] == "hdquad"
    # dimensions
    dim = 50
    # function setup
    ρ = 0.9
    Σ = [ρ^abs(i-j) for i = 1:dim, j = 1:dim]
    q = 1 / (1-ρ^2)
    d0 = [q; (1+ρ^2)*q*ones(dim-2); q]
    d1 = -ρ*q*ones(dim-1)
    iΣ = SymTridiagonal(d0, d1)
    ∇0f = x::Vector{<:Real} -> (x'*iΣ*x) / 2
    ∇1f = x::Vector{<:Real} ->     iΣ*x
    ∇2f = x::Vector{<:Real} ->     iΣ
    # initial condition
    y0 = 2*rand(dim).-1
    y0 *= 50/norm(y0)
    # simulations setup
    h = 0.1
    epochs = 10000
    # graph options
    xlim = [1e1, epochs]
    ylim = [1e-16, 3e5]
elseif lookup_nick[problem] == "logreg"
    # dimensions
    dim = 2
    # function setup
    X = [0.5, 2.5]
    Y = [0.2, 0.9]
    ∇0f =   mse(X,Y)
    ∇1f =  ∇mse(X,Y)
    ∇2f = ∇2mse(X,Y)
    # initial condition
    y0 = [1.0, 1.0]
    # simulations setup
    h = 0.1
    epochs = 50000
    # graph options
    xlim = [1e2, epochs]
    ylim = [1e-18, 1e0]
elseif lookup_nick[problem] == "logloss"
    # dimensions
    bias, reduce = true, true
    dimin, dimout = 4, 3
    dimin += bias
    dimout -= reduce
    dim = dimout*dimin
    samples = 10
    # function setup
    randomdata = false
    if randomdata
        X = round.(2.0*rand(dimin-1,samples).-1.0, digits=1)
        Y = zeros(dimout+reduce,samples)
        for (j,k) in [(rand(1:(dimout+reduce)),k) for k in 1:samples]
            Y[j,k] = 1.0
        end
        Y = Y[1:dimout,:]
    else
        X = [ 0.0  -0.3  -0.7  -0.1  -1.0   0.6   0.2   0.4   0.6   0.5
             -0.6  -0.8  -1.0   0.3  -0.8   0.7   0.5  -0.5  -0.5   0.8
              0.9  -0.1  -0.4   0.9  -0.1  -1.0   0.5   0.6  -0.3  -0.0
             -0.3  -0.3   1.0   0.3   0.3  -0.7  -0.8   0.4  -0.9   1.0]
        Y = [ 0.0  0.0  0.0  0.0  1.0  1.0  0.0  1.0  0.0  0.0
              0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0
              1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0]
        Y = Y[1:(end-reduce),:]
    end
    ∇0f = wb ->   ll_log_aff(X,Y,wb)
    ∇1f = wb ->  ∇ll_log_aff(X,Y,wb)
    ∇2f = wb -> ∇2ll_log_aff(X,Y,wb)
    # ∇0f = wb ->   mse_log_aff(X,Y,wb)
    # ∇1f = wb ->  ∇mse_log_aff(X,Y,wb)
    # ∇2f = wb -> ∇2mse_log_aff(X,Y,wb)
    # initial condition
    if randomdata
        wb = round.(2.0*rand(dimout,dimin).-1.0, digits=1)
        wb[:,end] .= 0.0
    else
        wb = [ 0.2   0.0  -0.3  -0.3  0.0
              -0.9   0.1   0.6  -0.4  0.0
               0.0   0.0   0.0   0.0  0.0]
        wb = wb[1:(end-reduce),:]
    end
    y0 = vec(wb)
    # simulations setup
    h = 0.9451430311574998
    epochs = 12000
    # graph options
    xlim = [1e1, epochs]
    ylim = [1e-7, Inf]
end

## Cofficients

# Constant (Exponential Dilation)
μΦ(h) = (1.0+exp(-h))/(1.0+exp(h))
ηΦ(h) = 2.0/(1.0+exp(h))*h^2
μc₀, ηc₀ = μΦ(h), ηΦ(h)
μc(k) = μc₀
ηc(k) = ηc₀

# Bounded (Potential Dilation)
simplify = false
if !simplify
    μb(k::Float64,n) = (k^n+(k-1)^n)/(k^n+(k+1)^n)
    ηb(k::Float64,n) = 2*k^n/(k^n+(k+1)^n)*h^2
    μb(k,n) = μb(Float64(k),n)
    ηb(k,n) = ηb(Float64(k),n)
else
    μb(k,n) = (k-n)/(k+n)
    ηb(k,n) = (2*k)/(2*k+n)*h^2
end

# Unbounded (Mod. Potential Dilation)
μu = μb
ηu(k,n,C) = (n-1)^2*C*(k*h)^(n-3)*ηb(k,n)

# Palindromic coefficients
μp(k,n) = (k/(k+1))^n
ηp(k,n,C) = (n-1)^2*C*((k+.5)*h)^(n-3)*ηb(k,2*n-3)

# 3-phase coefficients
μ3(k,n) = (k-n)/k
η3(k,n,C) = (n-1)^2*C*(k*h)^(n-3)*h^2

## simulations
y = Matrix{Float64}(undef,dim,epochs)
y[:,1] = y0

#************************    Constant coefficients    *************************#

# PHB with constant coefficients
phb!(∇1f,y;μ=μc,η=ηc)
push!(simulations,Simulation(∇0f,y,:phb,:c))
push!(trajectory,(:phb,:c) => simulations[end].trajectory)
push!(objvalues, (:phb,:c) => simulations[end].values)

# NAG with constant coefficients
nag!(∇1f,y;μ=μc,η=ηc)
push!(simulations,Simulation(∇0f,y,:nag,:c))
push!(trajectory,(:nag,:c) => simulations[end].trajectory)
push!(objvalues, (:nag,:c) => simulations[end].values)

for n = 3:n_max

    p = n-1
    C = 3^((p-2)/2) / (4^(p-1) * p^p)

    global μb(k) = μb(k,n)
    global ηb(k) = ηb(k,n)
    global ηu(k) = ηu(k,n,C)
    global μp(k) = μp(k,n)
    global ηp(k) = ηp(k,n,C)
    global μ3(k) = μ3(k,n)
    global η3(k) = η3(k,n,C)

    for c in (:b, :u)
        for m in (:phb, :nag)
            @time "$m$c (n = $n)" @eval $(m*:!)(∇1f, y; μ=$(:μ*c), η=$(:η*c))
            push!(simulations, Simulation(∇0f, y, m, c))
            push!(trajectory,(m, c, n) => simulations[end].trajectory)
            push!(objvalues, (m, c, n) => simulations[end].values)
        end
    end

    # WWJ 3-phase
    if p == 2
        @time "WWJ (n = $n)" wwj!(∇1f, y, h)
    elseif p == 3
        @time "WWJ (n = $n)" wwj!(∇1f, ∇2f, y, h)
    end
    push!(simulations,Simulation(∇0f,y,:wwj,:u))
    push!(trajectory,(:wwj,n) => simulations[end].trajectory)
    push!(objvalues, (:wwj,n) => simulations[end].values)
end

# * Plots

# PHB vs NAG with constant and bounded coefficients
plot(
    # title=lookup_name[problem],#*" (n = $n)",
    xlabel="epoch", ylabel="residual",
    xaxis=:log, yaxis=:log,
    xlim=xlim, ylim=ylim,
    legend=:bottomleft,
    color_palette=:Paired_10,
    size=(600, 400)
)
for n = 3:n_max
    plot!(objvalues[:phb,:b,n], label="PHB/bounded (n = $n) ", linecolor=8n-23) #1,9
    plot!(objvalues[:nag,:b,n], label="NAG/bounded (n = $n) ", linecolor=8n-22) #2,10
end
plot!(objvalues[:phb,:c], label="PHB/constant ", linecolor=3) #3
plot!(objvalues[:nag,:c], label="NAG/constant ", linecolor=4) #4
push!(plots,:phbvsnag => current())
plotdisplay && display(current())
if saveplots
    for ext in plotextensions
        savefig(plotfolder*lookup_nick[problem]*"PHBvsNAG"*ext)
    end
end

# NAG with "all" coefficients
for n = 3:n_max
    plot(
        #title=lookup_name[problem],
        xlabel="epoch", ylabel="residual",
        xaxis=:log, yaxis=:log,
        xlim=xlim, ylim=ylim,
        legend=:bottomleft,
#        size=(692, 462)
#        size=(768, 486)
#        size=(1024, 576)
    )
    plot!(objvalues[:nag,:c], label="NAG/constant ", linecolor=3)
    plot!(objvalues[:nag,:b,n], label="NAG/bounded (n = $n) ", linecolor=1)
    plot!(objvalues[:nag,:u,n], label="NAG/unbounded (n = $n) ", linecolor=4)
    plot!(objvalues[:wwj,n], label="WWJ (n = $n) ", linecolor=7)
    push!(plots,(:all,n) => current())
    plotdisplay && display(current())
    if saveplots
        for ext in plotextensions
            savefig(plotfolder*lookup_nick[problem]*"AllNAG$n"*ext)
        end
    end
end

for n = 3:n_max
    plot()
    plot!(objvalues[:nag,:c], label="NAG/constant ", linecolor=3)
    plot!(objvalues[:nag,:b,n], label="NAG/bounded (n = $n) ", linecolor=1)
    plot!(objvalues[:nag,:u,n], label="NAG/unbounded (n = $n) ", linecolor=4)
    plot!(objvalues[:wwj,n], label="WWJ (n = $n) ", linecolor=7)
    push!(plots,(:_all,n) => current())
end
if n_max == 4
    plot!(plots[:_all,3],xformatter=_->"")#,title=lookup_name[problem])
    plot!(plots[:_all,4],xlabel="epoch")
    plot(plots[:_all,3],plots[:_all,4],layout=(2,1))
    plot!(
        ylabel="residual",
        xaxis=:log, yaxis=:log,
        xlim=xlim, ylim=ylim,
        legend=:bottomleft,
        size=(720, 800)
#        size=(692, 808)
#        size=(768, 896)
#        size=(1024, 1024)
    )
    push!(plots,:all => current())
    plotdisplay && display(current())
    if saveplots
        for ext in plotextensions
            savefig(plotfolder*lookup_nick[problem]*"AllNAG"*ext)
        end
    end
end

if gentrajectoryplots && lookup_nick[problem] in ("ros","yatf")
    local n = 3
    if lookup_nick[problem] == "ros"
        rng = 1:epochs
        plot(
            #title=lookup_name[problem],
            xlabel="epoch", ylabel="y[k]",
            size=(600, 400))
        plot!(trajectory[:nag,:b,n][[1;20;29;30],rng]', labels=["y[1]" "y[20]" "y[29]" "y[30]"])
    elseif lookup_nick[problem] == "yatf"
        rng = 765:1450
        xmin, xmax = extrema([trajectory[:phb,:b,n][1,rng]; trajectory[:nag,:b,n][1,rng]])
        ymin, ymax = extrema([trajectory[:phb,:b,n][2,rng]; trajectory[:nag,:b,n][2,rng]])
        δx = 0.1*(xmax-xmin)
        δy = 0.1*(ymax-ymin)
        xmin, xmax = xmin-δx, xmax+δx
        ymin, ymax = ymin-δx, ymax+δx
        xrng = range(xmin, xmax, length=241)
        yrng = range(ymin, ymax, length=241)
        f(x,y) = log(yatf(x,y) + 0.6574000294758535)
        plot(
            #title=lookup_name[problem],
            xlabel="x", ylabel="y",
            legend=:topleft, color_palette=:Paired_10,
            st=:heatmap, grid=:none, colorbar=:none, c=:linear_grey_0_100_c0_n256,
            xrng, yrng, f,
            size=(600, 400))
        plot!(trajectory[:phb,:b,n][1,rng], trajectory[:phb,:b,n][2,rng], label="PHB/bounded (n = $n) ", color=1)
        plot!(trajectory[:nag,:b,n][1,rng], trajectory[:nag,:b,n][2,rng], label="NAG/bounded (n = $n) ", color=2)
    end
    push!(plots,:trajectory => current())
    plotdisplay && display(current())
    if saveplots
        for ext in plotextensions
            savefig(plotfolder*lookup_nick[problem]*ext)
        end
    end
end
