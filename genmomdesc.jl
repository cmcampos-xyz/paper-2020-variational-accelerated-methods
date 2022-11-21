################################################################################
#
# Copyright (C) 2020-2021 Cédric M. Campos <https://cmcampos.xyz>
#                         Alejandro Mahillo
#                         David Martín de Diego
#
# This file is part of the work "Discrete Variational Calculus for Accelerated
# Optimization" <https://arxiv.org/abs/2106.02700>
#
# Except where otherwise noted, this file and the whole work are licensed under
# a Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
# You should have received a copy of the license along with this work. If not,
# see <http://creativecommons.org/licenses/by-sa/4.0/>.
#
################################################################################

# ∇f: gradient function
# y0: initial search point
# y: end point OR
#    search trajectory for which
#    y[:,1]: initial search point
# μ: momentum coefficient
# η: learning rate

using LinearAlgebra: I, norm
using NLsolve: nlsolve

## Polyak's Heavy Ball (aka Classical Momentum)
function phb(
    ∇f::Function,
    x0::Union{Real,Vector{<:Real}};
    μ::Function=k->0.9,
    η::Function=k->0.01,
    epochs::Integer=100
    )

    Δx = - η(1)*∇f(x0)
    x = x0 + Δx
    for k in 2:epochs
        Δx = μ(k)*Δx - η(k)*∇f(x)
        x = x + Δx
    end
    return x
end
cm = phb

function phb!(
    ∇f::Function,
    x::Matrix{<:Real};
    μ::Function=k->0.9,
    η::Function=k->0.01
    )
    epochs = size(x,2)-1
    epochs == 0 && return

    Δx = - η(1)*∇f(x[:,1])
    x[:,2] = x[:,1] + Δx
    for k in 2:epochs
        Δx = μ(k)*Δx - η(k)*∇f(x[:,k])
        x[:,k+1] = x[:,k] + Δx
    end
end
cm! = phb!

## Nesterov Accelerated Gradient
function nag(
    ∇f::Function,
    x0::Union{Real,Vector{<:Real}};
    μ::Function=k->0.9,
    η::Function=k->0.01,
    epochs::Integer=100
    )

    x = x0
    y0 = x
    for k in 1:epochs
        y1 = x-η(k)*∇f(x)
        x = y1 + μ(k)*(y1-y0)
        y0 = y1
    end
    return x
end

function nag!(
    ∇f::Function,
    x::Matrix{<:Real};
    μ::Function=k->0.9,
    η::Function=k->0.01
    )
    epochs = size(x,2)-1

    y0 = x[:,1]
    for k in 1:epochs
        y1 = x[:,k]-η(k)*∇f(x[:,k])
        x[:,k+1] = y1 + μ(k)*(y1-y0)
        y0 = y1
    end
end

## Generalized Momemtum-Descent
function gmd(
    ∇f::Function,
    x0::Union{Real,Vector{<:Real}};
    μ::Function=k->0.9,
    η::Function=k->0.01,
    ε::Bool=true,
    epochs::Integer=100
    )

    Δx = zero(x0)
    x = x0
    for k in 1:epochs
        Δx *= μ(k)
        Δx -= η(k)*∇f(ε ? x : x + Δx)
        x += Δx
    end
    return x
end

function gmd!(
    ∇f::Function,
    x::Matrix{<:Real};
    μ::Function=k->0.9,
    η::Function=k->0.01,
    ε::Bool=true
    )
    epochs = size(x,2)-1

    y0 = x[:,1]
    y1 = x[:,1] - η(1)*∇f(x[:,1])
    x[:,2] = y1 + μ(1)*(ε ? y1-y0 : zero(x[:,1]))
    for k in 2:epochs
        y0 = y1
        y1 = x[:,k] - η(k)*∇f(x[:,k])
        x[:,k+1] = y1 + μ(k)*(ε ? y1-y0 : x[:,k]-x[:,k-1])
    end
end

## Wibisono, Wilson, Jordan
 # PNAS 2016, 113 (47)
 # DOI: 10.1073/pnas.1614734113
function wwj!(
    ∇f::Function,
    x::Matrix{<:Real},
    δ::Real
    )
    epochs = size(x,2)-1
    epochs == 0 && return

    # Consts. meeting conds. of Coro. 2.5 for p = 2.
    p = 2; N = 2; C = 2^-4

    ϵ = δ^p
    zk = x[:,1]
    for k in 1:epochs
        xk = x[:,k]
        yk = xk - 1/N*ϵ*∇f(xk)
        zk = zk - C*p*k*ϵ*∇f(yk)
        x[:,k+1] = p/(k+p) * zk  + (k/(k+p)) * yk
    end
end

function wwj(
    ∇f::Function,
    x0::Vector{<:Real},
    δ::Real,
    epochs::Integer
    )
    epochs = abs(epochs)
    epochs <= 1 && return x0

    x = Matrix{eltype(x0)}(undef,length(x0),epochs)
    x[:,1] = x0
    wwj!(∇f,x,δ)
    return x[:,end]
end

function wwj!(
    ∇f::Function,
    ∇2f::Function,
    x::Matrix{<:Real},
    δ::Real
    )
    epochs = size(x,2)-1
    epochs == 0 && return

    # Consts. meeting conds. of Coro. 2.5 for p = 3.
    p = 3; N = 2; C = 2^-8

    ϵ = δ^p
    zk = x[:,1]
    for k in 1:epochs
        xk = x[:,k]
        ∇fxk = ∇f(xk)
        ∇2fxk = ∇2f(xk)
        function f!(F,v)
           F[:] = ∇fxk + (∇2fxk+N/ϵ*norm(v)*I)*v
        end
        function j!(J,v)
           n = norm(v)
           J[:,:] = ∇2fxk+N/ϵ*(v/n*v'+n*I)
        end
        yk = xk + nlsolve(f!,j!,-∇2fxk\∇fxk,ftol=1e-8).zero
        zk = zk - C*p*k^2*ϵ*∇f(yk)
        x[:,k+1] = p/(k+p) * zk  + (k/(k+p)) * yk
    end
end

function wwj(
    ∇f::Function,
    ∇2f::Function,
    x0::Vector{<:Real},
    δ::Real,
    epochs::Integer
    )
    epochs = abs(epochs)
    epochs <= 1 && return x0

    x = Matrix{eltype(x0)}(undef,length(x0),epochs)
    x[:,1] = x0
    wwj!(∇f,∇2f,x,δ)
    return x[:,end]
end

nothing;
