using LinearAlgebra: diagm, Tridiagonal, SymTridiagonal

rosenbrock(x::Real,y::Real;a::Real=1.0,b::Real=100.0) = (a-x)^2+b*(y-x^2)^2
rosenbrock(x::Vector{<:Real}) = sum((1.0.-x[1:end-1]).^2+100.0*(x[2:end]-x[1:end-1].^2).^2)

function ∇rosenbrock(x::Real,y::Real;a::Real=1.0,b::Real=100.0)
  z = 2.0*b*(y-x^2)
  return [2.0*(x.*(1.0.-z)-a),z]
end

function ∇rosenbrock(x::Vector{<:Real})
  v = x[1:end-1].^2-x[2:end]
  ∇r = [400.0*x[1:end-1].*v+2.0*(x[1:end-1].-1.0);0.0]
  ∇r -= [0.0;200.0*v]
  return ∇r
end

function ∇2rosenbrock(x::Real,y::Real;a::Real=1.0,b::Real=100.0)
  ∇2r = Array{Float64}(undef,2,2)
  ∇2r[1,1] = 4.0*b*(3.0*x^2 - y) + 2.0
  ∇2r[1,2] = -4.0*b*x
  ∇2r[2,1] = ∇2r[1,2]
  ∇2r[2,2] = 2.0*b
  return ∇2r
end

function ∇2rosenbrock(x::Vector{<:Real};typeout::Union{Type{Matrix},Type{Tridiagonal},Type{SymTridiagonal}}=Matrix)
  d1 = -400.0*x
  d0 = [ 1200.0*x[1:end-1].^2 + d1[2:end] ; 200.0 ]
  d0[1] += 2.0
  d0[2:end-1] .+= 202.0
  d1 = d1[1:end-1]
  typeout === Matrix &&  return diagm(-1 => d1, 0 => d0, 1 => d1)
  typeout === Tridiagonal && return Tridiagonal(d1, d0, d1)
  typeout === SymTridiagonal && return SymTridiagonal(d0, d1)
end
