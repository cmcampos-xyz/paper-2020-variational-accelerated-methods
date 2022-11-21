yatf(x::Real,y::Real) = sin(.5*x^2-.25*y^2+3.0)*cos(2.0*x+1.0-exp(y))

function ∇yatf(x::Real,y::Real)
  α = .5*x^2-.25*y^2+3.0
  β = 2.0*x+1.0-exp(y)
  ∇ = cos(α)*cos(β)*[x,-.5*y]+sin(α)*sin(β)*[-2.0,exp(y)]
  return ∇
end

function ∇2yatf(x::Real,y::Real)
  expy = exp(y)

  α = .5*x^2-.25*y^2+3.0
  β = 2.0*x+1.0-expy

  sinα, cosα = sin(α), cos(α)
  sinβ, cosβ = sin(β), cos(β)

  ∇2 = Vector{Float64}(undef,4)
  ∇2[1:3] = sinα*cosβ*[-x^2-4.0,.5*x*y+2.0*expy,-.25*y^2-expy^2] +
            cosα*sinβ*[-4.0*x,y+x*expy,-y*expy]
  ∇2[1] += cosα*cosβ
  ∇2[3] += -.5*cosα*cosβ + expy*sinα*sinβ
  ∇2[4] = ∇2[3]
  ∇2[3] = ∇2[2]
  ∇2 = reshape(∇2,2,2)
  return ∇2
end
