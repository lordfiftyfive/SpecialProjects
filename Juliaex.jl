#using RxInfer, Random
using Graphs
using MaxEntropyGraphs

#n = 500  # Number of coin flips
#p = 0.75 # Bias of a coin

#distribution = Bernoulli(p) 
#dataset      = float.(rand(Bernoulli(p), n))
#G = smallgraph(:karate)
#model = CReM(G)
#model = UBCM(G)
using RadiiPolynomial

function f(x)
    a, ϵ = 5, 1
    γ, u₁, u₂ = x
    return [u₁*(u₁ - a)*(1 - u₁) - u₂, ϵ*(u₁ - γ*u₂)]
end

function Df(x)
    a, ϵ = 5, 1
    γ, u₁, u₂ = x
    return [0 a*(2u₁-1)+(2-3u₁)*u₁ -1 ; -ϵ*u₂ ϵ -ϵ*γ]
end

import LinearAlgebra: ⋅

F(x, v, w) = [(x - w) ⋅ v ; f(x)]

DF(x, v) = [transpose(v) ; Df(x)]

import LinearAlgebra: nullspace

x_init = [2, 1.129171306613029, 0.564585653306514] # initial point on the branch of equilibria

v = vec(nullspace(Df(x_init))) # initial tangent vector

δ = 5e-2 # step size

w = x_init + δ * v # predictor

x_final, success = newton(x -> (F(x, v, w), DF(x, v)), w)

#computer assisted proof

R = 1e-1

x₀_interval = Interval.(x_init) .+ Interval(0.0, 1.0) .* (Interval.(x_final) .- Interval.(x_init))
x₀R_interval = Interval.(inf.(x₀_interval .- R), sup.(x₀_interval .+ R))

F_interval = F(x₀_interval, v, x₀_interval)
F_interval[1] = 0 # the first component is trivial by definition
DF_interval = DF(x₀_interval, v)

A = inv(mid.(DF_interval))

Y = norm(Sequence(A * F_interval), Inf)

Z₁ = opnorm(LinearOperator(A * DF_interval - I), Inf)

a, ϵ = 5, 1
Z₂ = opnorm(LinearOperator(Interval.(A)), Inf) * max(2abs(a) + 2 + 6(abs(x₀_interval[2]) + R), 2abs(ϵ))

showfull(interval_of_existence(Y, Z₁, Z₂, R))