#using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq,Plots
#import ModelingToolkit: Interval, infimum, supremum
#using Flux#: Dense, Chain, sigmoid_fast
"""
@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [Dtt(u1(t, x)) ~ Dxx(u1(t, x)) + u3(t, x) * sin(pi * x),
    Dtt(u2(t, x)) ~ Dxx(u2(t, x)) + u3(t, x) * cos(pi * x),
    0.0 ~ u1(t, x) * sin(pi * x) + u2(t, x) * cos(pi * x) - exp(-t)]

bcs = [u1(0, x) ~ sin(pi * x),
    u2(0, x) ~ cos(pi * x),
    Dt(u1(0, x)) ~ -sin(pi * x),
    Dt(u2(0, x)) ~ -cos(pi * x),
    u1(t, 0) ~ 0.0,
    u2(t, 0) ~ exp(-t),
    u1(t, 1) ~ 0.0,
    u2(t, 1) ~ -exp(-t)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# Neural network
input_ = length(domains)
n = 15
chain = [Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) for _ in 1:3]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)

@named pdesystem = PDESystem(eqs, bcs, domains, [t, x], [u1(t, x), u2(t, x), u3(t, x)])
prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 1)#000)

#phi = discretization.phi

"""

#above code works 
"""
@parameters t, σ_, β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]
dt = 0.01

input_ = length(domains)
n = 8
chain1 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
chain2 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
chain3 = Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, n, Lux.σ),
                   Dense(n, 1))
Chain(
    layer_1 = Dense(1 => 8, sigmoid_fast),  # 16 parameters
    layer_2 = Dense(8 => 8, sigmoid_fast),  # 72 parameters
    layer_3 = Dense(8 => 8, sigmoid_fast),  # 72 parameters
    layer_4 = Dense(8 => 1),            # 9 parameters
)         # Total: 169 parameters,
          #        plus 0 states.
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]
function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us, ts_]
end
data = getData(sol)

(u_, t_) = data
len = length(data[2])
init_params = [Flux.destructure(c)[1] for c in [chain1, chain2, chain3]]
acum = [0; accumulate(+, length.(init_params))]
sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]
(u_, t_) = data
len = length(data[2])

function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, θ[sep[i]]) .- u_[[i], :]) / len for i in 1:1:3)
end

discretization = NeuralPDE.PhysicsInformedNN([chain1, chain2, chain3],
                                             NeuralPDE.GridTraining(dt), param_estim = true,
                                             additional_loss = additional_loss)
@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t), y(t), z(t)], [σ_, ρ, β],defaults = Dict([p .=> 1.0 for p in [σ_, ρ, β]]))
prob = NeuralPDE.discretize(pde_system, discretization)
callback = function (p, l)
    
    return false
end
#res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 5000)
"""

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DiffEqBase
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
    u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]

# Space
x_0 = 0.0
x_end = 1.0
x_domain = Interval(x_0, x_end)
y_domain = Interval(0.0, 1.0)
domains = [x ∈ x_domain,
    y ∈ y_domain]

count_decomp = 10

# Neural network
af = Lux.tanh
inner = 10
chains = [Lux.Chain(Dense(2, inner, af), Dense(inner, inner, af), Dense(inner, 1))
          for _ in 1:count_decomp]
init_params = map(c -> Float64.(ComponentArray(Lux.setup(Random.default_rng(), c)[1])),
                  chains)

xs_ = infimum(x_domain):(1 / count_decomp):supremum(x_domain)
xs_domain = [(xs_[i], xs_[i + 1]) for i in 1:(length(xs_) - 1)]
domains_map = map(xs_domain) do (xs_dom)
    x_domain_ = Interval(xs_dom...)
    domains_ = [x ∈ x_domain_,
        y ∈ y_domain]
end

analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
function create_bcs(x_domain_, phi_bound)
    x_0, x_e = x_domain_.left, x_domain_.right
    if x_0 == 0.0
        bcs = [u(0, y) ~ 0.0,
            u(x_e, y) ~ analytic_sol_func(x_e, y),
            u(x, 0) ~ 0.0,
            u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
        return bcs
    end
    bcs = [u(x_0, y) ~ phi_bound(x_0, y),
        u(x_e, y) ~ analytic_sol_func(x_e, y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
    bcs
end

reses = []
phis = []
pde_system_map = []

for i in 1:count_decomp
    println("decomposition $i")
    domains_ = domains_map[i]
    phi_in(cord) = phis[i - 1](cord, reses[i - 1].minimizer)
    phi_bound(x, y) = phi_in(vcat(x, y))
    @register phi_bound(x, y)
    Base.Broadcast.broadcasted(::typeof(phi_bound), x, y) = phi_bound(x, y)
    bcs_ = create_bcs(domains_[1].domain, phi_bound)
    @named pde_system_ = PDESystem(eq, bcs_, domains_, [x, y], [u(x, y)])
    push!(pde_system_map, pde_system_)
    strategy = NeuralPDE.GridTraining([0.1 / count_decomp, 0.1])

    discretization = NeuralPDE.PhysicsInformedNN(chains[i], strategy;
                                                 init_params = init_params[i])

    prob = NeuralPDE.discretize(pde_system_, discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system_, discretization)
    res_ = Optimization.solve(prob, BFGS(), maxiters = 1000)
    phi = discretization.phi
    push!(reses, res_)
    push!(phis, phi)
end

function compose_result(dx)
    u_predict_array = Float64[]
    diff_u_array = Float64[]
    ys = infimum(domains[2].domain):dx:supremum(domains[2].domain)
    xs_ = infimum(x_domain):dx:supremum(x_domain)
    xs = collect(xs_)
    function index_of_interval(x_)
        for (i, x_domain) in enumerate(xs_domain)
            if x_ <= x_domain[2] && x_ >= x_domain[1]
                return i
            end
        end
    end
    for x_ in xs
        i = index_of_interval(x_)
        u_predict_sub = [first(phis[i]([x_, y], reses[i].minimizer)) for y in ys]
        u_real_sub = [analytic_sol_func(x_, y) for y in ys]
        diff_u_sub = abs.(u_predict_sub .- u_real_sub)
        append!(u_predict_array, u_predict_sub)
        append!(diff_u_array, diff_u_sub)
    end
    xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    u_predict = reshape(u_predict_array, (length(xs), length(ys)))
    diff_u = reshape(diff_u_array, (length(xs), length(ys)))
    u_predict, diff_u
end
dx = 0.01
u_predict, diff_u = compose_result(dx)

inner_ = 18
af = Lux.tanh
chain2 = Lux.Chain(Dense(2, inner_, af),
                   Dense(inner_, inner_, af),
                   Dense(inner_, inner_, af),
                   Dense(inner_, inner_, af),
                   Dense(inner_, 1))

init_params2 = Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain2)[1]))

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

losses = map(1:count_decomp) do i
    loss(cord, θ) = chain2(cord, θ) .- phis[i](cord, reses[i].minimizer)
end

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

prob_ = NeuralPDE.neural_adapter(losses, init_params2, pde_system_map,
                                 NeuralPDE.GridTraining([0.1 / count_decomp, 0.1]))
res_ = Optimization.solve(prob_, BFGS(); callback = callback, maxiters = 2000)
prob_ = NeuralPDE.neural_adapter(losses, res_.minimizer, pde_system_map,
                                 NeuralPDE.GridTraining([0.05 / count_decomp, 0.05]))
res_ = Optimization.solve(prob_, BFGS(); callback = callback, maxiters = 1000)

phi_ = NeuralPDE.get_phi(chain2)

xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_predict_ = reshape([first(phi_([x, y], res_.minimizer)) for x in xs for y in ys],
                     (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                 (length(xs), length(ys)))
diff_u_ = u_predict_ .- u_real

using Plots

p1 = plot(xs, ys, u_predict, linetype = :contourf, title = "predict 1");
p2 = plot(xs, ys, u_predict_, linetype = :contourf, title = "predict 2");
p3 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
p4 = plot(xs, ys, diff_u, linetype = :contourf, title = "error 1");
p5 = plot(xs, ys, diff_u_, linetype = :contourf, title = "error 2");
plot(p1, p2, p3, p4, p5)
"""
"""