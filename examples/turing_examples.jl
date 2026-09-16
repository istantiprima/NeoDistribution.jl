# Bayesian modeling with NeoDistribution.jl + Turing.jl
#
# Requires Turing (and, transitively, Bijectors + DynamicPPL), which triggers
# the NeoDistributionTuringExt extension. Run with:
#
#     julia --project -e 'using Pkg; Pkg.add("Turing")'
#     julia --project examples/turing_examples.jl

using NeoDistribution
using Turing
using Random

Random.seed!(2024)

# ----------------------------------------------------------------------
# 1. Simulate skewed data from MSNBurr
# ----------------------------------------------------------------------
true_μ, true_σ, true_α = 1.0, 2.0, 3.0
y = rand(MSNBurr(true_μ, true_σ, true_α), 200)

# ----------------------------------------------------------------------
# 2. Model: MSNBurr likelihood with priors on μ, σ, α
# ----------------------------------------------------------------------
@model function msnburr_model(y)
    μ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 3), 0, Inf)
    α ~ truncated(Normal(1, 2), 0.1, 20)

    for i in eachindex(y)
        y[i] ~ MSNBurr(μ, σ, α)
    end
end

chain = sample(msnburr_model(y), NUTS(), 1000)
display(chain)

# ----------------------------------------------------------------------
# 3. Bayesian regression with MSNBurr-IIa errors
# ----------------------------------------------------------------------
x = randn(200)
y_reg = 2.0 .+ 1.5 .* x .+ rand(MSNBurr2a(0.0, 1.0, 2.0), 200)

@model function robust_regression(x, y)
    intercept ~ Normal(0, 10)
    slope ~ Normal(0, 5)
    σ ~ truncated(Normal(0, 2), 0, Inf)
    α ~ truncated(Normal(1, 1), 0.1, 10)

    for i in eachindex(y)
        μᵢ = intercept + slope * x[i]
        y[i] ~ MSNBurr2a(μᵢ, σ, α)
    end
end

chain_reg = sample(robust_regression(x, y_reg), NUTS(), 1000)
display(chain_reg)
