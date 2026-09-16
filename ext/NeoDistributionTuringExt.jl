"""
    NeoDistributionTuringExt

Extension module that provides Turing.jl compatibility for NeoDistribution distributions.

This extension enables MSNBurr and MSNBurr-IIa distributions to be used directly
in Turing probabilistic models for Bayesian inference.

A correctly-implemented `ContinuousUnivariateDistribution` over the whole real line
already works out-of-the-box as a *likelihood* in Turing:

```julia
@model function m(y)
    # ...
    y[i] ~ MSNBurr(μ, σ, α)      # uses logpdf — nothing else needed
end
```

The only thing Turing needs beyond `logpdf`/`rand` is a *bijector*: the transform
mapping the (constrained) support to ℝ for unconstrained HMC/NUTS sampling. This
matters when one of these distributions is placed on a *latent* variable
(e.g. `x ~ MSNBurr(...)`), not just on observed data.

Both MSNBurr and MSNBurr-IIa are supported on all of ℝ, so the correct transform is
the identity. (Bijectors auto-derives this from the infinite support, but we state
it explicitly so it is robust and self-documenting.)

# Examples

```julia
using NeoDistribution
using Turing

# Define a Bayesian regression model with MSNBurr errors
@model function msnburr_regression(x, y)
    # Priors
    intercept ~ Normal(0, 10)
    slope ~ Normal(0, 5)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    α ~ truncated(Normal(1, 1), 0.1, 10)

    # Likelihood with MSNBurr errors
    μ = intercept .+ slope .* x
    for i in eachindex(y)
        y[i] ~ MSNBurr(μ[i], σ, α)
    end
end

# Fit the model
model = msnburr_regression(x_data, y_data)
chain = sample(model, NUTS(), 1000)
```
"""
module NeoDistributionTuringExt

using NeoDistribution
using Turing
using DynamicPPL
import Bijectors

# Both MSNBurr and MSNBurr-IIa are supported on all of ℝ, so the correct
# unconstraining transform for latent variables is the identity.
Bijectors.bijector(::MSNBurr)   = identity
Bijectors.bijector(::MSNBurr2a) = identity

end # module
