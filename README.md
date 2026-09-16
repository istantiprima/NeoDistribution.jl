# NeoDistribution.jl

A Julia package implementing **Neo-Normal distributions** for flexible Bayesian inference. This package provides implementations of distributions that generalize the normal distribution to accommodate skewness.

## Features

- **MSNBurr Distribution**: Modified to be Stable Normal from Burr distribution (left-skew neo-normal)
- **MSNBurr-IIa Distribution** (`MSNBurr2a`): mirror image of MSNBurr (right-skew neo-normal)
- Full compatibility with `Distributions.jl` interface
- AD-friendly, numerically stable `logpdf`/`logcdf` (ready for NUTS/HMC)
- Seamless integration with `Turing.jl` for Bayesian modeling
- Analytical formulas for moments (mean, variance, skewness, kurtosis)
- Efficient inverse-CDF random number generation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/istantiprima/NeoDistribution.jl")
```

Or in development mode:
```julia
] dev /path/to/NeoDistribution.jl
```

## Quick Start

```julia
using NeoDistribution

# Create distributions
d_msnburr   = MSNBurr(0.0, 1.0, 2.0)    # μ, σ, α
d_msnburr2a = MSNBurr2a(0.0, 1.0, 2.0)  # μ, σ, α

# Calculate PDF and CDF
pdf(d_msnburr, 0.5)
cdf(d_msnburr2a, 1.0)

# Generate random samples
samples = rand(d_msnburr, 1000)

# Get distribution moments
mean(d_msnburr)
var(d_msnburr)
skewness(d_msnburr)
kurtosis(d_msnburr)

# Summary statistics
summary_dist(d_msnburr)
```

## Distributions

### MSNBurr (Modified to be Stable Normal from Burr)

The MSNBurr distribution is parameterized by:
- `μ ∈ ℝ`: location parameter (mode)
- `σ > 0`: scale parameter
- `α > 0`: shape parameter (`α < 1` left skew, `α > 1` right skew)

**Density function** (with `ω = (1 + 1/α)^(α+1) / √(2π)`):
```
f(x|μ,σ,α) = (ω/σ) exp(-ω(x-μ)/σ) [1 + (1/α)exp(-ω(x-μ)/σ)]^(-(α+1))
```

At `α = 1` it is symmetric and mimics `Normal(μ, σ)` (same mode and peak height).

### MSNBurr-IIa

The MSNBurr-IIa distribution is the **mirror image** of MSNBurr with the same
parameterization `(μ, σ, α)`:

**Density function** (with `ω = (1 + 1/α)^(α+1) / √(2π)`):
```
f(x|μ,σ,α) = (ω/σ) exp(ω(x-μ)/σ) [1 + (1/α)exp(ω(x-μ)/σ)]^(-(α+1))
```

Same scale/variance as MSNBurr, opposite skewness sign: `α < 1` right skew, `α > 1` left skew.

**Special cases:**
- When `α = 1`, both distributions closely mimic the standard normal at the mode.
- `MSNBurr(μ, σ, α)` is exactly the `β = 1` case of the Generalized MSNBurr (GMSNBurr) family.

## Bayesian Modeling with Turing.jl

NeoDistribution distributions integrate seamlessly with Turing.jl:

```julia
using NeoDistribution
using Turing

# Model with MSNBurr likelihood
@model function flexible_model(y)
    μ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 3), 0, Inf)
    α ~ truncated(Normal(1, 2), 0.1, 20)

    for i in eachindex(y)
        y[i] ~ MSNBurr(μ, σ, α)
    end
end

# Bayesian regression with MSNBurr-IIa errors
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

# Fit the model
model = flexible_model(data)
chain = sample(model, NUTS(), 1000)
```

## API Reference

### Distribution Functions

| Function | Description |
|----------|-------------|
| `pdf(d, x)` | Probability density function |
| `logpdf(d, x)` | Log probability density function |
| `cdf(d, x)` | Cumulative distribution function |
| `logcdf(d, x)` | Log CDF |
| `ccdf(d, x)` | Complementary CDF (survival) |
| `logccdf(d, x)` | Log complementary CDF |
| `quantile(d, p)` | p-th quantile |
| `rand(d)` | Random sample |
| `rand(d, n)` | n random samples |

### Moment Functions

| Function | Description |
|----------|-------------|
| `mean(d)` | Expected value |
| `var(d)` | Variance |
| `std(d)` | Standard deviation |
| `mode(d)` | Mode |
| `median(d)` | Median |
| `skewness(d)` | Skewness |
| `kurtosis(d)` | Excess kurtosis |

### Summary Functions

```julia
# Comprehensive summary
summary_dist(d)

# Raw moments tuple
moments(d)  # Returns (μ₁=mean, μ₂=var, μ₃=skewness, μ₄=kurtosis)
```

## Moment Formulas

For both distributions, moments are expressed using polygamma functions (with
`ω = (1 + 1/α)^(α+1) / √(2π)`):

### MSNBurr
- **Mean**: E[X] = μ + (σ/ω)(ψ(α) - ψ(1) - log(α))
- **Variance**: Var[X] = (σ/ω)²(ψ₁(α) + ψ₁(1))
- **Skewness**: γ₁ = (ψ₂(α) - ψ₂(1))/(ψ₁(α) + ψ₁(1))^(3/2)
- **Kurtosis**: κ = (ψ₃(α) + ψ₃(1))/(ψ₁(α) + ψ₁(1))²

### MSNBurr-IIa (mirror image)
- **Mean**: E[X] = μ + (σ/ω)(ψ(1) - ψ(α) + log(α))
- **Variance**: Var[X] = (σ/ω)²(ψ₁(α) + ψ₁(1))
- **Skewness**: γ₁ = (ψ₂(1) - ψ₂(α))/(ψ₁(α) + ψ₁(1))^(3/2)
- **Kurtosis**: κ = (ψ₃(α) + ψ₃(1))/(ψ₁(α) + ψ₁(1))²

where ψₙ is the n-th order polygamma function.

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.jl`: Basic PDF/CDF/moments/sampling usage
- `turing_examples.jl`: Bayesian modeling with Turing.jl

## References

1. Iriawan, N. (2000). *Computationally Intensive Approaches to Inference in Neo-Normal Linear Models*. Curtin University of Technology.

2. Choir, A. S. (2020). *The New Neo-Normal Distributions and their Properties*. Dissertation. Institut Teknologi Sepuluh Nopember.

## License

MIT

## Authors

- Istanti Prima Nurani
