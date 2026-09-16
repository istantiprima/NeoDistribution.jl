# Distributions

This page provides detailed documentation of the distributions implemented in NeoDistribution.jl.

## MSNBurr Distribution

The **Modified to be Stable Normal from Burr** (MSNBurr) distribution is a flexible three-parameter distribution that can model skewness while retaining a normal-like center.

### Definition

```julia
MSNBurr(μ, σ, α)
```

**Parameters:**
- `μ ∈ ℝ`: location parameter (mode)
- `σ > 0`: scale parameter
- `α > 0`: shape parameter
  - `α > 1`: right-skewed
  - `α < 1`: left-skewed
  - `α = 1`: symmetric (mimics `Normal(μ, σ)` at the mode)

### Probability Density Function

```math
f(x|μ,σ,α) = \frac{ω}{σ} \exp\left(-ω\frac{x-μ}{σ}\right)
\left(1 + \frac{1}{α}\exp\left(-ω\frac{x-μ}{σ}\right)\right)^{-(α+1)}
```

where:
```math
ω = \frac{(1 + 1/α)^{α+1}}{\sqrt{2π}}
```

### Moments

Expressed using polygamma functions:

- **Mean**:
  ```math
  E[X] = μ + \frac{σ}{ω}\left(ψ(α) - ψ(1) - \log(α)\right)
  ```
- **Variance**:
  ```math
  \text{Var}[X] = \left(\frac{σ}{ω}\right)^2 \left(ψ_1(α) + ψ_1(1)\right)
  ```
- **Skewness**:
  ```math
  γ_1 = \frac{ψ_2(α) - ψ_2(1)}{(ψ_1(α) + ψ_1(1))^{3/2}}
  ```
- **Kurtosis**:
  ```math
  κ = \frac{ψ_3(α) + ψ_3(1)}{(ψ_1(α) + ψ_1(1))^2}
  ```

where `ψ_n(x)` is the n-th order polygamma function.

### Example

```julia
using NeoDistribution

# Create MSNBurr distribution
d = MSNBurr(0.0, 1.0, 2.0)

# PDF and CDF
pdf(d, 0.5)
cdf(d, 1.0)

# Quantiles
quantile(d, 0.5)  # median
quantile(d, 0.95) # 95th percentile

# Random samples
samples = rand(d, 10000)

# Moments
mean(d)
var(d)
skewness(d)  # positive (right-skewed for α > 1)
```

---

## MSNBurr-IIa Distribution

The **MSNBurr-IIa** distribution (`MSNBurr2a`) is the mirror image of MSNBurr. It shares the same scale/variance but has the opposite skewness sign.

### Definition

```julia
MSNBurr2a(μ, σ, α)
```

**Parameters:**
- `μ ∈ ℝ`: location parameter (mode)
- `σ > 0`: scale parameter
- `α > 0`: shape parameter
  - `α > 1`: left-skewed
  - `α < 1`: right-skewed
  - `α = 1`: symmetric (mimics `Normal(μ, σ)` at the mode)

### Probability Density Function

```math
f(x|μ,σ,α) = \frac{ω}{σ} \exp\left(ω\frac{x-μ}{σ}\right)
\left(1 + \frac{1}{α}\exp\left(ω\frac{x-μ}{σ}\right)\right)^{-(α+1)}
```

where `ω = (1 + 1/α)^{α+1} / \sqrt{2π}` (identical to MSNBurr).

### Moments

- **Mean**:
  ```math
  E[X] = μ + \frac{σ}{ω}\left(ψ(1) - ψ(α) + \log(α)\right)
  ```
- **Variance**:
  ```math
  \text{Var}[X] = \left(\frac{σ}{ω}\right)^2 \left(ψ_1(α) + ψ_1(1)\right)
  ```
- **Skewness**:
  ```math
  γ_1 = \frac{ψ_2(1) - ψ_2(α)}{(ψ_1(α) + ψ_1(1))^{3/2}}
  ```
- **Kurtosis**: identical to MSNBurr (kurtosis is mirror-invariant).

### Example

```julia
using NeoDistribution

d1 = MSNBurr(0.0, 1.0, 2.0)
d2 = MSNBurr2a(0.0, 1.0, 2.0)

# Mirror relationship: logpdf(d2, μ + t) == logpdf(d1, μ - t)
logpdf(d2, 1.0) ≈ logpdf(d1, -1.0)   # true

# Opposite skewness sign
skewness(d1) ≈ -skewness(d2)         # true

# Summary statistics
summary_dist(d2)
```

---

## Distribution Summary

Use `summary_dist()` to get a comprehensive overview:

```julia
julia> d = MSNBurr(0.0, 1.0, 2.0)
julia> summary_dist(d)
════════════════════════════════════════════════════════════
Distribution Summary: Modified to be Stable Normal from Burr (MSNBurr)
════════════════════════════════════════════════════════════

Parameters:
  μ = 0.0
  σ = 1.0
  α = 2.0

Statistics:
  ├─ Median          : ...
  ├─ Mean            : ...
  ├─ Mode            : 0.0
  ├─ Variance        : ...
  ├─ Std. Deviation  : ...
  ├─ Skewness        : ...
  └─ Excess Kurtosis : ...
════════════════════════════════════════════════════════════
```

## Comparison Table

| Feature | MSNBurr | MSNBurr-IIa |
|---------|---------|-------------|
| Parameters | 3 (μ, σ, α) | 3 (μ, σ, α) |
| Support | (-∞, +∞) | (-∞, +∞) |
| Mode | μ | μ |
| Symmetric when | α = 1 | α = 1 |
| Normal approx. (at mode) | α = 1 | α = 1 |
| Skew direction (α < 1) | left | right |
| CDF form | closed-form | closed-form |
