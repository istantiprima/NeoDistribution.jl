"""
    MSNBurr(μ, σ, α)

The *Modified to be Stable Normal from Burr* (MSNBurr) distribution with location
parameter `μ`, scale parameter `σ`, and shape parameter `α`.

# Parameters
- `μ ∈ ℝ`: location parameter (mode of the distribution)
- `σ > 0`: scale parameter
- `α > 0`: shape parameter (controls skewness)

# Mathematical Details

The density function is (with `ω = (1 + 1/α)^(α+1) / √(2π)`):
```math
f(x|μ,σ,α) = \\frac{ω}{σ} \\exp\\left(-ω\\frac{x-μ}{σ}\\right)
\\left(1 + \\frac{1}{α}\\exp\\left(-ω\\frac{x-μ}{σ}\\right)\\right)^{-(α+1)}
```

# Special Cases
- At `α = 1` it is symmetric and *mimics* `Normal(μ, σ)` — it shares the same
  mode (`x = μ`) and peak height, but not the tails (it is not literally Normal).
- Skews **left** for `α < 1`, **right** for `α > 1`.

`MSNBurr(μ, σ, α)` coincides exactly with the special case `β = 1` of the
Generalized MSNBurr (GMSNBurr) family.

# Examples
```julia
using NeoDistribution

# Create MSNBurr distribution
d = MSNBurr(0.0, 1.0, 1.0)

# Calculate PDF
pdf(d, 0.5)

# Calculate CDF
cdf(d, 0.5)

# Generate random samples
rand(d, 1000)

# Get moments
mean(d)
var(d)
skewness(d)
kurtosis(d)
```

# References
- Iriawan, N. (2000). Computationally Intensive Approaches to Inference in
  Neo-Normal Linear Models. Curtin University of Technology.
- Choir, A. S. (2020). The New Neo-Normal Distributions and their Properties.
  Dissertation. Institut Teknologi Sepuluh Nopember.

See also: [`MSNBurr2a`](@ref)
"""
struct MSNBurr{T<:Real} <: ContinuousUnivariateDistribution
    μ::T    # location
    σ::T    # scale
    α::T    # alpha (shape)
    # Explicit inner constructor so Julia does NOT auto-generate the
    # same-type outer `MSNBurr(::T,::T,::T)`, which would bypass the
    # checking/promoting outer constructor below (argument validation
    # and float promotion). This is the Distributions.jl pattern.
    MSNBurr{T}(μ::T, σ::T, α::T) where {T<:Real} = new{T}(μ, σ, α)
end

# Single outer constructor: promote inside so there is exactly one
# method for (Real, Real, Real) — avoids the method-overwriting that
# breaks precompilation when all three arguments share a type, and
# guarantees float promotion of integer arguments.
function MSNBurr(μ::Real, σ::Real, α::Real; check_args::Bool=true)
    if check_args
        (σ > zero(σ)) || throw(ArgumentError("MSNBurr: σ must be positive (got σ=$σ)"))
        (α > zero(α)) || throw(ArgumentError("MSNBurr: α must be positive (got α=$α)"))
    end
    μp, σp, αp = promote(float(μ), float(σ), float(α))
    return MSNBurr{typeof(μp)}(μp, σp, αp)
end
MSNBurr(μ::Real, σ::Real) = MSNBurr(μ, σ, one(float(μ)))
MSNBurr(μ::Real)          = MSNBurr(μ, one(float(μ)), one(float(μ)))
MSNBurr()                 = MSNBurr(0.0, 1.0, 1.0)

# Accessor functions
Distributions.params(d::MSNBurr) = (d.μ, d.σ, d.α)
Distributions.location(d::MSNBurr) = d.μ
Distributions.scale(d::MSNBurr) = d.σ
Distributions.partype(::MSNBurr{T}) where {T} = T
Base.eltype(::Type{MSNBurr{T}}) where {T} = T

# Support
Distributions.minimum(d::MSNBurr) = -Inf
Distributions.maximum(d::MSNBurr) = Inf
Distributions.insupport(d::MSNBurr, x::Real) = true

# Stable log of the normalizing constant ω = (1 + 1/α)^(α+1) / sqrt(2π).
# Written with log1p + exp so it is well-behaved and AD-differentiable.
const LOG2π = log(2 * π)                      # ≈ 1.8378770664093453 (constant)

"""
    _msnburr_log_omega(α)

Compute log(ω) for MSNBurr, where ω = (1 + 1/α)^(α+1) / √(2π), for numerical stability.
"""
@inline _msnburr_log_omega(α::Real) = (α + 1) * log1p(1 / α) - LOG2π / 2

"""
    _msnburr_omega(α)

Compute the ω constant for MSNBurr: ω = (1 + 1/α)^(α+1) / √(2π).
"""
@inline _msnburr_omega(α::Real) = exp(_msnburr_log_omega(α))

"""
    logpdf(d::MSNBurr, x::Real)

Compute the log probability density function of MSNBurr at x
(numerically stable, AD-friendly).
"""
function Distributions.logpdf(d::MSNBurr, x::Real)
    μ, σ, α = params(d)
    logω = _msnburr_log_omega(α)
    ω    = exp(logω)
    z    = ω * (x - μ) / σ                     # standardized argument
    return logω - log(σ) - z - (α + 1) * log1pexp(-z - log(α))
end

"""
    pdf(d::MSNBurr, x::Real)

Compute the probability density function of MSNBurr at x.
"""
Distributions.pdf(d::MSNBurr, x::Real) = exp(logpdf(d, x))

"""
    logcdf(d::MSNBurr, x::Real)

Compute the log cumulative distribution function of MSNBurr at x.
"""
function Distributions.logcdf(d::MSNBurr, x::Real)
    μ, σ, α = params(d)
    z = _msnburr_omega(α) * (x - μ) / σ
    return -α * log1pexp(-z - log(α))
end

"""
    cdf(d::MSNBurr, x::Real)

Compute the cumulative distribution function of MSNBurr at x.
"""
Distributions.cdf(d::MSNBurr, x::Real) = exp(logcdf(d, x))

"""
    ccdf(d::MSNBurr, x::Real)

Compute the complementary CDF (survival function) of MSNBurr at x.
"""
Distributions.ccdf(d::MSNBurr, x::Real) = -expm1(logcdf(d, x))

"""
    logccdf(d::MSNBurr, x::Real)

Compute the log complementary CDF of MSNBurr at x.
"""
Distributions.logccdf(d::MSNBurr, x::Real) = log1mexp(logcdf(d, x))

"""
    quantile(d::MSNBurr, p::Real)

Compute the p-th quantile (inverse CDF) of MSNBurr distribution.
"""
function Distributions.quantile(d::MSNBurr, p::Real)
    μ, σ, α = params(d)
    ω = _msnburr_omega(α)
    return μ - (σ / ω) * (log(α) + log(p^(-1 / α) - 1))
end

"""
    rand(rng::AbstractRNG, d::MSNBurr)

Generate a random sample from MSNBurr distribution (inverse-CDF method).
"""
function Base.rand(rng::AbstractRNG, d::MSNBurr)
    μ, σ, α = params(d)
    ω = _msnburr_omega(α)
    u = rand(rng)
    return μ - (σ / ω) * (log(α) + log(u^(-1 / α) - 1))
end

# ============================================================================
# Moment Functions for MSNBurr
# ============================================================================

"""
    mean(d::MSNBurr)

Compute the mean of MSNBurr distribution.

E[X] = μ + (σ/ω) * (ψ(α) - ψ(1) - log(α))

where ψ(x) is the digamma function.
"""
function Distributions.mean(d::MSNBurr)
    μ, σ, α = params(d)
    return μ + (σ / _msnburr_omega(α)) * (digamma(α) - digamma(1) - log(α))
end

"""
    var(d::MSNBurr)

Compute the variance of MSNBurr distribution.

Var[X] = (σ/ω)² * (ψ₁(α) + ψ₁(1))

where ψ₁(x) is the trigamma function.
"""
function Distributions.var(d::MSNBurr)
    _, σ, α = params(d)
    return (σ / _msnburr_omega(α))^2 * (trigamma(α) + trigamma(1))
end

"""
    std(d::MSNBurr)

Compute the standard deviation of MSNBurr distribution.
"""
Distributions.std(d::MSNBurr) = sqrt(var(d))

"""
    skewness(d::MSNBurr)

Compute the skewness of MSNBurr distribution.

γ₁ = (ψ₂(α) - ψ₂(1)) / (ψ₁(α) + ψ₁(1))^(3/2)

where ψₙ(x) is the n-th order polygamma function.
"""
function Distributions.skewness(d::MSNBurr)
    α = d.α
    return (polygamma(2, α) - polygamma(2, 1)) / (trigamma(α) + trigamma(1))^(3 / 2)
end

"""
    kurtosis(d::MSNBurr)

Compute the excess kurtosis of MSNBurr distribution.

κ = (ψ₃(α) + ψ₃(1)) / (ψ₁(α) + ψ₁(1))²

where ψₙ(x) is the n-th order polygamma function. This is the `β = 1` special
case of the GMSNBurr kurtosis, i.e. it belongs to the same neo-normal family.
"""
function Distributions.kurtosis(d::MSNBurr)
    α = d.α
    return (polygamma(3, α) + polygamma(3, 1)) / (trigamma(α) + trigamma(1))^2
end

"""
    mode(d::MSNBurr)

The mode of MSNBurr distribution is at μ.
"""
Distributions.mode(d::MSNBurr) = d.μ

"""
    median(d::MSNBurr)

Compute the median of MSNBurr distribution (via the analytic quantile).
"""
Distributions.median(d::MSNBurr) = quantile(d, 0.5)

# Helper: show method
function Base.show(io::IO, d::MSNBurr)
    print(io, "MSNBurr(μ=$(d.μ), σ=$(d.σ), α=$(d.α))")
end

function Base.show(io::IO, ::MIME"text/plain", d::MSNBurr)
    println(io, "MSNBurr distribution")
    println(io, "  μ = $(d.μ)")
    println(io, "  σ = $(d.σ)")
    print(io, "  α = $(d.α)")
end
