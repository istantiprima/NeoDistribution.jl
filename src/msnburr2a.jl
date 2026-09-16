"""
    MSNBurr2a(μ, σ, α)

The *MSNBurr-IIa* distribution with location parameter `μ`, scale parameter `σ`,
and shape parameter `α`. It is the mirror image of [`MSNBurr`](@ref).

# Parameters
- `μ ∈ ℝ`: location parameter (mode of the distribution)
- `σ > 0`: scale parameter
- `α > 0`: shape parameter (controls skewness)

# Mathematical Details

The density function is (with `ω = (1 + 1/α)^(α+1) / √(2π)`):
```math
f(x|μ,σ,α) = \\frac{ω}{σ} \\exp\\left(ω\\frac{x-μ}{σ}\\right)
\\left(1 + \\frac{1}{α}\\exp\\left(ω\\frac{x-μ}{σ}\\right)\\right)^{-(α+1)}
```

# Special Cases
- It is the mirror image of [`MSNBurr`](@ref): same scale/variance, opposite
  skewness sign.
- At `α = 1` it mimics `Normal(μ, σ)` (same mode & peak height).
- Skews **right** for `α < 1`, **left** for `α > 1`.

# Examples
```julia
using NeoDistribution

# Create MSNBurr-IIa distribution
d = MSNBurr2a(0.0, 1.0, 1.0)

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

See also: [`MSNBurr`](@ref)
"""
struct MSNBurr2a{T<:Real} <: ContinuousUnivariateDistribution
    μ::T    # location
    σ::T    # scale
    α::T    # alpha (shape)
    # Explicit inner constructor (see the note on `MSNBurr` for the rationale).
    MSNBurr2a{T}(μ::T, σ::T, α::T) where {T<:Real} = new{T}(μ, σ, α)
end

# Single outer constructor: promote inside, validate optionally.
function MSNBurr2a(μ::Real, σ::Real, α::Real; check_args::Bool=true)
    if check_args
        (σ > zero(σ)) || throw(ArgumentError("MSNBurr2a: σ must be positive (got σ=$σ)"))
        (α > zero(α)) || throw(ArgumentError("MSNBurr2a: α must be positive (got α=$α)"))
    end
    μp, σp, αp = promote(float(μ), float(σ), float(α))
    return MSNBurr2a{typeof(μp)}(μp, σp, αp)
end
MSNBurr2a(μ::Real, σ::Real) = MSNBurr2a(μ, σ, one(float(μ)))
MSNBurr2a(μ::Real)          = MSNBurr2a(μ, one(float(μ)), one(float(μ)))
MSNBurr2a()                 = MSNBurr2a(0.0, 1.0, 1.0)

# Accessor functions
Distributions.params(d::MSNBurr2a) = (d.μ, d.σ, d.α)
Distributions.location(d::MSNBurr2a) = d.μ
Distributions.scale(d::MSNBurr2a) = d.σ
Distributions.partype(::MSNBurr2a{T}) where {T} = T
Base.eltype(::Type{MSNBurr2a{T}}) where {T} = T

# Support
Distributions.minimum(d::MSNBurr2a) = -Inf
Distributions.maximum(d::MSNBurr2a) = Inf
Distributions.insupport(d::MSNBurr2a, x::Real) = true

# MSNBurr-IIa shares the exact same normalizing constant ω as MSNBurr; these
# thin aliases keep this file self-documenting while preserving a single
# source of truth for the formula (defined in `msnburr.jl`).
"""
    _msnburr2a_log_omega(α)

Compute log(ω) for MSNBurr-IIa (identical to MSNBurr's ω).
"""
@inline _msnburr2a_log_omega(α::Real) = _msnburr_log_omega(α)

"""
    _msnburr2a_omega(α)

Compute the ω constant for MSNBurr-IIa (identical to MSNBurr's ω).
"""
@inline _msnburr2a_omega(α::Real) = _msnburr_omega(α)

"""
    logpdf(d::MSNBurr2a, x::Real)

Compute the log probability density function of MSNBurr-IIa at x
(numerically stable, AD-friendly).
"""
function Distributions.logpdf(d::MSNBurr2a, x::Real)
    μ, σ, α = params(d)
    logω = _msnburr2a_log_omega(α)
    ω    = exp(logω)
    z    = ω * (x - μ) / σ
    return logω - log(σ) + z - (α + 1) * log1pexp(z - log(α))
end

"""
    pdf(d::MSNBurr2a, x::Real)

Compute the probability density function of MSNBurr-IIa at x.
"""
Distributions.pdf(d::MSNBurr2a, x::Real) = exp(logpdf(d, x))

"""
    logcdf(d::MSNBurr2a, x::Real)

Compute the log cumulative distribution function of MSNBurr-IIa at x.
"""
function Distributions.logcdf(d::MSNBurr2a, x::Real)
    μ, σ, α = params(d)
    z = _msnburr2a_omega(α) * (x - μ) / σ
    return log1mexp(-α * log1pexp(z - log(α)))          # log(1 - S)
end

"""
    cdf(d::MSNBurr2a, x::Real)

Compute the cumulative distribution function of MSNBurr-IIa at x.
"""
Distributions.cdf(d::MSNBurr2a, x::Real) = exp(logcdf(d, x))

"""
    logccdf(d::MSNBurr2a, x::Real)

Compute the log complementary CDF (survival function) of MSNBurr-IIa at x.
"""
function Distributions.logccdf(d::MSNBurr2a, x::Real)                  # survival = exp(-α·log1pexp)
    μ, σ, α = params(d)
    z = _msnburr2a_omega(α) * (x - μ) / σ
    return -α * log1pexp(z - log(α))
end

"""
    ccdf(d::MSNBurr2a, x::Real)

Compute the complementary CDF (survival function) of MSNBurr-IIa at x.
"""
Distributions.ccdf(d::MSNBurr2a, x::Real) = exp(logccdf(d, x))

"""
    quantile(d::MSNBurr2a, p::Real)

Compute the p-th quantile (inverse CDF) of MSNBurr-IIa distribution.
"""
function Distributions.quantile(d::MSNBurr2a, p::Real)
    μ, σ, α = params(d)
    ω = _msnburr2a_omega(α)
    return μ + (σ / ω) * (log(α) + log((1 - p)^(-1 / α) - 1))
end

"""
    rand(rng::AbstractRNG, d::MSNBurr2a)

Generate a random sample from MSNBurr-IIa distribution (inverse-CDF method).
"""
function Base.rand(rng::AbstractRNG, d::MSNBurr2a)
    μ, σ, α = params(d)
    ω = _msnburr2a_omega(α)
    u = rand(rng)
    return μ + (σ / ω) * (log(α) + log((1 - u)^(-1 / α) - 1))
end

# ============================================================================
# Moment Functions for MSNBurr-IIa
# ============================================================================

"""
    mean(d::MSNBurr2a)

Compute the mean of MSNBurr-IIa distribution.

E[X] = μ + (σ/ω) * (ψ(1) - ψ(α) + log(α))

where ψ(x) is the digamma function.
"""
function Distributions.mean(d::MSNBurr2a)
    μ, σ, α = params(d)
    return μ + (σ / _msnburr2a_omega(α)) * (digamma(1) - digamma(α) + log(α))
end

"""
    var(d::MSNBurr2a)

Compute the variance of MSNBurr-IIa distribution.

Var[X] = (σ/ω)² * (ψ₁(α) + ψ₁(1))

where ψ₁(x) is the trigamma function.
"""
function Distributions.var(d::MSNBurr2a)
    _, σ, α = params(d)
    return (σ / _msnburr2a_omega(α))^2 * (trigamma(α) + trigamma(1))
end

"""
    std(d::MSNBurr2a)

Compute the standard deviation of MSNBurr-IIa distribution.
"""
Distributions.std(d::MSNBurr2a) = sqrt(var(d))

"""
    skewness(d::MSNBurr2a)

Compute the skewness of MSNBurr-IIa distribution (opposite sign to MSNBurr).

γ₁ = (ψ₂(1) - ψ₂(α)) / (ψ₁(α) + ψ₁(1))^(3/2)

where ψₙ(x) is the n-th order polygamma function.
"""
function Distributions.skewness(d::MSNBurr2a)
    α = d.α
    return (polygamma(2, 1) - polygamma(2, α)) / (trigamma(α) + trigamma(1))^(3 / 2)
end

"""
    kurtosis(d::MSNBurr2a)

Compute the excess kurtosis of MSNBurr-IIa distribution.

κ = (ψ₃(α) + ψ₃(1)) / (ψ₁(α) + ψ₁(1))²

Kurtosis is mirror-invariant, so it equals that of [`MSNBurr`](@ref) with the
same `α` (the `β = 1` special case of the GMSNBurr kurtosis).
"""
function Distributions.kurtosis(d::MSNBurr2a)
    α = d.α
    return (polygamma(3, α) + polygamma(3, 1)) / (trigamma(α) + trigamma(1))^2
end

"""
    mode(d::MSNBurr2a)

The mode of MSNBurr-IIa distribution is at μ.
"""
Distributions.mode(d::MSNBurr2a) = d.μ

"""
    median(d::MSNBurr2a)

Compute the median of MSNBurr-IIa distribution (via the analytic quantile).
"""
Distributions.median(d::MSNBurr2a) = quantile(d, 0.5)

# Helper: show method
function Base.show(io::IO, d::MSNBurr2a)
    print(io, "MSNBurr2a(μ=$(d.μ), σ=$(d.σ), α=$(d.α))")
end

function Base.show(io::IO, ::MIME"text/plain", d::MSNBurr2a)
    println(io, "MSNBurr-IIa distribution")
    println(io, "  μ = $(d.μ)")
    println(io, "  σ = $(d.σ)")
    print(io, "  α = $(d.α)")
end
