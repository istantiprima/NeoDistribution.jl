"""
    moments.jl

Summary functions for Neo-Normal distributions, providing comprehensive
statistical summaries including moments and other characteristics.
"""

"""
    DistributionSummary

A struct to hold summary statistics of a distribution.
"""
struct DistributionSummary
    distribution::String
    parameters::NamedTuple
    median::Float64
    mean::Float64
    mode::Float64
    variance::Float64
    std::Float64
    skewness::Float64
    excess_kurtosis::Float64
end

"""
    summary_dist(d::MSNBurr)
    summary_dist(d::MSNBurr2a)

Compute a comprehensive summary of distribution characteristics.

Returns a `DistributionSummary` containing:
- Distribution name
- Parameters
- Median, Mean, Mode
- Variance, Standard deviation
- Skewness, Excess kurtosis

# Examples
```julia
using NeoDistribution

# MSNBurr distribution summary
d_msnburr = MSNBurr(0.0, 1.0, 2.0)
summary_dist(d_msnburr)

# MSNBurr-IIa distribution summary
d_msnburr2a = MSNBurr2a(0.0, 1.0, 2.0)
summary_dist(d_msnburr2a)
```
"""
function summary_dist(d::MSNBurr)
    μ, σ, α = params(d)

    DistributionSummary(
        "Modified to be Stable Normal from Burr (MSNBurr)",
        (μ=μ, σ=σ, α=α),
        Float64(median(d)),
        Float64(mean(d)),
        Float64(mode(d)),
        Float64(var(d)),
        Float64(std(d)),
        Float64(skewness(d)),
        Float64(kurtosis(d))
    )
end

function summary_dist(d::MSNBurr2a)
    μ, σ, α = params(d)

    DistributionSummary(
        "MSNBurr-IIa (mirror of MSNBurr)",
        (μ=μ, σ=σ, α=α),
        Float64(median(d)),
        Float64(mean(d)),
        Float64(mode(d)),
        Float64(var(d)),
        Float64(std(d)),
        Float64(skewness(d)),
        Float64(kurtosis(d))
    )
end

"""
    moments(d::ContinuousUnivariateDistribution)

Compute the first four moments/moment-based statistics of a distribution.

Returns a NamedTuple with:
- `μ₁`: first moment (mean)
- `μ₂`: second central moment (variance)
- `μ₃`: third standardized moment (skewness)
- `μ₄`: fourth standardized moment (excess kurtosis)

# Examples
```julia
d = MSNBurr(0.0, 1.0, 2.0)
m = moments(d)
m.μ₁  # mean
m.μ₂  # variance
m.μ₃  # skewness
m.μ₄  # excess kurtosis
```
"""
function moments(d::ContinuousUnivariateDistribution)
    return (
        μ₁ = mean(d),
        μ₂ = var(d),
        μ₃ = skewness(d),
        μ₄ = kurtosis(d)
    )
end

"""
    raw_moments(d::MSNBurr, k::Int)

Compute the k-th raw moment for MSNBurr distribution using numerical integration.
"""
function raw_moments(d::MSNBurr, k::Int)
    if k == 0
        return 1.0
    else
        return _numerical_moment(d, k)
    end
end

"""
    raw_moments(d::MSNBurr2a, k::Int)

Compute the k-th raw moment for MSNBurr-IIa distribution using numerical integration.
"""
function raw_moments(d::MSNBurr2a, k::Int)
    if k == 0
        return 1.0
    else
        return _numerical_moment(d, k)
    end
end

"""
    _numerical_moment(d::ContinuousUnivariateDistribution, k::Int)

Compute the k-th raw moment using numerical integration.
"""
function _numerical_moment(d::ContinuousUnivariateDistribution, k::Int)
    # Adaptive quadrature integration
    integrand(x) = x^k * pdf(d, x)

    # Split integration at mode for better accuracy
    m = mode(d)

    # Left tail
    left, _ = quadgk(integrand, -Inf, m; rtol=1e-8)

    # Right tail
    right, _ = quadgk(integrand, m, Inf; rtol=1e-8)

    return left + right
end

"""
    central_moments(d::ContinuousUnivariateDistribution, k::Int)

Compute the k-th central moment E[(X - μ)^k].
"""
function central_moments(d::ContinuousUnivariateDistribution, k::Int)
    μ = mean(d)
    integrand(x) = (x - μ)^k * pdf(d, x)

    # Integration
    result, _ = quadgk(integrand, -Inf, Inf; rtol=1e-8)
    return result
end

# Pretty printing for DistributionSummary
function Base.show(io::IO, s::DistributionSummary)
    println(io, "═" ^ 60)
    println(io, "Distribution Summary: $(s.distribution)")
    println(io, "═" ^ 60)
    println(io)
    println(io, "Parameters:")
    for (k, v) in pairs(s.parameters)
        println(io, "  $k = $v")
    end
    println(io)
    println(io, "Statistics:")
    println(io, "  ├─ Median          : $(round(s.median, digits=6))")
    println(io, "  ├─ Mean            : $(round(s.mean, digits=6))")
    println(io, "  ├─ Mode            : $(round(s.mode, digits=6))")
    println(io, "  ├─ Variance        : $(round(s.variance, digits=6))")
    println(io, "  ├─ Std. Deviation  : $(round(s.std, digits=6))")
    println(io, "  ├─ Skewness        : $(round(s.skewness, digits=6))")
    println(io, "  └─ Excess Kurtosis : $(round(s.excess_kurtosis, digits=6))")
    print(io, "═" ^ 60)
end

# Short display
function Base.show(io::IO, ::MIME"text/plain", s::DistributionSummary)
    show(io, s)
end
