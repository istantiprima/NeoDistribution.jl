# Basic usage of NeoDistribution.jl
#
# Run with:  julia --project examples/basic_usage.jl

using NeoDistribution
using Random

Random.seed!(2024)

# ----------------------------------------------------------------------
# 1. Construct distributions
# ----------------------------------------------------------------------
d   = MSNBurr(0.0, 1.0, 2.0)     # μ, σ, α  (right-skewed since α > 1)
d2a = MSNBurr2a(0.0, 1.0, 2.0)   # mirror image (left-skewed since α > 1)

println("d   = ", d)
println("d2a = ", d2a)

# ----------------------------------------------------------------------
# 2. Density, distribution and quantile functions
# ----------------------------------------------------------------------
println("\npdf(d, 0.5)      = ", pdf(d, 0.5))
println("logpdf(d, 0.5)   = ", logpdf(d, 0.5))
println("cdf(d, 1.0)      = ", cdf(d, 1.0))
println("ccdf(d, 1.0)     = ", ccdf(d, 1.0))
println("quantile(d, 0.5) = ", quantile(d, 0.5))
println("quantile(d, 0.95)= ", quantile(d, 0.95))

# ----------------------------------------------------------------------
# 3. Moments
# ----------------------------------------------------------------------
println("\nmean(d)     = ", mean(d))
println("var(d)      = ", var(d))
println("std(d)      = ", std(d))
println("skewness(d) = ", skewness(d))
println("kurtosis(d) = ", kurtosis(d))
println("mode(d)     = ", mode(d))
println("median(d)   = ", median(d))

# ----------------------------------------------------------------------
# 4. Random sampling
# ----------------------------------------------------------------------
samples = rand(d, 10_000)
println("\nsample mean = ", sum(samples) / length(samples), " (theory: ", mean(d), ")")

# ----------------------------------------------------------------------
# 5. Comprehensive summary
# ----------------------------------------------------------------------
println()
display(summary_dist(d))
println()
display(summary_dist(d2a))
println()

# ----------------------------------------------------------------------
# 6. First four moments as a tuple
# ----------------------------------------------------------------------
println("\nmoments(d) = ", moments(d))
