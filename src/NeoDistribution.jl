"""
    NeoDistribution

A Julia package implementing Neo-Normal distributions for Bayesian inference.

This package provides implementations of:
- **MSNBurr**: Modified to be Stable Normal from Burr distribution (left-skew neo-normal)
- **MSNBurr-IIa** (`MSNBurr2a`): mirror image of MSNBurr (right-skew neo-normal)

Both distributions are compatible with the Distributions.jl interface, are AD-friendly,
and can be used with Turing.jl for Bayesian modeling (NUTS/HMC).

# References
- Iriawan, N. (2000). Computationally Intensive Approaches to Inference in
  Neo-Normal Linear Models. Curtin University of Technology.
- Choir, A. S. (2020). The New Neo-Normal Distributions and their Properties.
  Dissertation. Institut Teknologi Sepuluh Nopember.
"""
module NeoDistribution

using Distributions
# Numerically stable building blocks:
#   log1pexp(x) = log(1 + exp(x))  -> avoids overflow/underflow
#   log1mexp(x) = log(1 - exp(x))  -> stable for x <= 0
# NOTE: these live in LogExpFunctions.jl (not SpecialFunctions.jl).
using LogExpFunctions: log1pexp, log1mexp
# digamma/trigamma/polygamma are used only for the analytic moments
# (mean/var/skewness/kurtosis); they are not on the NUTS gradient path.
using SpecialFunctions: digamma, trigamma, polygamma
using QuadGK
using Random

import Distributions: pdf, logpdf, cdf, logcdf, ccdf, logccdf, quantile, rand, mean, var, std
import Distributions: skewness, kurtosis, mode, median, minimum, maximum
import Distributions: params, location, scale, insupport, partype
import Random: AbstractRNG

# Export distributions
export MSNBurr, MSNBurr2a

# Export distribution functions (re-export from Distributions)
export pdf, logpdf, cdf, logcdf, ccdf, logccdf, quantile, rand

# Export moment functions (re-export from Distributions)
export mean, var, std, skewness, kurtosis, mode, median

# Export summary functions
export summary_dist, moments

# Export utility functions
export params

# Include source files
include("msnburr.jl")
include("msnburr2a.jl")
include("moments.jl")

end # module
