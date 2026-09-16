# NeoDistribution.jl Documentation

```@meta
CurrentModule = NeoDistribution
```

## Introduction

**NeoDistribution.jl** is a Julia package for working with Neo-Normal distributions - a family of flexible probability distributions that generalize the normal distribution to accommodate skewness.

## Key Features

- **MSNBurr Distribution**: Modified to be Stable Normal from Burr distribution
- **MSNBurr-IIa Distribution** (`MSNBurr2a`): mirror image of MSNBurr
- Full `Distributions.jl` compatibility
- AD-friendly, numerically stable log-density (ready for NUTS/HMC)
- Seamless `Turing.jl` integration for Bayesian modeling
- Analytical moment calculations

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/istantiprima/NeoDistribution.jl")
```

## Quick Example

```julia
using NeoDistribution

# Create an MSNBurr distribution
d = MSNBurr(0.0, 1.0, 2.0)

# Basic operations
pdf(d, 0.5)      # Density at x=0.5
cdf(d, 1.0)      # CDF at x=1.0
rand(d, 1000)    # Generate 1000 samples

# Moments
mean(d)          # Mean
var(d)           # Variance
skewness(d)      # Skewness
kurtosis(d)      # Excess kurtosis

# Summary
summary_dist(d)  # Complete summary
```

## Contents

```@contents
Pages = [
    "distributions.md",
]
Depth = 2
```

## References

1. Iriawan, N. (2000). *Computationally Intensive Approaches to Inference in Neo-Normal Linear Models*. Curtin University of Technology.

2. Choir, A. S. (2020). *The New Neo-Normal Distributions and their Properties*. Dissertation. Institut Teknologi Sepuluh Nopember.
