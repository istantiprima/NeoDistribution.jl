using NeoDistribution
using Distributions
using Statistics
using Test
using ForwardDiff
using Random

# Reference values were produced independently in Python/SciPy with the
# log-sum-exp stable formulas, then pasted here as ground truth. They are the
# regression guard proving the restructuring did not change any computation.

@testset "NeoDistribution.jl" begin

    @testset "logpdf / cdf reference values" begin
        d = MSNBurr(0.0, 1.0, 1.0)
        @test logpdf(d, -1.0) ≈ -1.4976386992 atol = 1e-8
        @test logpdf(d,  0.0) ≈ -0.9189385332 atol = 1e-8   # = -0.5log(2π)
        @test logpdf(d,  2.0) ≈ -2.8047546048 atol = 1e-8
        @test cdf(d,  0.0)    ≈ 0.5            atol = 1e-10
        @test cdf(d,  2.0)    ≈ 0.9605146017   atol = 1e-8

        d2 = MSNBurr(2.0, 1.5, 0.5)
        @test logpdf(d2, 0.0) ≈ -2.1447342369 atol = 1e-8
        @test cdf(d2,    0.0) ≈ 0.1748079374   atol = 1e-8

        d3 = MSNBurr(0.0, 1.0, 3.0)
        @test logpdf(d3, 2.0) ≈ -2.3956071259 atol = 1e-8
        @test cdf(d3,    2.0) ≈ 0.9237944027   atol = 1e-8

        e = MSNBurr2a(2.0, 1.5, 0.5)
        @test logpdf(e, 0.0) ≈ -2.6185577482 atol = 1e-8
        @test cdf(e,    0.0) ≈ 0.0576449030   atol = 1e-8

        e2 = MSNBurr2a(0.0, 1.0, 3.0)
        @test logpdf(e2, -1.0) ≈ -1.3901478588 atol = 1e-8
        @test cdf(e2,    -1.0) ≈ 0.2372403094   atol = 1e-8
    end

    @testset "pdf = exp(logpdf) & non-negatif" begin
        for D in (MSNBurr, MSNBurr2a), (μ, σ, α) in
                ((0.0, 1.0, 0.5), (2.0, 1.5, 1.0), (-1.0, 0.7, 3.0))
            d = D(μ, σ, α)
            for x in range(μ - 6σ, μ + 6σ; length = 40)
                @test pdf(d, x) ≈ exp(logpdf(d, x)) rtol = 1e-10
                @test pdf(d, x) ≥ 0
            end
        end
    end

    @testset "limit Normal pada alpha=1 (di modus)" begin
        for D in (MSNBurr, MSNBurr2a), (μ, σ) in ((0.0, 1.0), (2.0, 1.5))
            d = D(μ, σ, 1.0)
            @test logpdf(d, μ) ≈ logpdf(Normal(μ, σ), μ) atol = 1e-10
            @test mode(d) == μ
        end
    end

    @testset "pdf integrates to 1" begin
        for d in (MSNBurr(0.0, 1.0, 0.3), MSNBurr2a(1.0, 2.0, 4.0))
            xs = range(-60, 60; length = 200_001)
            I = sum(pdf.(Ref(d), xs)) * step(xs)
            @test I ≈ 1.0 atol = 1e-3
        end
    end

    @testset "cdf/ccdf/logcdf konsisten & monoton" begin
        for D in (MSNBurr, MSNBurr2a), (μ, σ, α) in
                ((0.0, 1.0, 0.5), (1.0, 2.0, 3.0))
            d = D(μ, σ, α)
            @test cdf(d, μ - 50σ) ≈ 0 atol = 1e-6
            @test cdf(d, μ + 50σ) ≈ 1 atol = 1e-6
            xs = range(μ - 8σ, μ + 8σ; length = 30)
            @test issorted(cdf.(Ref(d), xs))
            for x in xs
                @test 0 ≤ cdf(d, x) ≤ 1
                @test cdf(d, x)  ≈ exp(logcdf(d, x)) rtol = 1e-9
                @test ccdf(d, x) ≈ 1 - cdf(d, x)     atol = 1e-9
            end
        end
    end

    @testset "cdf(quantile(p)) == p" begin
        for d in (MSNBurr(0.0, 1.0, 0.7), MSNBurr2a(-1.0, 2.0, 3.0))
            for p in (0.05, 0.25, 0.5, 0.8, 0.99)
                @test cdf(d, quantile(d, p)) ≈ p atol = 1e-8
            end
        end
    end

    @testset "moments (analytic vs reference)" begin
        @test mean(MSNBurr(0.0, 1.0, 0.1))     ≈ -1.3526 atol = 1e-3
        @test mean(MSNBurr2a(0.0, 1.0, 0.1))   ≈ +1.3526 atol = 1e-3
        @test var(MSNBurr(0.0, 1.0, 0.1))      ≈  3.3135 atol = 1e-3
        @test var(MSNBurr2a(0.0, 1.0, 0.1))    ≈  3.3135 atol = 1e-3
        @test skewness(MSNBurr(0.0, 1.0, 0.1)) ≈ -1.9106 atol = 1e-3
        @test skewness(MSNBurr2a(0.0,1.0,0.1)) ≈ +1.9106 atol = 1e-3
        @test mode(MSNBurr(3.0, 1.0, 0.4))     == 3.0
    end

    @testset "sampling matches theory" begin
        rng = MersenneTwister(2024)
        for D in (MSNBurr, MSNBurr2a)
            d = D(0.5, 1.2, 2.0)
            s = rand(rng, d, 200_000)
            @test mean(s) ≈ mean(d) atol = 2e-2
            @test var(s)  ≈ var(d)  atol = 5e-2
        end
    end

    @testset "AD-compatibility (ForwardDiff)" begin
        y = [0.3, -1.1, 2.0, 0.7, -0.4]
        for D in (MSNBurr, MSNBurr2a)
            f = θ -> sum(logpdf(D(θ[1], θ[2], θ[3]; check_args = false), yi) for yi in y)
            g = ForwardDiff.gradient(f, [0.2, 1.3, 1.7])
            @test all(isfinite, g)
            @test length(g) == 3
        end
    end

    # ------------------------------------------------------------------
    #  Validasi argumen, promosi tipe, dan sifat cermin
    # ------------------------------------------------------------------
    @testset "validasi argumen konstruktor" begin
        for D in (MSNBurr, MSNBurr2a)
            @test D(0.0, 1.0, 0.5) isa D
            @test_throws ArgumentError D(0.0, -1.0, 0.5)   # sigma < 0
            @test_throws ArgumentError D(0.0,  0.0, 0.5)   # sigma = 0
            @test_throws ArgumentError D(0.0,  1.0, -0.5)  # alpha < 0
            @test_throws ArgumentError D(0.0,  1.0,  0.0)  # alpha = 0
            @test D(0.0, -1.0, 0.5; check_args = false) isa D
        end
    end

    @testset "promosi tipe parameter (float)" begin
        for D in (MSNBurr, MSNBurr2a)
            @test D(0, 1, 1) isa D
            @test partype(D(0, 1, 1)) <: AbstractFloat
            @test all(x -> x isa AbstractFloat, params(D(0, 1, 1)))
        end
    end

    @testset "sifat cermin MSNBurr <-> MSNBurr2a" begin
        for (μ, σ, α) in ((0.0, 1.0, 0.5), (2.0, 1.5, 3.0), (-1.0, 0.7, 2.0))
            d1 = MSNBurr(μ, σ, α)
            d2 = MSNBurr2a(μ, σ, α)
            for t in range(0, 6σ; length = 20)
                @test logpdf(d2, μ + t) ≈ logpdf(d1, μ - t) atol = 1e-9
            end
            @test mean(d1) + mean(d2)  ≈ 2μ atol = 1e-9
            @test var(d1) ≈ var(d2)         atol = 1e-9
            @test skewness(d1) ≈ -skewness(d2) atol = 1e-9
        end
    end

    # ------------------------------------------------------------------
    #  Method aditif baru (kurtosis, median, support, summary_dist, moments)
    #  — perhitungan lama tidak berubah; ini hanya melengkapi interface
    #    agar sepadan dengan NeoNormal (FOSSEP/GMSNBurr).
    # ------------------------------------------------------------------
    @testset "support (minimum/maximum/insupport)" begin
        for D in (MSNBurr, MSNBurr2a)
            d = D(0.0, 1.0, 2.0)
            @test minimum(d) == -Inf
            @test maximum(d) == Inf
            @test insupport(d, 0.0)
            @test insupport(d, 1e6)
        end
    end

    @testset "kurtosis (mirror-invariant, β=1 GMSNBurr)" begin
        for (μ, σ, α) in ((0.0, 1.0, 0.5), (2.0, 1.5, 3.0), (-1.0, 0.7, 2.0))
            k1 = kurtosis(MSNBurr(μ, σ, α))
            k2 = kurtosis(MSNBurr2a(μ, σ, α))
            @test isfinite(k1)
            @test k1 ≈ k2 atol = 1e-12       # kurtosis is mirror-invariant
        end
    end

    @testset "median = quantile(0.5)" begin
        for d in (MSNBurr(0.0, 1.0, 0.7), MSNBurr2a(-1.0, 2.0, 3.0))
            @test median(d) ≈ quantile(d, 0.5) atol = 1e-12
            @test cdf(d, median(d)) ≈ 0.5 atol = 1e-8
        end
    end

    @testset "summary_dist & moments" begin
        for d in (MSNBurr(0.0, 1.0, 2.0), MSNBurr2a(1.0, 1.5, 0.8))
            s = summary_dist(d)
            @test s.mean      ≈ mean(d)      rtol = 1e-10
            @test s.variance  ≈ var(d)       rtol = 1e-10
            @test s.std       ≈ std(d)       rtol = 1e-10
            @test s.skewness  ≈ skewness(d)  rtol = 1e-10
            @test s.mode      ≈ mode(d)      rtol = 1e-10
            @test s.median    ≈ median(d)    rtol = 1e-10
            @test s.excess_kurtosis ≈ kurtosis(d) rtol = 1e-10

            m = moments(d)
            @test m.μ₁ == mean(d)
            @test m.μ₂ == var(d)
            @test m.μ₃ == skewness(d)
            @test m.μ₄ == kurtosis(d)
        end
    end

    @testset "numerical moments cocok dgn analitik" begin
        # raw_moments(d, 1) (numerik, quadgk) harus ≈ mean(d) (analitik)
        for d in (MSNBurr(0.0, 1.0, 2.0), MSNBurr2a(0.0, 1.0, 0.5))
            @test NeoDistribution.raw_moments(d, 1) ≈ mean(d) rtol = 1e-5
            @test NeoDistribution.central_moments(d, 2) ≈ var(d) rtol = 1e-5
        end
    end

    @testset "show tidak error" begin
        for d in (MSNBurr(0.0, 1.0, 2.0), MSNBurr2a(1.0, 1.5, 0.8))
            @test occursin("μ", sprint(show, d))
            @test occursin("Distribution Summary", sprint(show, summary_dist(d)))
        end
    end

end
