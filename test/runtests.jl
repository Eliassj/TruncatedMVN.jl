using TruncatedMVN
using Test
using Random

@testset "TruncatedMVN.jl" begin

    μ = [0.0, 1.0, 1.0]
    Σ = [
        1.0 0.4 0.4;
        0.4 1.0 0.4;
        0.4 0.4 2.0]
    lb = [0.0, 0.0, 2.0]
    ub = [Inf, Inf, 4.0]
    d = TruncatedMVN.TruncatedMVNormal(μ, Σ, lb, ub)

    Random.seed!(1)
    X = TruncatedMVN.sample(d, 10_000)

    Random.seed!(1)
    Y = TruncatedMVN.sample(d, 10_000)

    @test X == Y

    # larger example
    Random.seed!(1)
    n = 2000
    d = 40
    μ = [1/i for i in 1:d]
    Σ = [exp.(-abs(i-j)/d) for i in 1:d, j in 1:d]
    lb = [isodd(i) ? i^(1/2) : -Inf for i in 1:d]
    ub = [iseven(i) ? i^(1/2) : Inf for i in 1:d]
    dist_d = TruncatedMVNormal(μ, Σ, lb, ub)
    Xd = TruncatedMVN.sample(dist_d, n)
    @test all([all(lb .≤ row .≤ ub) for row in eachcol(Xd)])
end
