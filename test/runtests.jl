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
end
