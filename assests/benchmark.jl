using BenchmarkTools, HypothesisTests, Random, Statistics, Distributions

# Model from tests
μ = [0.0, 1.0, 1.0]
Σ = [
    1.0 0.4 0.4;
    0.4 1.0 0.4;
    0.4 0.4 2.0]
lb = [0.0, 0.0, 2.0]
ub = [Inf, Inf, 4.0]
d = TruncatedMVN.TruncatedMVNormal(μ, Σ, lb, ub)
Random.seed!(1)
X =  TruncatedMVN.sample(d, 10_000)'
# TEST
ExactOneSampleKSTest(rand(truncated(Normal(0, 1), 0.0, Inf), 10_000), truncated(Normal(0, 1), 0.0, Inf))
#= V1.0.4
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.00588711

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.8768

Details:
    number of observations:   10000
=#
ExactOneSampleKSTest(X[:,1], truncated(Normal(0, 1), 0.0, Inf))
#=
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.092803

Test summary:
    outcome with 95% confidence: reject h_0
    two-sided p-value:           <1e-74

Details:
    number of observations:   10000
=#
ExactOneSampleKSTest(X[:,2], truncated(Normal(1, 1), 0.0, Inf))
#=
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.186968

Test summary:
    outcome with 95% confidence: reject h_0
    two-sided p-value:           <1e-99

Details:
    number of observations:   10000
=#
ExactOneSampleKSTest(X[:,3], truncated(Normal(1, sqrt(2.0)), 2.0, 4.0))
#=
Exact one sample Kolmogorov-Smirnov test
----------------------------------------
Population details:
    parameter of interest:   Supremum of CDF differences
    value under h_0:         0.0
    point estimate:          0.0378942

Test summary:
    outcome with 95% confidence: reject h_0
    two-sided p-value:           <1e-12

Details:
    number of observations:   10000
=#
d = TruncatedMVN.TruncatedMVNormal(μ, Σ, lb, ub)
Random.seed!(1)
@benchmark TruncatedMVN.sample(d, 10_000) seconds = 30
#= V1.0.4
BenchmarkTools.Trial: 52 samples with 1 evaluation per sample.
 Range (min … max):  459.788 ms … 711.358 ms  ┊ GC (min … max): 32.92% … 33.80%
 Time  (median):     595.091 ms               ┊ GC (median):    25.83%
 Time  (mean ± σ):   581.991 ms ±  62.498 ms  ┊ GC (mean ± σ):  26.40% ±  3.72%

                 ▃ ▃      ▃▃     ▃    ▃ ▃▃ ▃ ▃  ▃    ▃    █ ▃    
  ▇▇▁▁▁▁▇▇▇▇▁▁▇▇▁█▁█▇▇▁▇▇▁██▁▁▁▁▁█▁▇▁▁█▁██▇█▇█▇▁█▇▁▇▇█▇▁▇▁█▁█▁▇ ▁
  460 ms           Histogram: frequency by time          675 ms <

 Memory estimate: 2.24 GiB, allocs estimate: 1052.
=#
#= V1.0.5
Opt_1
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  1.802 ms …  12.778 ms  ┊ GC (min … max): 0.00% … 74.11%
 Time  (median):     1.924 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.178 ms ± 533.260 μs  ┊ GC (mean ± σ):  9.14% ± 13.34%

  ▃██▇▆▅▃▂▁          ▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁                     ▂
  ██████████▇▇▇▆▅▆▅▅▆███████████████████████▇▇▆▆▇▆▅▅▅▄▆▅▅▅▅▅▄ █
  1.8 ms       Histogram: log(frequency) by time      4.03 ms <

 Memory estimate: 5.05 MiB, allocs estimate: 738.
Opt_2
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  1.788 ms …  10.983 ms  ┊ GC (min … max): 0.00% … 77.52%
 Time  (median):     1.877 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   2.085 ms ± 461.647 μs  ┊ GC (mean ± σ):  7.69% ± 12.41%

  ▂██▆▄▃▁             ▂▂▂▂▂▂▁▁▁▁▁ ▁▁▁▁▁▁▁                     ▂
  ████████▇▆▄▆▅▄▅▅▅▅▆██████████████████████▇▇▇▆▆▅▅▅▄▄▄▄▅▄▄▅▃▄ █
  1.79 ms      Histogram: log(frequency) by time      3.78 ms <

 Memory estimate: 3.87 MiB, allocs estimate: 545.
=#