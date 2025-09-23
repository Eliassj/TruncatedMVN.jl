# TruncatedMVN.jl

Julia reimplementation of a truncated multivariate normal distribution with perfect sampling.

Distribution and sampling method by [Botev_2017](@cite).

The code is based on MATLAB implementation by Zdravko Botev and Python implementation by Paul Brunzema.
These may be found here: [MATLAB by Botev](https://mathworks.com/matlabcentral/fileexchange/53792-truncated-multivariate-normal-generator), [Python by Brunzema](https://github.com/brunzema/truncated-mvn-sampler)

## Example : Simulation of Gaussian variables subject to linear restrictions 

This example is adapted from the [MATLAB](https://mathworks.com/matlabcentral/fileexchange/53792-truncated-multivariate-normal-generator) package image and the R package `TruncatedNormal` [vignette](https://cloud.r-project.org/web/packages/TruncatedNormal/vignettes/TruncatedNormal_vignette.html).

Suppose we wish to simulate a bivariate vector $\boldsymbol{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ conditional on $X_1-X_2 < -6$. Setting $\mathbf{A} = \left( \begin{smallmatrix} 1 & -1 \\ 0 & 1\end{smallmatrix}\right)$. We can simulate $\boldsymbol{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu},\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$ conditional on $\boldsymbol{l} \leq \boldsymbol{Y} \leq \boldsymbol{u}$ and then set $\boldsymbol{X} = \mathbf{A}^{-1}\boldsymbol{Y}$.


```julia
using Random, Distributions
using CairoMakie, LatexStrings
using TruncatedMVN

# Problem setup
Random.seed!(1234) 
Σ = [1.0 0.9; 
     0.9 1.0]
μ = [-3.0, 0.0]
ub = [-6.0, Inf]
lb = [-Inf, -Inf]
A = [1.0 -1.0;
     0.0  1.0]

n = 1000

# Sample from truncated MVN under A*x ≤ ub
tmvn = TruncatedMVNormal(A * μ, A * Σ * A', lb, ub)
Y_truncated = TruncatedMVN.sample(tmvn, n)          # 2×n matrix
X_conditional = A \ Y_truncated                       # 2×n matrix

# Unconstrained samples for comparison
X_unconditional = rand(MvNormal(μ, Σ), n)     # 2×n matrix

# Plot
with_theme(theme_latexfonts(), fontsize = 18) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"X_1", ylabel = L"X_2", limits = (-8, 0, -5, 5))

    lines!(ax, [-8.0, 0.0], [-2.0, 6.0], color = :black, linestyle = :dash, label = L"X_1 - X_2 = -6")

    scatter!(ax, X_conditional[1, :], X_conditional[2, :], color = :red, markersize = 7, label = L"X \sim \mathcal{N}(\mu, \Sigma) \mid (X_1 - X_2 < -6)")
    scatter!(ax, X_unconditional[1, :],  X_unconditional[2, :],  color = :blue, markersize = 7, label = L"X \sim \mathcal{N}(\mu, \Sigma)")

    axislegend(ax, position = :lt)
    fig
end
```

![Conditional sampling with truncated MVN](assets/truncated_mvn_example.svg)