#=
Truncated multivariate normal distribution per reference below. Based on MATLAB implementation by Zdravko Botev and python implementation by Paul Brunzema (both linked below).

- MATLAB implementation: Zdravko Botev (2024). Truncated Normal and Student's t-distribution toolbox (https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox), MATLAB Central File Exchange. Retrieved May 24, 2024. 

- Python implementation: https://github.com/brunzema/truncated-mvn-sampler

=#

module TruncatedMVN

import LinearAlgebra: diag, I, diagm
import SpecialFunctions: erfcx, erfc, erfcinv
using NonlinearSolve

export TruncatedMVNormal
export sample

"""
    TruncatedMVNormal{S<:AbstractArray{<:AbstractFloat},T<:AbstractVector{<:AbstractFloat},U<:Integer,V<:AbstractFloat,P<:AbstractVector{<:Integer}}

Truncated multivariate normal distribution with minimax tilting-based sampling.

"""
mutable struct TruncatedMVNormal{S<:AbstractArray{<:AbstractFloat},T<:AbstractVector{<:AbstractFloat},U<:Integer,V<:AbstractFloat,P<:AbstractVector{<:Integer}}
    dim::U
    mu::T
    orig_mu::T
    cov::S
    lb::T
    ub::T
    orig_lb::T
    orig_ub::T
    L::S
    L_unscaled::S
    EPS::V
    perm::P
    x::T
    psistar::T

    @doc """
         TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}

    Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](@ref) distribution.

    Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](@ref).

    # Arguments

    - `mu::T`: D-dimensional vector of means.
    - `cov::S`: DxD-dimensional covariance matrix.
    - `lb::T`: D-dimensional vector of lower bounds.
    - `ub::T`: D-dimensional vector of upper bounds.

    Bounds may be `-Inf`/`Inf`.

    """
    function TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
        d = length(mu)
        if size(cov, 1) != size(cov, 2)
            throw(DimensionMismatch("cov matrix must be square"))
        end

        if length(lb) != d || size(cov, 1) != d || length(ub) != d
            throw(DimensionMismatch("Dimensions of mu, lb, ub and cov must match each other"))
        end

        if any(ub .<= lb)
            throw(ArgumentError("All upper bounds (ub) must be greater than all lower bounds (lb)"))
        end

        new{typeof(cov),typeof(mu),typeof(d),Float64,Vector{Int64}}(d, Vector{eltype(mu)}(), mu, cov, lb .- mu, ub .- mu, lb, ub, similar(cov), similar(cov), 10.0e-15, [], [], [])
    end # Inner TruncatedMVNormal constructor
end # TruncatedMVNormal struct

"""
    sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)

Sample `n` samples from the distribution `d`.

Returns an D x n `Matrix` of samples where D is the dimension of the distribution `d`.
"""
function sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)
    if isempty(d.psistar)
        compute_factors!(d)
    end

    rv = Matrix{Float64}(undef, d.dim, 0)

    accept, iteration = 0, 0

    while accept < n
        logpr, Z = mvnrnd(d, n, d.mu)

        idx = @. -log($(rand(n))) > (d.psistar - logpr)

        rv = hcat(rv, Z[:, idx])

        accept += size(rv, 2)

        iteration += 1

        if iteration > 1000
            @warn "Acceptance prob. less than 0.001"
        elseif iteration > max_iter
            @warn "Max iterations $(max_iter) reached. Sample is only approximately distributed."
            accept = n
            rv = hcat(rv, Z)
        end
    end
    # Finish and postprocess
    order = sortperm(d.perm)
    rv = rv[:, begin:n]
    rv = d.L_unscaled * rv
    rv = rv[order, :]

    # retransfer to original mean
    rv .+= repeat(reshape(d.orig_mu, (d.dim, 1)), 1, size(rv, 2))
    return rv
end


function mvnrnd(d::TruncatedMVNormal, n::Integer, mud)
    mu = deepcopy(mud)
    push!(mu, 0.0)
    z = zeros(Float64, d.dim, n)
    logpr = fill(0.0, n)
    for k in 1:d.dim
        # Multiply L * Z
        col = d.L[[2], begin:k] * z[begin:k, :]
        # Limits of truncation
        tl = @. d.lb[k] - mu[k] - col
        tu = @. d.ub[k] - mu[k] - col

        z[k, :] = mu[k] .+ trandn(tl, tu)

        logpr .+= (@.($(lnNormalProb(tl, tu)) + 0.5 * mu[k]^2 - mu[k] * z[k, :]))[1]
    end
    return logpr, z
end

function trandn(lb::T, ub::T) where {T}
    length(lb) != length(ub) && throw(DimensionMismatch("Lengths of lb and ub must be equal"))

    x = similar(ub)

    a = 0.66 # Treshold from MATLAB implementation
    l = reshape(lb, (1, length(lb)))
    u = reshape(ub, (1, length(ub)))
    # Consider 3 cases
    idx1 = vec(l .> a)
    if any(idx1)
        tl = l[:, idx1]
        tu = u[:, idx1]
        x[idx1] = ntail(tl, tu)
    end
    idx2 = vec(ub .< -a)
    if any(idx2)
        tl = -ub[idx2]
        tu = -lb[idx2]
        x[idx2] = ntail(tl, tu)
    end
    idx3 = .!(idx1 .| idx2)
    if any(idx3)
        tl = lb[idx3]
        tu = ub[idx3]
        x[idx3] = tn(tl, tu)
    end
    return x
end

function tn(lb::T, ub::T, sw=2.0) where {T}
    x = similar(ub)
    # abs(ub-lb) > sw -> use accept-reject
    idx1 = @. abs(ub - lb) > sw
    if any(idx1)
        tl = lb[idx1]
        tu = ub[idx1]
        x[idx1] = trnd(tl, tu)
    end
    # For other cases use inverse-transform
    idx2 = .!idx1
    if any(idx2)
        tl = lb[idx2]
        tu = ub[idx2]
        pl = @. erfc(tl / sqrt(2)) / 2
        pu = @. erfc(tu / sqrt(2)) / 2
        x[idx2] = @. sqrt(2) * erfcinv(2 * (pl - (pl - pu) * $(rand(length(tl)))))
    end
    return x
end

function trnd(lb::T, ub::T) where {T}
    x = randn(length(lb))

    test = @. (x < lb) | (x > ub)
    idx = findall(test)
    d = length(idx)
    while d > 0
        ly = lb[idx]
        uy = ub[idx]
        y = randn(length(uy))
        idx2 = @. (y > ly) & (y < uy)
        x[idx[idx2]] = y[idx2]
        idx = idx[.!idx2]
        d = length(idx)
    end

    return x
end

function ntail(lb::T, ub::T) where {T}
    c = @. lb^2 / 2
    n = length(lb)
    f = @. exp(c - ub^2 / 2) - 1
    x = @. c - log(1 + $(rand(n)) * f)
    props = @. ($(rand(n))^2 * x)
    rejected = findall(props .> c) # Find rejected
    d = length(rejected)
    while d > 0
        cy = c[rejected]
        y = @. cy - log(1 + $(rand(d)) * f[rejected])
        idx = findall((rand(d) .^ 2 .* y) .< cy) # Find accepted
        x[rejected[idx]] = y[idx]
        deleteat!(rejected, idx)
        d = length(rejected)
    end

    return @. sqrt(2 * x)
end

function compute_factors!(d::TruncatedMVNormal)
    d.L_unscaled, d.perm = colperm!(d)

    D = diag(d.L_unscaled)
    any(D .< 1.0e-15) && @warn "Method might fail as covariance matrix is singular!"

    scaled_L = d.L_unscaled ./ repeat(reshape(D, d.dim, 1), 1, d.dim)

    d.lb = d.lb ./ D
    d.ub = d.ub ./ D

    d.L = scaled_L - I

    x0 = zeros(2 * (d.dim - 1))
    p = [d.L, d.lb, d.ub]

    fun = NonlinearFunction(gradpsi, jac=jacpsi)
    prob = NonlinearProblem(fun, x0, p)
    sol = solve(prob)

    d.x = sol.u[begin:d.dim-1]
    d.mu = sol.u[d.dim:end]

    d.psistar = [psy(d, d.x, d.mu)]

end

function psy(d::TruncatedMVNormal, xd, mud)
    x = vcat(xd, [0.0])
    mu = vcat(mud, [0.0])

    c = d.L * x

    lt = @. d.lb - mu - c
    ut = @. d.ub - mu - c

    sum(lnNormalProb(lt, ut) .+ 0.5 .* mu .^ 2 .- x .* mu)
end

function gradpsi(y, p)
    L, l, u = p
    d = length(u)
    c = zeros(Float64, d)
    mu = deepcopy(c)
    x = deepcopy(c)

    x[begin:d-1] = y[begin:d-1]
    mu[begin:d-1] = y[d:end]

    c[2:d] = L[2:d, :] * x
    lt = @. l - mu - c
    ut = @. u - mu - c

    w = lnNormalProb(lt, ut)
    pl = @. exp(-0.5 * lt^2 - w) / sqrt(2π)
    pu = @. exp(-0.5 * ut^2 - w) / sqrt(2π)
    P = pl - pu

    # Gradient
    dfdx = -mu[1:d-1] + transpose((transpose(P) * L[:, 1:d-1]))
    dfdm = @. mu - x + P
    grad = cat(dfdx, dfdm[begin:end-1], dims=1)
    # NOTE: indexing is wrong here or in compute_factors.
    return grad
end

function jacpsi(y, p)
    L, l, u = p
    d = length(u)
    c = zeros(Float64, d)
    mu = deepcopy(c)
    x = deepcopy(c)

    x[begin:d-1] = y[begin:d-1]
    mu[begin:d-1] = y[d:end]

    c[2:d] = L[2:d, :] * x
    lt = @. l - mu - c
    ut = @. u - mu - c

    w = lnNormalProb(lt, ut)
    pl = @. exp(-0.5 * lt^2 - w) / sqrt(2π)
    pu = @. exp(-0.5 * ut^2 - w) / sqrt(2π)
    P = pl - pu

    # Jacobian
    lt[isinf.(lt)] .= 0.0
    ut[isinf.(ut)] .= 0.0

    dP = @. -P^2 + lt * pl - ut * pu
    DL = repeat(reshape(dP, (d, 1)), 1, d) .* L
    mx = DL - I
    xx = transpose(L) * DL
    mx = mx[begin:end-1, begin:end-1]
    xx = xx[begin:end-1, begin:end-1]


    out = hvcat((2, 2), xx, transpose(mx), mx, diagm(1 .+ dP[begin:end-1]))
    return out
end

function colperm!(d::TruncatedMVNormal)
    perm = collect(1:d.dim)
    L = fill(0.0, size(d.cov))
    z = fill(0.0, length(d.orig_mu))

    for j in deepcopy(perm)
        pr = fill(Inf, size(z))
        i = j:d.dim
        D = diag(d.cov)
        s = D[i] .- sum(L[i, 1:j] .^ 2, dims=2)
        s[s.<0.0] .= 1.0e-15
        @. s = sqrt(s)

        tl = (d.lb[i] .- L[i, 1:j] * z[1:j]) ./ s
        tu = (d.ub[i] .- L[i, 1:j] * z[1:j]) ./ s
        pr[i] = lnNormalProb(tl, tu)

        k = argmin(pr)

        jk = [j, k]
        kj = [k, j]

        d.cov[jk, :] = d.cov[kj, :]
        d.cov[:, jk] = d.cov[:, kj]

        L[jk, :] = L[kj, :]

        d.lb[jk] = d.lb[kj]
        d.ub[jk] = d.ub[kj]
        perm[jk] = perm[kj]


        s = d.cov[j, j] - sum(L[j, 1:j] .^ 2)
        if s < -0.01
            throw(DomainError(s, "Sigma is not a positive semi-definite"))
        elseif s < 0.0
            s = 1.0e-15
        end
        L[j, j] = sqrt(s)
        new_L = d.cov[(j+1):d.dim, j] - L[(j+1):d.dim, 1:j] * L[j, 1:j]
        L[(j+1):d.dim, j] = new_L ./ L[j, j]

        tl = ((d.lb[j] .- L[[j], 1:j] * z[1:j]) ./ L[j, j])
        tu = ((d.ub[j] .- L[[j], 1:j] * z[1:j]) ./ L[j, j])

        w = lnNormalProb(tl, tu)
        z[j] = (@. exp(-0.5 * tl[1]^2 - w[1]) - exp.(-0.5 * tu[1]^2 - w[1])) / sqrt(2π)

    end
    return L, perm
end

#=
    lnNormalProb(a, b)

Accurately compute `ln(P(a<Z<b))` `where Z~N(0,1)`.
=#
function lnNormalProb(a::T, b::T) where {T}
    p = zeros(eltype(a), size(a))

    # b>a>0
    idx1 = a .> zero(eltype(a))
    if any(idx1)
        pa = lnPhi(a[idx1])
        pb = lnPhi(b[idx1])
        @. p[idx1] = pa + log1p(-exp(pb - pa))
    end

    # a<b<0
    idx2 = b .< zero(eltype(b))
    if any(idx2)
        pa = lnPhi(-a[idx2])
        pb = lnPhi(-b[idx2])
        @. p[idx2] = pb + log1p(-exp(pa - pb))
    end

    # a<0<b
    idx3 = @. !idx1 && !idx2
    if any(idx3)
        pa = @. erfc(-a[idx3] / sqrt(2)) / 2
        pb = @. erfc(b[idx3] / sqrt(2)) / 2
        @. p[idx3] = log1p(-pa - pb)
    end

    return p

end

function lnPhi(x)
    @. -0.5 * x^2 - log(2) + log(erfcx(x / sqrt(2)) + 1.0e-15)
end

end
