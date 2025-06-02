#=
Truncated multivariate normal distribution per reference below. Based on MATLAB implementation by Zdravko Botev and python implementation by Paul Brunzema (both linked below).

- MATLAB implementation: Zdravko Botev (2024). Truncated Normal and Student's t-distribution toolbox (https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox), MATLAB Central File Exchange. Retrieved May 24, 2024. 

- Python implementation: https://github.com/brunzema/truncated-mvn-sampler

=#

module TruncatedMVN

import LinearAlgebra: diag, I, diagm
import SpecialFunctions: erfcx, erfc, erfcinv
using NonlinearSolve
using StaticArrays
using DocStringExtensions

export TruncatedMVNormal
export sample

@template (FUNCTIONS, METHODS, MACROS) = """
                                         $(SIGNATURES)
                                         $(DOCSTRING)
                                         """
@template (TYPES,) = """
    $(TYPEDEF)
    $(DOCSTRING)
    """


"""
Truncated multivariate normal distribution with minimax tilting-based sampling. 
"""
struct TruncatedMVNormal{T,V<:AbstractVector{<:T},M<:AbstractMatrix{<:T}}
    dim::Int
    mu::V
    orig_mu::V
    cov::M
    lb::V
    ub::V
    orig_lb::V
    orig_ub::V
    L::M
    L_unscaled::M
    EPS::T
    perm::Vector{Int}
    x::V
    psistar::V

    @doc """
    Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](@ref) distribution.

    Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](@ref).

    # Arguments

    - `mu`: D-dimensional vector of means.
    - `cov`: DxD-dimensional covariance matrix.
    - `lb`: D-dimensional vector of lower bounds.
    - `ub`: D-dimensional vector of upper bounds.

    Bounds may be `-Inf`/`Inf`.

    """
    function TruncatedMVNormal(mu::V, cov::M, lb::V, ub::V) where {T<:Number,V<:AbstractVector{<:T},M<:AbstractArray{<:T}}
        dim = length(mu)
        if size(cov, 1) != size(cov, 2)
            throw(DimensionMismatch("cov matrix must be square"))
        end

        if length(lb) != dim || size(cov, 1) != dim || length(ub) != dim
            throw(DimensionMismatch("Dimensions of mu, lb, ub and cov must match each other"))
        end

        if any(ub .<= lb)
            throw(ArgumentError("All upper bounds (ub) must be greater than all lower bounds (lb)"))
        end

        orig_mu = copy(mu)


        lb_s = lb .- orig_mu
        ub_s = ub .- orig_mu

        L_unscaled, perm = colperm2!(dim, cov, mu, lb_s, ub_s)

        L, x, mu, psistar = compute_factors2!(L_unscaled, lb_s, ub_s, dim)


        new{T,V,M}(
            dim,
            mu,
            orig_mu,
            cov,
            lb_s,
            ub_s,
            lb, # Original lb
            ub, # Origingal ub
            L,
            L_unscaled,
            10.0e-15,
            perm,
            x,
            psistar
        )
    end # Inner TruncatedMVNormal constructor
end # TruncatedMVNormal struct

function Base.show(io::IO, d::TruncatedMVNormal)
    print(io,
        typeof(d), "\n",
        "mean: ", d.orig_mu, "\n",
        "ub: ", d.orig_ub, "\n",
        "lb: ", d.orig_lb, "\n",
        "cov: ", d.cov
    )
end

"""
Sample `n` samples from the [`TruncatedMVNormal`](@ref) distribution `d`.

Returns a D x n `Matrix` of samples where D is the dimension of the distribution `d`.
"""
function sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)

    accept, iteration = 0, 0

    # Preallocate constant StaticArrays for mvnrnd
    Smu = SVector{length(d.mu) + 1}(vcat(d.mu, [0.0]))
    SL = SMatrix{size(d.L)...}(d.L)
    Slb = SVector{length(d.lb)}(d.lb)
    Sub = SVector{length(d.ub)}(d.ub)


    # Preallocate normal arrays
    Z = zeros(Float64, d.dim, n)
    Zview = @view Z[:, begin:end]
    logpr = zeros(Float64, n)
    logprview = @view logpr[begin:end]


    # Preallocate output
    rv = Matrix{Float64}(undef, d.dim, n)
    rvindx = 1


    while accept < n
        mvnrnd!(Zview, logprview, d, Smu, SL, Slb, Sub)

        idx = @. -log($(rand(length(logprview)))) > (d.psistar - logprview)

        naccepted = count(idx)


        rv[:, rvindx:(rvindx+naccepted-1)] = Zview[:, idx]


        # rv = hcat(rv, Z[:, idx])

        # accept += size(rv, 2)
        accept += naccepted
        rvindx = accept + 1

        iteration += 1

        if iteration > 1000
            @warn "Acceptance prob. less than 0.001"
        elseif iteration > max_iter
            @warn "Max iterations $(max_iter) reached. Sample is only approximately distributed."
            accept = n
            rv[:, accept+1:end] = Zview[:, .!idx]
        end
        # reset and resize result arrays
        Zview = @view Z[:, begin:(n-accept)]
        fill!(Zview, 0.0)
        logprview = @view logpr[begin:(n-accept)]
        fill!(logprview, 0.0)
    end
    # Finish and postprocess
    order = sortperm(d.perm)
    rv = d.L_unscaled * rv
    rv = rv[order, :]

    # retransfer to original mean
    rv .+= d.orig_mu
    return rv
end

#Generates samples from a normal distribution.
function mvnrnd!(z::AbstractArray, logpr::AbstractArray, d::TruncatedMVNormal, mu::AbstractArray, L::AbstractArray, lb::AbstractArray, ub::AbstractArray)
    for k in 1:d.dim
        # Multiply L * Z
        col = L[[k], begin:k] * z[begin:k, :]
        # Limits of truncation
        tl = vec(@. lb[k] - mu[k] - col)
        tu = vec(@. ub[k] - mu[k] - col)

        z[k, :] = mu[k] .+ trandn(tl, tu)
        a = (@.($(lnNormalProb(tl, tu)) + 0.5 * mu[k]^2 - mu[k] * z[[k], :]))
        for i in eachindex(logpr)
            logpr[i] += a[i]
        end
    end
    return logpr, z
end


function trandn(lb::T, ub::T) where {T}
    length(lb) != length(ub) && throw(DimensionMismatch("Lengths of lb and ub must be equal"))
    x = similar(ub)

    a = 0.66 # Treshold from MATLAB implementation
    # Consider 3 cases
    idx1 = vec(lb .> a)
    if any(idx1)
        tl = lb[idx1]
        tu = ub[idx1]
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


function compute_factors2!(L_unscaled, lb, ub, dim)

    D = diag(L_unscaled)
    any(D .< 1.0e-15) && @warn "Method might fail as covariance matrix is singular!"

    scaled_L = L_unscaled ./ repeat(reshape(D, dim, 1), 1, dim)

    lb .= lb ./ D
    ub .= ub ./ D

    L = scaled_L - I

    x0 = zeros(2 * (dim - 1))
    p = [L, lb, ub]

    fun = NonlinearFunction(gradpsi, jac=jacpsi)
    prob = NonlinearProblem(fun, x0, p)
    sol = solve(prob)

    x = sol.u[begin:dim-1]
    mu = sol.u[dim:end]

    psistar = [psy2(L, lb, ub, x, mu)]

    return L, x, mu, psistar
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


function psy2(L, lb, ub, xd, mud)
    x = vcat(xd, [0.0])
    mu = vcat(mud, [0.0])

    c = L * x

    lt = @. lb - mu - c
    ut = @. ub - mu - c

    sum(lnNormalProb(lt, ut) .+ 0.5 .* mu .^ 2 .- x .* mu)
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

function colperm2!(dim, cov, orig_mu, lb, ub)
    perm = collect(1:dim)
    L = fill(0.0, size(cov))
    z = fill(0.0, length(orig_mu))

    for j in deepcopy(perm)
        pr = fill(Inf, size(z))
        i = j:dim
        D = diag(cov)
        s = D[i] .- sum(L[i, 1:j] .^ 2, dims=2)
        s[s.<0.0] .= 1.0e-15
        @. s = sqrt(s)

        tl = (lb[i] .- L[i, 1:j] * z[1:j]) ./ s
        tu = (ub[i] .- L[i, 1:j] * z[1:j]) ./ s
        pr[i] = lnNormalProb(tl, tu)

        k = argmin(pr)

        jk = [j, k]
        kj = [k, j]

        cov[jk, :] = cov[kj, :]
        cov[:, jk] = cov[:, kj]

        L[jk, :] = L[kj, :]

        lb[jk] = lb[kj]
        ub[jk] = ub[kj]
        perm[jk] = perm[kj]


        s = cov[j, j] - sum(L[j, 1:j] .^ 2)
        if s < -0.01
            throw(DomainError(s, "Sigma is not a positive semi-definite"))
        elseif s < 0.0
            s = 1.0e-15
        end
        L[j, j] = sqrt(s)
        new_L = cov[(j+1):dim, j] - L[(j+1):dim, 1:j] * L[j, 1:j]
        L[(j+1):dim, j] = new_L ./ L[j, j]

        tl = ((lb[j] .- L[[j], 1:j] * z[1:j]) ./ L[j, j])
        tu = ((ub[j] .- L[[j], 1:j] * z[1:j]) ./ L[j, j])

        w = lnNormalProb(tl, tu)
        z[j] = (@. exp(-0.5 * tl[1]^2 - w[1]) - exp.(-0.5 * tu[1]^2 - w[1])) / sqrt(2π)

    end
    return L, perm
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
