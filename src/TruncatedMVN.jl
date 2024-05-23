module TruncatedMVN

using Distributions
import LinearAlgebra: diag, I
import SpecialFunctions: erfcx, erfc
using NonlinearSolve

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

    function TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
        d = length(mu)
        if size(cov, 1) != size(cov, 2)
            throw(DimensionMismatch("cov matrix must be square!"))
        end

        if length(lb) != d || size(cov, 1) != d || length(ub) != d
            throw(DimensionMismatch("Dimensions of mu, lb, ub and cov must match each other!"))
        end

        if any(ub .<= lb)
            throw(ArgumentError("All upper bounds (ub) must be greater than all lower bounds (lb)!"))
        end

        new{typeof(cov),typeof(mu),typeof(d),Float64,Vector{Int64}}(d, Vector{eltype(mu)}(), mu, cov, lb .- mu, ub .- mu, lb, ub, similar(cov), similar(cov), 10.0e-15, [], [], [])
    end # Inner TruncatedMVNormal constructor
end # TruncatedMVNormal struct

function sample(d::TruncatedMVNormal, n::Integer)
    if isempty(d.pistar)
        compute_factors!(d)
    end

    rv = Matrix{Float64}(undef, d.dim, 0)

    accept, iteration = 0, 0

    while accept < n
        #

    end
end

function compute_factors!(d::TruncatedMVNormal)
    d.L_unscaled, d.perm = colperm(d)

    D = diag(d.L_unscaled)
    any(D .< 1.0e-15) && @warn "Method might fail as covariance matrix is singular!"

    scaled_L = d.L_unscaled ./ repeat(reshape(D, d.dim, 1), 1, d.dim)

    d.lb = d.lb ./ D
    d.ub = d.ub ./ D

    d.L = scaled_L - I

    x0 = zeros(2 * d.dim)
    p = [d.L, d.lb, d.ub]

    fun = NonlinearFunction(gradpsi, jac=jacpsi)
    prob = NonlinearProblem(fun, x0, p)
    sol = solve(prob)

    d.x = sol.u[begin:d.dim]
    d.mu = sol.u[(d.dim+1):end]

    d.psistar = psy(d, d.x, d.mu)

end

function psy(d::TruncatedMVNormal, x, mu)
    x = deepcopy(x)
    mu = deepcopy(mu)
    x[d.dim] = 0.0
    mu[d.dim] = 0.0

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

    x = y[begin:d]
    mu = y[(d+1):end]

    c[2:d-1] = L[2:d-1, :] * x
    lt = @. l - mu - c
    ut = @. u - mu - c

    w = lnNormalProb(lt, ut)
    pl = @. exp(-0.5 * lt^2 - w) / sqrt(2π)
    pu = @. exp(-0.5 * ut^2 - w) / sqrt(2π)
    P = pl - pu

    # Gradient
    dfdx = -mu[1:d] + transpose((transpose(P) * L[:, 1:d]))
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

    x = y[begin:d]
    mu = y[(d+1):end]

    c[2:d-1] = L[2:d-1, :] * x
    lt = @. l - mu - c
    ut = @. u - mu - c

    w = lnNormalProb(lt, ut)
    pl = @. exp(-0.5 * lt^2 - w) / sqrt(2π)
    pu = @. exp(-0.5 * ut^2 - w) / sqrt(2π)
    P = pl - pu

    # Jacobian
    @view(lt[isinf.(lt)]) .= 0.0
    @view(ut[isinf.(ut)]) .= 0.0

    dP = @. -P^2 + lt * pl - ut * pu
    DL = repeat(reshape(dP, (d, 1)), 1, d) .* L
    mx = DL - I
    xx = transpose(L) * DL
    mx = mx[begin:end-1, begin:end-1]
    xx = xx[begin:end-1, begin:end-1]

    hvcat((2, 2), xx, transpose(mx), mx, 1 .+ dP[begin:end-1])
end

function colperm(d::TruncatedMVNormal)
    perm = collect(1:d.dim)
    L = fill(0.0, size(d.cov))
    z = fill(0.0, length(d.orig_mu))

    for j in perm
        pr = fill(Inf, size(z))
        i = j:d.dim
        D = diag(d.cov)
        s = D[i] .- sum(L[i, 1:j] .^ 2, dims=2)
        s[s.<0.0] .= 0.0
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
        elseif s < 0
            s = 1.0e-15
        end
        L[j, j] = sqrt(s)
        new_L = d.cov[(j+1):d.dim, j] - L[(j+1):d.dim, 1:j] * L[j, 1:j]
        L[(j+1):d.dim, j] = new_L ./ L[j, j]

        tl = ((d.lb[j] .- L[[j], 1:(j)] * z[1:j]) ./ L[j, j])
        tu = ((d.ub[j] .- L[[j], 1:(j)] * z[1:j]) ./ L[j, j])

        w = lnNormalProb(tl, tu)
        z[j] = (@. exp(-0.5 * tl[1]^2 - w[1]) - exp.(-0.5 * tu[1]^2 - w[1])) / sqrt(2π)

    end
    return L, perm
end

"""
    lnNormalProb(a, b)

Accurately compute `ln(P(a<Z<b))` `where Z~N(0,1)`.
"""
function lnNormalProb(a, b)
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
