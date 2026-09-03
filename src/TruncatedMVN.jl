#=
Truncated multivariate normal distribution per reference below. Based on MATLAB implementation by Zdravko Botev and python implementation by Paul Brunzema (both linked below).

- MATLAB implementation: Zdravko Botev (2024). Truncated Normal and Student's t-distribution toolbox (https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox), MATLAB Central File Exchange. Retrieved May 24, 2024. 

- Python implementation: https://github.com/brunzema/truncated-mvn-sampler

=#

module TruncatedMVN

import LinearAlgebra: diag, I, diagm, mul!
import SpecialFunctions: erfcx, erfc, erfcinv, expm1
using NonlinearSolve
using Random

export TruncatedMVNormal
export sample

const INV_SQRT2 = 0.7071067811865476   # 1/sqrt(2)


"""


Truncated multivariate normal distribution with minimax tilting-based sampling.

"""
mutable struct TruncatedMVNormal{T <: AbstractFloat}
    dim::Int
    mu::Vector{T}
    orig_mu::Vector{T}
    cov::Matrix{T}
    lb::Vector{T}
    ub::Vector{T}
    orig_lb::Vector{T}
    orig_ub::Vector{T}
    L::Matrix{T} 
    L_unscaled::Matrix{T}
    EPS::T
    perm::Vector{Int}
    x::Vector{T} 
    psistar::Vector{T} 

    @doc """
         TruncatedMVNormal(mu::AbstractVector{T}, cov::AbstractMatrix{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where {T <:AbstractFloat}

    Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](@ref) distribution.

    Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](@ref).

    # Arguments

    - `mu::T`: D-dimensional vector of means.
    - `cov::S`: DxD-dimensional covariance matrix.
    - `lb::T`: D-dimensional vector of lower bounds.
    - `ub::T`: D-dimensional vector of upper bounds.

    Bounds may be `-Inf`/`Inf`.

    """
    function TruncatedMVNormal(mu::AbstractVector{T}, cov::AbstractMatrix{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where {T <:AbstractFloat}
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

        new{T}(d, Vector{T}(), copy(mu), copy(cov), lb .- mu, ub .- mu, lb, ub, similar(cov), similar(cov), 10.0e-15, Int[], T[], T[])
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
    sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)

Sample `n` samples from the distribution `d`.

Returns an D x n `Matrix` of samples where D is the dimension of the distribution `d`.
"""
function sample(d::TruncatedMVNormal{T}, n::Integer, max_iter::Integer=10000) where T
    return sample(Random.default_rng(), d, n, max_iter) 
end
function sample(rng::AbstractRNG, d::TruncatedMVNormal{T}, n::Integer, max_iter::Integer=10000) where T
    if isempty(d.psistar)
        compute_factors!(d)
    end

    accept, iteration = 0, 0

    # Preallocate constant StaticArrays for mvnrnd
    #Smu = SVector{length(d.mu) + 1}(vcat(d.mu, [0.0]))
    #SL = SMatrix{size(d.L)...}(d.L)
    #Slb = SVector{length(d.lb)}(d.lb)
    #Sub = SVector{length(d.ub)}(d.ub)

    # Preallocate normal arrays
    Z = zeros(Float64, d.dim, n)
    Zview = @view Z[:, begin:end]
    logpr = zeros(Float64, n)
    logprview = @view logpr[begin:end]

    # Preallocate output
    rv = Matrix{Float64}(undef, d.dim, n)
    rvindx = 1


    while accept < n
        mvnrnd!(rng, Zview, logprview, d)

        idx = @. -log($(rand(rng, length(logprview)))) > (d.psistar - logprview)

        naccepted = count(idx)

        rv[:, rvindx:(rvindx+naccepted-1)] = Zview[:, idx]

        # rv = hcat(rv, Z[:, idx])
        # accept += size(rv, 2)
        accept += naccepted
        rvindx = accept + 1

        iteration += 1

        if accept < n                     
            if iteration >= max_iter       
                @warn "Max iterations $(max_iter) reached. Sample is only approximately distributed."
                rv[:, rvindx:n] = Zview[:, .!idx]  
                accept = n
                break         
            elseif iteration > 1000
                @warn "Acceptance prob. less than 0.001" maxlog = 1 
            end
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

"""
    mvnrnd!(z::AbstractArray, logpr::AbstractArray, d::TruncatedMVNormal, mu::AbstractArray, L::AbstractArray, lb::AbstractArray, ub::AbstractArray)

Generates samples from a normal distribution.
"""
function mvnrnd!(rng, z::AbstractArray, logpr::AbstractArray, d::TruncatedMVNormal{T}) where T
    n   = size(z, 2)
    col = Vector{T}(undef, n)
    tl  = Vector{T}(undef, n)
    tu  = Vector{T}(undef, n)
    #=
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
    =#
    for k in 1:d.dim 
        # Multiply L * Z
        if k == 1
            fill!(col, zero(T))
        else
            mul!(col, transpose(view(z, 1:k-1, :)), view(d.L, k, 1:k-1))
        end
        # Limits of truncation
        mk   = d.mu[k]
        lk   = d.lb[k]
        uk   = d.ub[k]
        hmk2 = T(0.5) * mk * mk
        @inbounds @simd for j in 1:n
            tl[j] = lk - mk - col[j]
            tu[j] = uk - mk - col[j]
        end
        zk      = trandn(rng, tl, tu)
        w       = lnNormalProb(tl, tu)
        @inbounds @simd for j in 1:n
            z[k, j]   = mk + zk[j]
            logpr[j] += w[j] + hmk2 - mk * z[k, j]  
        end
    end
    return logpr, z
end


@inline function trandn(rng, l::T, u::T) where T <: AbstractFloat
    a = T(0.66)                      # threshold from the MATLAB original
    if l > a
        return ntail(rng, l, u)
    elseif u < -a
        return -ntail(rng, -u, -l)
    else
        return tn(rng, l, u)
    end
end
function trandn(rng, lb::AbstractArray{T}, ub::AbstractArray{T}) where T
    length(lb) != length(ub) && throw(DimensionMismatch("Lengths of lb and ub must be equal"))
    x = similar(ub)

    a = T(0.66) # Threshold from MATLAB implementation
    # Consider 3 cases
    idx1 = lb .> a
    if any(idx1)
        tl = lb[idx1]
        tu = ub[idx1]
        x[idx1] = ntail(rng, tl, tu)
    end
    idx2 = ub .< -a
    if any(idx2)
        tl = -ub[idx2]
        tu = -lb[idx2]
        x[idx2] = -ntail(rng, tl, tu)
    end
    idx3 = .!(idx1 .| idx2)
    if any(idx3)
        tl = lb[idx3]
        tu = ub[idx3]
        x[idx3] = tn(rng, tl, tu)
    end
    return x
    
end
@inline function tn(rng, l::T, u::T) where T <: AbstractFloat
    if u - l > T(2)
        return trnd(rng, l, u)
    else
        iv = T(INV_SQRT2)
        pl = erfc(l * iv) / 2
        pu = erfc(u * iv) / 2
        return sqrt(T(2)) * erfcinv(2 * (pl - (pl - pu) * rand(rng)))
    end
end

function tn(rng, lb::AbstractArray{T}, ub::AbstractArray{T}, sw::T = T(2)) where T
    x = similar(ub)
    # abs(ub-lb) > sw -> use accept-reject
    idx1 = @. abs(ub - lb) > sw
    if any(idx1)
        tl = lb[idx1]
        tu = ub[idx1]
        x[idx1] = trnd(rng, tl, tu)
    end
    # For other cases use inverse-transform
    idx2 = .!idx1
    if any(idx2)
        tl = lb[idx2]
        tu = ub[idx2]
        pl = @. erfc(tl / sqrt(2)) / 2
        pu = @. erfc(tu / sqrt(2)) / 2
        x[idx2] = @. sqrt(2) * erfcinv(2 * (pl - (pl - pu) * $(rand(rng, length(tl)))))
    end
    return x
end

@inline function trnd(rng, l::T, u::T) where T <: AbstractFloat
    while true
        x = randn(rng)
        (x >= l) & (x <= u) && return x
    end
end
function trnd(rng, lb::AbstractArray{T}, ub::AbstractArray{T}) where T <: AbstractFloat
    x = randn(rng, length(lb))

    test = @. (x < lb) | (x > ub)
    idx = findall(test)
    d = length(idx)
    while d > 0
        ly = lb[idx]
        uy = ub[idx]
        y = randn(rng, length(uy))
        idx2 = @. (y > ly) & (y < uy)
        x[idx[idx2]] = y[idx2]
        idx = idx[.!idx2]
        d = length(idx)
    end

    return x
    
end

# tail case: Rayleigh proposal with accept-reject (Botev 2017, sec. 3)
@inline function ntail(rng, l::T, u::T) where T <: AbstractFloat
    c = T(0.5) * l * l
    f = expm1(c - T(0.5) * u * u)  
    while true
        x = c - log1p(rand(rng) * f)
        r = rand(rng)
        r * r * x < c && return sqrt(2 * x)
    end
end
function ntail(rng, lb::AbstractArray{T}, ub::AbstractArray{T}) where {T}
    c = @. lb^2 / 2
    n = length(lb)
    f = @. expm1(c - ub^2 / 2)
    x = @. c - log(1 + $(rand(rng, n)) * f)
    props = @. ($(rand(rng, n))^2 * x)
    rejected = findall(props .> c) # Find rejected
    d = length(rejected)
    while d > 0
        cy = c[rejected]
        y = @. cy - log(1 + $(rand(rng, d)) * f[rejected])
        idx = findall((rand(rng, d) .^ 2 .* y) .< cy) # Find accepted
        x[rejected[idx]] = y[idx]
        deleteat!(rejected, idx)
        d = length(rejected)
    end
    return @. sqrt(2 * x)
end

function compute_factors!(d::TruncatedMVNormal{T}) where T
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
    d.mu = push!(collect(T, sol.u[d.dim:end]), zero(T))

    d.psistar = [psy(d, d.x, d.mu)]

end

function psy(d::TruncatedMVNormal, xd::AbstractArray{T}, mu::AbstractArray{T}) where T
    x = vcat(xd, zeros(T, 1))
    c = d.L * x
    #lt = @. d.lb - mu - c
    #ut = @. d.ub - mu - c
    #sum(lnNormalProb(lt, ut) .+ 0.5 .* mu .^ 2 .- x .* mu)
    #lnp = lnNormalProb(lt, ut)
    #sum(@. lnp + 0.5 * mu * mu - x * mu)
    result = zero(T)
    @inbounds @simd for i in 1:length(c)
        mui = mu[i]
        ci  = c[i]
        result += lnNormalProb(d.lb[i] - mui - ci, d.ub[i] - mui - ci) + 0.5 * mui * mui - x[i] * mui
    end
    return result 
end

function gradpsi(y, p)
    L, l, u = p
    d = length(u)
    c = zeros(Float64, d)
    mu = copy(c)
    x = copy(c)

    x[begin:d-1] = y[begin:d-1]
    mu[begin:d-1] = y[d:end]

    c[2:d] = view(L, 2:d, :) * x
    lt = @. l - mu - c
    ut = @. u - mu - c

    w = lnNormalProb(lt, ut)
    pl = @. exp(-0.5 * lt^2 - w) / sqrt(2π)
    pu = @. exp(-0.5 * ut^2 - w) / sqrt(2π)
    P = pl - pu

    # Gradient
    dfdx = -mu[1:d-1] + transpose((transpose(P) * view(L, :, 1:d-1)))
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

    c[2:d] = view(L, 2:d, :) * x
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

        Li1j = view(L, i, 1:j)

        s = D[i] .- sum(Li1j .^ 2, dims=2)
        s[s .< 0.0] .= 1.0e-15
        @. s = sqrt(s)

        Li1jz = Li1j * z[1:j]

        tl = (d.lb[i] .- Li1jz) ./ s
        tu = (d.ub[i] .- Li1jz) ./ s
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


        s = d.cov[j, j] - sum(abs2, view(L, j, 1:j))
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


function lnNormalProb(a::T, b::T) where T <: AbstractFloat
    iv = T(INV_SQRT2)
    if a > zero(T)
        ea   = erfcx(iv * a)
        base = muladd(-T(0.5) * a, a, log(T(0.5) * ea))
        isinf(b) && return base
        eb = erfcx(iv * b)
        return base + log1p(-exp(T(0.5) * (a - b) * (a + b)) * (eb / ea))
    elseif b < zero(T)
        ea   = erfcx(-iv * b)
        base = muladd(-T(0.5) * b, b, log(T(0.5) * ea))
        isinf(a) && return base
        eb = erfcx(-iv * a)
        return base + log1p(-exp(T(0.5) * (b - a) * (b + a)) * (eb / ea))
    else
        return log1p(-T(0.5) * erfc(-iv * a) - T(0.5) * erfc(iv * b))
    end
end

lnNormalProb(a::AbstractArray, b::AbstractArray) = lnNormalProb.(a, b)

"""
    lnNormalProb(a, b)

Accurately compute `ln(P(a<Z<b))` `where Z~N(0,1)`.
"""
#=
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
=#
function lnPhi(x)
    @. -0.5 * x^2 - log(2) + log(erfcx(x / sqrt(2)) + 1.0e-15)
end

end
