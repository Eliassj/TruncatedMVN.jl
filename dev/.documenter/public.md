
# Public API {#Public-API}

Documentation for the public API of TruncatedMVN.jl.
<details class='jldocstring custom-block' open>
<summary><a id='TruncatedMVN.TruncatedMVNormal' href='#TruncatedMVN.TruncatedMVNormal'><span class="jlbinding">TruncatedMVN.TruncatedMVNormal</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
TruncatedMVNormal{S<:AbstractArray{<:AbstractFloat},T<:AbstractVector{<:AbstractFloat},U<:Integer,V<:AbstractFloat,P<:AbstractVector{<:Integer}}
```


Truncated multivariate normal distribution with minimax tilting-based sampling.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/ca3c4921d015a31e9797b49333a761f594889450/src/TruncatedMVN.jl#L20-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:AbstractVector{<:AbstractFloat}, S<:AbstractArray{<:AbstractFloat}}' href='#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:AbstractVector{<:AbstractFloat}, S<:AbstractArray{<:AbstractFloat}}'><span class="jlbinding">TruncatedMVN.TruncatedMVNormal</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
 TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
```


Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](/public#TruncatedMVN.TruncatedMVNormal) distribution.

Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](/public#TruncatedMVN.sample).

**Arguments**
- `mu::T`: D-dimensional vector of means.
  
- `cov::S`: DxD-dimensional covariance matrix.
  
- `lb::T`: D-dimensional vector of lower bounds.
  
- `ub::T`: D-dimensional vector of upper bounds.
  

Bounds may be `-Inf`/`Inf`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/ca3c4921d015a31e9797b49333a761f594889450/src/TruncatedMVN.jl#L42-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='TruncatedMVN.sample' href='#TruncatedMVN.sample'><span class="jlbinding">TruncatedMVN.sample</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)
```


Sample `n` samples from the distribution `d`.

Returns an D x n `Matrix` of samples where D is the dimension of the distribution `d`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/ca3c4921d015a31e9797b49333a761f594889450/src/TruncatedMVN.jl#L87-L93" target="_blank" rel="noreferrer">source</a></Badge>

</details>

