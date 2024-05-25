


# TruncatedMVN {#TruncatedMVN}

Documentation for [TruncatedMVN](https://github.com/Eliassj/TruncatedMVN.jl).
- [`TruncatedMVN.TruncatedMVNormal`](#TruncatedMVN.TruncatedMVNormal)
- [`TruncatedMVN.TruncatedMVNormal`](#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S},%20Tuple{T},%20Tuple{T,%20S,%20T,%20T}}%20where%20{T<:(AbstractVector{<:AbstractFloat}),%20S<:(AbstractArray{<:AbstractFloat})})
- [`TruncatedMVN.sample`](#TruncatedMVN.sample)

<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.TruncatedMVNormal' href='#TruncatedMVN.TruncatedMVNormal'>#</a>&nbsp;<b><u>TruncatedMVN.TruncatedMVNormal</u></b> &mdash; <i>Type</i>.




```julia
TruncatedMVNormal{S<:AbstractArray{<:AbstractFloat},T<:AbstractVector{<:AbstractFloat},U<:Integer,V<:AbstractFloat,P<:AbstractVector{<:Integer}}
```


Truncated multivariate normal distribution with minimax tilting-based sampling.


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/beb130f96afc3dbb70432305a3474bff3a72f091/src/TruncatedMVN.jl#L16-L21)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}' href='#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}'>#</a>&nbsp;<b><u>TruncatedMVN.TruncatedMVNormal</u></b> &mdash; <i>Method</i>.




```julia
TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
```


Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](/index#TruncatedMVN.TruncatedMVNormal) distribution.

**Arguments**
- `mu::T`: D-dimensional vector of means.
  
- `cov::S`: DxD-dimensional covariance matrix.
  
- `lb::T`: D-dimensional vector of lower bounds.
  
- `ub::T`: D-dimensional vector of upper bounds.
  


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/beb130f96afc3dbb70432305a3474bff3a72f091/src/TruncatedMVN.jl#L38-L49)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.sample' href='#TruncatedMVN.sample'>#</a>&nbsp;<b><u>TruncatedMVN.sample</u></b> &mdash; <i>Function</i>.




```julia
sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)
```


Sample `n` samples from the distribution `d`.


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/beb130f96afc3dbb70432305a3474bff3a72f091/src/TruncatedMVN.jl#L68-L72)

</div>
<br>
