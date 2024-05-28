---
layout: home

hero:
  name: "TruncatedMVN.jl"
  tagline: Truncated multivariate normal distribution with exact sampling


---





# TruncatedMVN {#TruncatedMVN}

Julia reimplementation of a truncated multivariate normal distribution with perfect sampling.

Distribution and sampling method by [[1](/index#Botev_2017)].

Code based on MATLAB implementation by Zdravko Botev and Python implementation by Paul Brunzema. These may be found here: [MATLAB by Botev](https://mathworks.com/matlabcentral/fileexchange/53792-truncated-multivariate-normal-generator), [Python by Brunzema](https://github.com/brunzema/truncated-mvn-sampler)

Exports 1 struct + its constructor and 1 function:
- [`TruncatedMVN.TruncatedMVNormal`](/index#TruncatedMVN.TruncatedMVNormal): Struct and constructor for initializing the distribution.
  
- [`TruncatedMVN.sample`](/index#TruncatedMVN.sample): Exact sampling from the distribution.
  
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.TruncatedMVNormal' href='#TruncatedMVN.TruncatedMVNormal'>#</a>&nbsp;<b><u>TruncatedMVN.TruncatedMVNormal</u></b> &mdash; <i>Type</i>.




```julia
TruncatedMVNormal{S<:AbstractArray{<:AbstractFloat},T<:AbstractVector{<:AbstractFloat},U<:Integer,V<:AbstractFloat,P<:AbstractVector{<:Integer}}
```


Truncated multivariate normal distribution with minimax tilting-based sampling.


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/11ad953766248812342dcbd9adff57d379a24493/src/TruncatedMVN.jl#L19-L24)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}' href='#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}'>#</a>&nbsp;<b><u>TruncatedMVN.TruncatedMVNormal</u></b> &mdash; <i>Method</i>.




```julia
 TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
```


Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](/index#TruncatedMVN.TruncatedMVNormal) distribution.

Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](/index#TruncatedMVN.sample).

**Arguments**
- `mu::T`: D-dimensional vector of means.
  
- `cov::S`: DxD-dimensional covariance matrix.
  
- `lb::T`: D-dimensional vector of lower bounds.
  
- `ub::T`: D-dimensional vector of upper bounds.
  


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/11ad953766248812342dcbd9adff57d379a24493/src/TruncatedMVN.jl#L41-L54)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}-2' href='#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T<:(AbstractVector{<:AbstractFloat}), S<:(AbstractArray{<:AbstractFloat})}-2'>#</a>&nbsp;<b><u>TruncatedMVN.TruncatedMVNormal</u></b> &mdash; <i>Method</i>.




```julia
 TruncatedMVNormal(mu::T, cov::S, lb::T, ub::T) where {T<:AbstractVector{<:AbstractFloat},S<:AbstractArray{<:AbstractFloat}}
```


Inner constructor of the [`TruncatedMVN.TruncatedMVNormal`](/index#TruncatedMVN.TruncatedMVNormal) distribution.

Generates a truncated multivariate normal distribution which may be accurately sampled from using [`TruncatedMVN.sample`](/index#TruncatedMVN.sample).

**Arguments**
- `mu::T`: D-dimensional vector of means.
  
- `cov::S`: DxD-dimensional covariance matrix.
  
- `lb::T`: D-dimensional vector of lower bounds.
  
- `ub::T`: D-dimensional vector of upper bounds.
  

Bounds may be `-Inf`/`Inf`.


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/11ad953766248812342dcbd9adff57d379a24493/src/TruncatedMVN.jl#L41-L57)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='TruncatedMVN.sample' href='#TruncatedMVN.sample'>#</a>&nbsp;<b><u>TruncatedMVN.sample</u></b> &mdash; <i>Function</i>.




```julia
sample(d::TruncatedMVNormal, n::Integer, max_iter::Integer=10000)
```


Sample `n` samples from the distribution `d`.

Returns an n x D `Matrix` of samples where D is the dimension of the distribution `d`.


[source](https://github.com/Eliassj/TruncatedMVN.jl/blob/11ad953766248812342dcbd9adff57d379a24493/src/TruncatedMVN.jl#L76-L82)

</div>
<br>

# References {#References}

***
# Bibliography

1. Z. I. Botev. [_The normal law under linear restrictions: Simulation and estimation via minimax tilting_](http://dx.doi.org/10.1111/rssb.12162). [Journal of the Royal Statistical Society. Series B, Statistical methodology **79**, 125–148](https://doi.org/10.1111/rssb.12162) (2017).
  
