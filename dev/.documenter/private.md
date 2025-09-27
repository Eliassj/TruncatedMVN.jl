
# Internals {#Internals}

Documentation for internal functions.
<details class='jldocstring custom-block' open>
<summary><a id='TruncatedMVN.lnNormalProb-Union{Tuple{T}, Tuple{T, T}} where T' href='#TruncatedMVN.lnNormalProb-Union{Tuple{T}, Tuple{T, T}} where T'><span class="jlbinding">TruncatedMVN.lnNormalProb</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
lnNormalProb(a, b)
```


Accurately compute `ln(P(a<Z<b))` `where Z~N(0,1)`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/6529738ccbf48893874d6f02c72d4d301e7bb0aa/src/TruncatedMVN.jl#L425-L429" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='TruncatedMVN.mvnrnd!-Tuple{AbstractArray, AbstractArray, TruncatedMVNormal, Vararg{AbstractArray, 4}}' href='#TruncatedMVN.mvnrnd!-Tuple{AbstractArray, AbstractArray, TruncatedMVNormal, Vararg{AbstractArray, 4}}'><span class="jlbinding">TruncatedMVN.mvnrnd!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mvnrnd!(z::AbstractArray, logpr::AbstractArray, d::TruncatedMVNormal, mu::AbstractArray, L::AbstractArray, lb::AbstractArray, ub::AbstractArray)
```


Generates samples from a normal distribution.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/6529738ccbf48893874d6f02c72d4d301e7bb0aa/src/TruncatedMVN.jl#L163-L167" target="_blank" rel="noreferrer">source</a></Badge>

</details>

