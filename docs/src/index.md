```@raw html
---
layout: home

hero:
  name: "TruncatedMVN.jl"
  tagline: Truncated multivariate normal distribution with exact sampling


---

```

```@meta
CurrentModule = TruncatedMVN
```

# TruncatedMVN

Julia reimplementation of a truncated multivariate normal distribution with perfect sampling.

Distribution and sampling method by [Botev_2017](@cite).

Code based on MATLAB implementation by Zdravko Botev and Python implementation by Paul Brunzema.
These may be found here: [MATLAB by Botev](https://mathworks.com/matlabcentral/fileexchange/53792-truncated-multivariate-normal-generator), [Python by Brunzema](https://github.com/brunzema/truncated-mvn-sampler)

Exports 1 struct + its constructor and 1 function:

- [`TruncatedMVN.TruncatedMVNormal`](@ref): Struct and constructor for initializing the distribution.
- [`TruncatedMVN.sample`](@ref): Exact sampling from the distribution.

Documentation for these can be found in the [Public API](@ref) section.

# References

```@bibliography

```
