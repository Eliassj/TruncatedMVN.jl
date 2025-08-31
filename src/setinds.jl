function gen_lininds(a::Type{<:SArray{L,T,D}}) where {L,T,D}
    ds = get_dims(a)
    LinearIndices(
        ntuple(i -> SOneTo(ds[i]), D)
    )
end

get_dims(::Type{<:SArray{D}}) where {D} = D.parameters

@propagate_inbounds function newsetindex(a::StaticArray, x, indices...)
    _newsetindex(Length(a), a, x, LinearIndices(a)[indices...])
end


function newnewsetindex(a::StaticArray{<:Tuple,T,D}, x, indices...) where {T,D,L}
    _newnewsetindex(Length(a), a, x, indices...)
end
function _newnewsetindex(::Length{L}, a::StaticArray{<:Tuple,T,D}, x, indices...) where {T,D,L}
    inds = LinearIndices(x)[indices...]
    @show x inds L
    typeof(a)(
        ntuple(L) do i
            ifelse(i in inds, x[inds[i]], a[i])
        end
    )
end

@generated function _newsetindex(::Length{L}, a::StaticArray{<:Tuple,T}, x, index) where {L,T}
    exprs = [:(ifelse($i in index, x[index[$i]], a[$i])) for i = 1:L]
    Core.println(exprs)
    return quote
        @_propagate_inbounds_meta
        @boundscheck if any(index .< 1 .|| index .> $(L))
            throw(BoundsError(a, index))
        end
        Core.println("index ", index)
        Core.println("x ", x)
        return typeof(a)(tuple($(exprs...)))
    end
end
# @propagate_inbounds function newsetindex(a::StaticArray, x, inds...)
#     @show inds
#     @show LinearIndices(a)[inds...]
#     _newsetindex(Length(a), a, x, LinearIndices(a)[inds...])
# end
