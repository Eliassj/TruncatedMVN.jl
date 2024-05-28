using TruncatedMVN
using Documenter
using DocumenterCitations
using DocumenterVitepress

DocMeta.setdocmeta!(TruncatedMVN, :DocTestSetup, :(using TruncatedMVN); recursive=true)
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

makedocs(;
    plugins=[bib],
    modules=[TruncatedMVN],
    authors="Eliassj <elias.sjolin@gmail.com> and contributors",
    sitename="TruncatedMVN.jl",
    format=MarkdownVitepress(
        repo="https://Eliassj.github.io/TruncatedMVN.jl",
    ),
    pages=[
        "Home" => "index.md",
    ],)

deploydocs(;
    repo="github.com/Eliassj/TruncatedMVN.jl",
    devbranch="master",
)
