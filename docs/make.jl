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
    authors="Elias Sjölin <elias.sjolin@gmail.com> and contributors",
    sitename="TruncatedMVN.jl",
    format=MarkdownVitepress(
        deploy_url="https://Eliassj.github.io/TruncatedMVN.jl",
        repo="github.com/Eliassj/TruncatedMVN.jl",
    ),
    pages=[
        "Home" => "index.md",
    ],)

deploydocs(;
    repo="github.com/Eliassj/TruncatedMVN.jl",
    devbranch="master",
    branch="gh-pages",
    push_preview=false
)
