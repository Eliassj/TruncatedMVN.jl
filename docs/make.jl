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
        repo="github.com/Eliassj/TruncatedMVN.jl",
        devbranch="master",
        devurl="dev"
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => "public.md",
        "Internals" => "private.md"
    ],)

deploydocs(;
    repo="github.com/Eliassj/TruncatedMVN.jl",
    target="build",
    devbranch="master",
    branch="gh-pages",
    push_preview=false
)
