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
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/Eliassj/TruncatedMVN.jl",
        #md_output_path=".", # For live preview
        #build_vitepress=false, # For live preview
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => "public.md",
        "Internals" => "private.md"
    ],
    #clean=false,# For live preview
)

DocumenterVitepress.deploydocs(;
    repo="github.com/Eliassj/TruncatedMVN.jl",
    target="build",
    devbranch="master",
    push_preview=false
)
