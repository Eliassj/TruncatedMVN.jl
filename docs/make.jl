using TruncatedMVN
using Documenter

DocMeta.setdocmeta!(TruncatedMVN, :DocTestSetup, :(using TruncatedMVN); recursive=true)

makedocs(;
    modules=[TruncatedMVN],
    authors="Eliassj <elias.sjolin@gmail.com> and contributors",
    sitename="TruncatedMVN.jl",
    format=Documenter.HTML(;
        canonical="https://Eliassj.github.io/TruncatedMVN.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Eliassj/TruncatedMVN.jl",
    devbranch="master",
)
