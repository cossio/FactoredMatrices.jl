using LowRankMatrices
using Documenter

DocMeta.setdocmeta!(LowRankMatrices, :DocTestSetup, :(using LowRankMatrices); recursive=true)

makedocs(;
    modules=[LowRankMatrices],
    authors="JFdCD <j.cossio.diaz@gmail.com>",
    repo="https://github.com/cossio/LowRankMatrices.jl/blob/{commit}{path}#{line}",
    sitename="LowRankMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
