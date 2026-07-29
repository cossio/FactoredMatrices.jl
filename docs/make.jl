import Documenter
import FactoredMatrices

ENV["JULIA_DEBUG"] = "Documenter"

Documenter.makedocs(
    modules = [FactoredMatrices],
    sitename = "FactoredMatrices.jl",
    repo = Documenter.Remotes.GitHub("cossio", "FactoredMatrices.jl"),
    pages = [
        "Home" => "index.md",
        "Reference" => "reference.md",
    ]
)

# Deploy docs to Github pages.
Documenter.deploydocs(
    repo = "github.com/cossio/FactoredMatrices.jl.git",
    devbranch = "main"
)
