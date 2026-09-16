using Documenter
using NeoDistribution

DocMeta.setdocmeta!(NeoDistribution, :DocTestSetup, :(using NeoDistribution); recursive=true)

makedocs(;
    modules=[NeoDistribution],
    authors="Istanti Prima Nurani <istantiprima@gmail.com>",
    sitename="NeoDistribution.jl",
    format=Documenter.HTML(;
        canonical="https://istantiprima.github.io/NeoDistribution.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Distributions" => "distributions.md",
    ],
)

deploydocs(;
    repo="github.com/istantiprima/NeoDistribution.jl",
    devbranch="main",
)
