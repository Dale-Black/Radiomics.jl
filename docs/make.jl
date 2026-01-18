using Documenter
using Radiomics

makedocs(
    sitename = "Radiomics.jl",
    authors = "Dale Black and contributors",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://daleblack.github.io/Radiomics.jl",
        assets = String[],
    ),
    modules = [Radiomics],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Feature Classes" => "features.md",
        "Configuration" => "configuration.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/daleblack/Radiomics.jl.git",
    devbranch = "main",
    push_preview = true,
)
