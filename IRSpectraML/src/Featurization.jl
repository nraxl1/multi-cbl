const Chem = pyimport("rdkit.Chem")

const FG_SMARTS = Dict(
  "Chlorine" => "[Cl]",
  "Ester" => "[CX3](=O)[OX2]",
  "Aromatic Ring" => "c1ccccc1",
  "Methyl" => "[CH3]",
  "Ethylene" => "[CH2]"
)
const FG_NAMES = sort(collect(keys(FG_SMARTS)))
const N_FG     = length(FG_NAMES)

const FG_PATTERNS = Dict(
    k => Chem.MolFromSmarts(v) for (k, v) in FG_SMARTS
)

"
Returns the functional group vector for the given smiles string
"
function featurize(smiles::AbstractString)::Union{Nothing, Vector{Float32}}
    mol = Chem.MolFromSmiles(smiles)
    pyisinstance(mol, Chem.rdchem.Mol) || return nothing

    feats = Vector{Float32}(undef, N_FG)
    for (i, k) in enumerate(FG_NAMES)
        n = length(mol.GetSubstructMatches(FG_PATTERNS[k]))
        feats[i] = n > 0 ? 1f0 : 0f0
    end
    return feats
end


"
Struct used to store parameters for normalization
"
struct Normalizer
    μ::Vector{Float32}
    σ::Vector{Float32}
end

"
Generate a ```Normalizer``` for the given data
"
function fit_normalizer(X::Matrix{Float32})
    μ = vec(mean(X, dims=2))
    σ = vec(std(X,  dims=2)) .+ 1f-6
    return Normalizer(μ, σ)
end

"
Normalize the given data with the parameters from ```norm```
"
function apply_normalizer(norm::Normalizer, X::Matrix{Float32})
    return (X .- norm.μ) ./ norm.σ
end
