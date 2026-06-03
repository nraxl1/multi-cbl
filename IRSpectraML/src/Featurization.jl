# Lazy rdkit import — only loaded when featurize() is called.
# This way the plastic classification pipeline can use this module
# without requiring rdkit to be installed.

const _HAVE_RDKIT = Ref(false)
const Chem = Ref{Any}(nothing)

function _ensure_rdkit()
    if _HAVE_RDKIT[]
        return true
    end
    try
        Chem[] = pyimport("rdkit.Chem")
        _HAVE_RDKIT[] = true
        return true
    catch e
        @warn "Could not import rdkit: $e. featurize() will be unavailable."
        return false
    end
end

const FG_SMARTS = Dict(
  "Chlorine" => "[Cl]",
  "Ester" => "[CX3](=O)[OX2]",
  "Aromatic Ring" => "c1ccccc1",
  "Methyl" => "[CH3]",
  "Ethylene" => "[CH2]"
)
const FG_NAMES = sort(collect(keys(FG_SMARTS)))
const N_FG     = length(FG_NAMES)

const FG_PATTERNS = Dict{String,Any}()

function _init_fg_patterns()
    for (k, v) in FG_SMARTS
        FG_PATTERNS[k] = Chem[].MolFromSmarts(v)
    end
end

"
Returns the functional group vector for the given smiles string.
Requires rdkit to be available; returns nothing with a warning otherwise.
"
function featurize(smiles::AbstractString)::Union{Nothing, Vector{Float32}}
    _ensure_rdkit() || return nothing
    if isempty(FG_PATTERNS)
        _init_fg_patterns()
    end

    mol = Chem[].MolFromSmiles(smiles)
    pyisinstance(mol, Chem[].rdchem.Mol) || return nothing

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
