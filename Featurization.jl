
############################################################
# RDKit
############################################################

const Chem = pyimport("rdkit.Chem")

############################################################
# FUNCTIONAL GROUPS
############################################################

const FG_SMARTS = Dict(
    "ester"    => "[CX3](=O)[OX2H0][#6]",
    "aromatic" => "c1ccccc1",
    "alkane"   => "[CX4]",
    "alkene"   => "C=C",
    "halogen"  => "[F,Cl,Br,I]",
    "alcohol"  => "[OX2H]",
    "ether"    => "[OD2]([#6])[#6]",
)

const FG_NAMES = sort(collect(keys(FG_SMARTS)))
const N_FG     = length(FG_NAMES)

const FG_PATTERNS = Dict(
    k => Chem.MolFromSmarts(v) for (k, v) in FG_SMARTS
)

############################################################
# FEATURIZE
############################################################

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

############################################################
# NORMALIZE
############################################################

struct Normalizer
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function fit_normalizer(X::Matrix{Float32})
    μ = vec(mean(X, dims=2))
    σ = vec(std(X,  dims=2)) .+ 1f-6
    return Normalizer(μ, σ)
end

function apply_normalizer(norm::Normalizer, X::Matrix{Float32})
    return (X .- norm.μ) ./ norm.σ
end
