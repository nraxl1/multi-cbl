module IRSpectraML

import PythonCall, CondaPkg
using PythonCall, CondaPkg, Lux, Enzyme, Reactant
using Random, Printf, JLD2, Statistics
using Optimisers, MLUtils, NNlib, DispatchDoctor

include_files = ["Featurization.jl", "LoadData.jl", "Model.jl", "Training.jl"]
for filename in include_files
    include(dirname(@__FILE__) * "/" * filename) # asterisk here does string concat idk ask the language developers
end

export Optimisers, Lux, N_FG, FG_NAMES

end # module IRSpectraML
