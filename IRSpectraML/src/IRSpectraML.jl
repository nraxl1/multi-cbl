module IRSpectraML

import PythonCall, CondaPkg
using PythonCall, CondaPkg, Lux, Enzyme, Reactant, Random, Printf, JLD2, Statistics, Optimisers, MLUtils, NNlib, DispatchDoctor

include_files = ["Featurization.jl", "LoadData.jl", "ModelNext.jl", "TrainingNext.jl"]
for filename in include_files
    include(dirname(@__FILE__) * "/" * filename) # asterisk here does string concat idk ask the language developers
end

greet() = print("Hello World!")

export Optimisers, Lux, N_FG, FG_NAMES


end # module IRSpectraML
