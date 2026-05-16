module IRSpectraML

include_files = ["Featurization.jl", "LoadData.jl", "ModelNext.jl", "TrainingNext.jl"]
for filename in include_files
    include(dirname(@__FILE__) * filename) # asterisk here does string concat idk ask the language developers
end

greet() = print("Hello World!")

end # module IRSpectraML
