module IRSpectraML

import PythonCall, CondaPkg
using PythonCall, CondaPkg, Lux, Enzyme, Reactant
using Random, Printf, JLD2, Statistics
using Optimisers, MLUtils, NNlib, DispatchDoctor
using DataInterpolations: LinearInterpolation, ExtrapolationType
using Lux: CrossEntropyLoss

include_files = ["Featurization.jl", "LoadData.jl", "Model.jl", "Training.jl"]
for filename in include_files
    include(dirname(@__FILE__) * "/" * filename) # asterisk here does string concat idk ask the language developers
end

# Legacy FG-pipeline exports (Phase 1 contrastive pretraining)
export N_FG, FG_NAMES

# Plastic classification pipeline exports (Phase 0)
export PLASTIC_TYPES, N_PLASTIC, PLASTIC_TYPE_TO_IDX,
       DEFAULT_SPEC_LEN, DEFAULT_WN_MIN, DEFAULT_WN_MAX,
       load_plastic_csv, load_plastic_csv_duckdb, interpolate_spectrum, plastic_type_from_path,
       discover_plastic_csvs, load_plastic_dataset, load_plastic_dataset_duckdb,
       split_train_val_test, prepare_plastic_data, cached_plastic_data,
       build_plastic_model, build_model_fg,
       train_plastic_model!, classification_accuracy,
       PLASTIC_CHECKPOINT_PATH, ARCH_VERSION

end # module IRSpectraML
