using Pkg

Pkg.activate(".")

using DuckDB, DBInterface, DataFrames
using Lux, Enzyme, Optimisers, NNlib
using Statistics, Random, Printf
using MLDataDevices
using Functors: fmap
using PythonCall, JLD2, CUDA, GPUArrays, Revise, ProgressMeter
using MLUtils: DataLoader
import .GC

include("src/Featurization.jl")
include("src/ModelNext.jl")
include("src/LoadData.jl")
include("src/TrainingNext.jl")

const CHUNKS = [
    "src/parquet-files/data/IR_data_chunk00$(i)_of_009.parquet" for i in 1:9
]

const CACHE_DIR = "chunk_cache"
const MODEL_PATH = "model.jld2"
const ARCH_VERSION = "rescnn-v2"

CUDA.functional() && CUDA.allowscalar(false)

# --- helpers ---

function count_params(x)
    if x isa AbstractArray
        return length(x)
    elseif x isa NamedTuple
        return sum(count_params, values(x))
    elseif x isa Tuple
        return sum(count_params, x)
    else
        return 0
    end
end

# --- main ---

function main()
    Random.seed!(42)

    dev = MLDataDevices.gpu_device()
    @info "Using device: $dev"

    println("=== Bootstrapping from chunk 1 ===")
    X1, Y1, s1 = cached_load_chunk(CHUNKS[1])

    spec_len = size(X1, 1)
    println("\nSpectrum length: $spec_len  |  Labels: $N_FG")
    println("Label order: ", FG_NAMES)

    tr1 = findall(s -> s < 8, s1)
    val1 = findall(s -> s == 8, s1)
    tst1 = findall(s -> s == 9, s1)

    norm = fit_normalizer(X1[:, tr1])

    Xv = apply_normalizer(norm, X1[:, val1])
    Yv = Y1[:, val1]
    Xt = apply_normalizer(norm, X1[:, tst1])
    Yt = Y1[:, tst1]

    println("Val: $(size(Xv,2))  Test: $(size(Xt,2))  (from chunk 1)")
    println("Training chunks: $(length(CHUNKS))  (~$(length(CHUNKS)*length(tr1)) train samples total)")

    X1 = Y1 = nothing
    GC.gc()

    # ---- model (Lux.jl) ----
    model = build_model(spec_len, N_FG)          # architecture (stateless)
    rng = Xoshiro(42)
    ps, st = Lux.setup(rng, model)                # init params + state
    model = model |> dev
    ps = fmap(x -> dev(x), ps)                     # params to GPU
    st = fmap(x -> x isa AbstractArray ? dev(x) : x, st)

    if isfile(MODEL_PATH)
        saved_arch = JLD2.load(MODEL_PATH, "arch_version")
        if saved_arch != ARCH_VERSION
            println("\nCheckpoint arch '$saved_arch' ≠ current '$ARCH_VERSION' — retraining.")
            rm(MODEL_PATH)
        end
    end

    if isfile(MODEL_PATH)
        println("\nLoading saved model from $MODEL_PATH ...")
        ps = fmap(x -> dev(x), JLD2.load(MODEL_PATH, "params"))
        st_cpu = JLD2.load(MODEL_PATH, "states")
        st = fmap(x -> x isa AbstractArray ? dev(x) : x, st_cpu)
        println("  Loaded. Skipping training — delete $MODEL_PATH to retrain.")
    else
        n_params = count_params(ps)
        println("\nModel parameters: $n_params")

        ps, st = train_model!(model, ps, st, CHUNKS, norm, Xv, Yv;
                              epochs=50, lr_start=1.0f-3, lr_min=1.0f-6, patience=5)

        println("\nSaving model → $MODEL_PATH")
        cpu_dev = cpu_device()
        JLD2.save(MODEL_PATH,
            "params",       fmap(x -> cpu_dev(x), ps),
            "states",       fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, st),
            "arch_version", ARCH_VERSION,
            "fg_names",     FG_NAMES,
            "norm_mu",      norm.μ,
            "norm_sigma",   norm.σ,
            "spec_len",     spec_len,
        )
        println("  Saved.")
    end

    # ---- test evaluation (batched to avoid VRAM OOM) ----
    st = Lux.test_mode(st)
    test_loader = DataLoader((Xt, Yt), batchsize=256)

    all_pred = Vector{Matrix{Float32}}()
    all_true = Vector{Matrix{Float32}}()

    for (Xb, Yb) in test_loader
        Xb_d = dev(Xb)
        y_pred, _ = model(Xb_d, ps, st)
        pred_b = cpu_device()(sigmoid.(y_pred))
        push!(all_pred, pred_b)
        push!(all_true, Yb)
    end

    pred_cpu = hcat(all_pred...)
    Yt_cpu  = hcat(all_true...)

    pred_bin = pred_cpu .> 0.5f0
    overall_acc = mean(pred_bin .== Yt_cpu)

    # per-label F1
    tp = vec(sum(pred_bin .& (Yt_cpu .== 1f0), dims=2))
    fp = vec(sum(pred_bin .& (Yt_cpu .== 0f0), dims=2))
    fn = vec(sum((pred_bin .== 0f0) .& (Yt_cpu .== 1f0), dims=2))

    precision = tp ./ (tp .+ fp .+ eps(Float32))
    recall    = tp ./ (tp .+ fn .+ eps(Float32))
    f1        = 2f0 .* precision .* recall ./ (precision .+ recall .+ eps(Float32))
    macro_f1  = mean(f1)

    println("\n=== TEST RESULTS ===")
    println("Overall accuracy: $(round(100*overall_acc, digits=2))%")
    println("Macro F1:         $(round(100*macro_f1, digits=2))%")

    for (i, name) in enumerate(FG_NAMES)
        acc_i = mean(pred_bin[i, :] .== Yt_cpu[i, :])
        @printf("  %-20s accuracy: %6.2f%%   F1: %6.2f%%\n", name, 100 * acc_i, 100 * f1[i])
    end
end

main()
