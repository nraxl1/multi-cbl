
using DuckDB, DBInterface, DataFrames
using Flux, Statistics, Random, Printf
using CUDA, PythonCall, JLD2, cuDNN
using MLUtils: DataLoader
import .GC

include("Initialize.jl")
include("Featurization.jl")
include("Model.jl")
include("LoadData.jl")
include("Training.jl")


CUDA.functional() && CUDA.allowscalar(false)

println(CUDA.functional() ? "🚀 GPU detected" : "⚠️  CPU mode")


const CHUNKS = [
    "parquet-files/data/IR_data_chunk00$(i)_of_009.parquet" for i in 1:9
]

const CACHE_DIR  = "chunk_cache"
const MODEL_PATH = "model.jld2"
const ARCH_VERSION = "rescnn-v1"


function main()
    Random.seed!(42)

    println("=== Bootstrapping from chunk 1 ===")
    X1, Y1, s1 = cached_load_chunk(CHUNKS[1])

    spec_len = size(X1, 1)
    println("\nSpectrum length: $spec_len  |  Labels: $N_FG")
    println("Label order: ", FG_NAMES)

    tr1  = findall(s -> s < 8,  s1)
    val1 = findall(s -> s == 8, s1)
    tst1 = findall(s -> s == 9, s1)

    norm = fit_normalizer(X1[:, tr1])

    Xv = apply_normalizer(norm, X1[:, val1]);  Yv = Y1[:, val1]
    Xt = apply_normalizer(norm, X1[:, tst1]);  Yt = Y1[:, tst1]

    println("Val: $(size(Xv,2))  Test: $(size(Xt,2))  (from chunk 1)")
    println("Training chunks: $(length(CHUNKS))  (~$(length(CHUNKS)*length(tr1)) train samples total)")

    X1 = Y1 = nothing;  GC.gc()

    # ---- model (load checkpoint or build fresh) ----
    model = build_model(spec_len, N_FG)
    if isfile(MODEL_PATH)
        saved_arch = JLD2.load(MODEL_PATH, "arch_version")
        if saved_arch != ARCH_VERSION
            println("\nCheckpoint arch '$saved_arch' ≠ current '$ARCH_VERSION' — retraining.")
            rm(MODEL_PATH)
        end
    end
    if isfile(MODEL_PATH)
        println("\nLoading saved model from $MODEL_PATH ...")
        cpu_state = JLD2.load(MODEL_PATH, "model_state")
        Flux.loadmodel!(model, cpu_state)
        println("  Loaded. Skipping training — delete $MODEL_PATH to retrain.")
    else
        n_params = sum(length, Flux.trainable(model))
        println("\nModel parameters: $n_params")

        train_model!(model, CHUNKS, norm, Xv, Yv; epochs=10)

        println("\nSaving model → $MODEL_PATH")
        JLD2.save(MODEL_PATH,
            "model_state",  Flux.state(cpu(model)),
            "arch_version", ARCH_VERSION,
            "fg_names",     FG_NAMES,
            "norm_mu",      norm.μ,
            "norm_sigma",   norm.σ,
            "spec_len",     spec_len,
        )
        println("  Saved.")
    end

    # ---- test evaluation (batched to avoid VRAM OOM) ----
    Flux.testmode!(model)
    test_loader = DataLoader((Xt, Yt), batchsize=128)

    all_pred = Vector{Matrix{Float32}}()
    all_true = Vector{Matrix{Float32}}()

    for (Xb, Yb) in test_loader
        Xb_d = CUDA.functional() ? gpu(Xb) : Xb
        pred_b = cpu(sigmoid.(model(Xb_d)))
        push!(all_pred, pred_b)
        push!(all_true, Yb)
    end

    pred_cpu = hcat(all_pred...)
    Yt_cpu   = hcat(all_true...)

    overall_acc = mean((pred_cpu .> 0.5f0) .== Yt_cpu)
    println("\n=== TEST RESULTS ===")
    println("Overall accuracy: $(round(100*overall_acc, digits=2))%")

    for (i, name) in enumerate(FG_NAMES)
        acc_i = mean((pred_cpu[i,:] .> 0.5f0) .== Yt_cpu[i,:])
        @printf("  %-12s accuracy: %.1f%%\n", name, 100*acc_i)
    end
end

main()
