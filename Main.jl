using Pkg
Pkg.activate("IRSpectraML")  # activate the package env that has Lux + Reactant

using Lux, Reactant, Optimisers, Enzyme, Random
using OneHotArrays
using Statistics, Printf, NNlib
using MLDataDevices
using Functors: fmap
using JLD2, ProgressMeter, AbbreviatedStackTraces, DispatchDoctor
using ADTypes
using MLUtils: DataLoader
import Base.GC
using IRSpectraML

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
const REPO_ROOT       = @__DIR__
const TRAIN_ROOT      = joinpath(REPO_ROOT, "online-data", "train")
const TEST_ROOT       = joinpath(REPO_ROOT, "online-data", "test")
const LAB_ROOT        = joinpath(REPO_ROOT, "lab-data")
const PLASTIC_MODEL   = joinpath(REPO_ROOT, "plastic_model.jld2")
const NORM_PATH       = joinpath(REPO_ROOT, "plastic_norm.jld2")

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

"Per-feature z-score normalizer. Fits on training data only."
struct PlasticNormalizer
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function fit_plastic_norm(X::Matrix{Float32})
    μ = vec(mean(X, dims=2))
    σ = vec(std(X,  dims=2)) .+ 1f-6
    return PlasticNormalizer(μ, σ)
end

apply_norm(n::PlasticNormalizer, X::Matrix{Float32}) = (X .- n.μ) ./ n.σ

"Build a confusion matrix (n_classes × n_classes). Rows = true, cols = pred."
function confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int}, n_classes::Int)
    cm = zeros(Int, n_classes, n_classes)
    for (t, p) in zip(y_true, y_pred)
        cm[t+1, p+1] += 1
    end
    return cm
end

function per_class_metrics(cm::Matrix{Int})
    n_classes = size(cm, 1)
    acc    = zeros(Float32, n_classes)
    prec   = zeros(Float32, n_classes)
    rec    = zeros(Float32, n_classes)
    f1     = zeros(Float32, n_classes)
    for c in 1:n_classes
        tp = cm[c, c]
        fn = sum(cm[c, :]) - tp
        fp = sum(cm[:, c]) - tp
        support = sum(cm[c, :])
        acc[c]  = support > 0 ? Float32(tp / support) : 0f0
        prec[c] = (tp + fp) > 0 ? Float32(tp / (tp + fp)) : 0f0
        rec[c]  = (tp + fn) > 0 ? Float32(tp / (tp + fn)) : 0f0
        f1[c]   = (prec[c] + rec[c]) > 0 ? Float32(2 * prec[c] * rec[c] / (prec[c] + rec[c])) : 0f0
    end
    return (acc=acc, prec=prec, rec=rec, f1=f1, macro_f1=mean(f1))
end

"Batched evaluation: returns (loss, accuracy, y_true, y_pred).
Note: y_true is converted to one-hot internally for the loss, but the
function returns the original integer labels in y_true for the confusion
matrix downstream."
function evaluate(model, ps, st, X::Matrix{Float32}, Y::Vector{Int}; batchsize=512, dev=cpu_device(), n_classes=IRSpectraML.N_PLASTIC)
    n = size(X, 2)
    loss_fn = CrossEntropyLoss(; logits=true)
    total_loss = 0f0
    n_batches = 0
    y_true = Int[]
    y_pred = Int[]
    state = Lux.testmode(st)
    for i in 1:batchsize:n
        idx = i:min(i+batchsize-1, n)
        Xb = X[:, idx] |> dev
        Yb_int = Y[idx]
        y_logits, _ = model(Xb, ps, state)
        # Build one-hot on the same device as the logits
        Yb_oh = onehotbatch(Yb_int, 0:(n_classes-1)) |> dev
        # Fall back to plain Array if dev is cpu_device (onehotbatch is on CPU)
        if dev === cpu_device()
            Yb_oh = onehotbatch(Yb_int, 0:(n_classes-1))
        end
        total_loss += loss_fn(y_logits, Yb_oh)
        append!(y_true, Yb_int)
        append!(y_pred, argmax.(eachcol(y_logits)))
        n_batches += 1
    end
    avg_loss = total_loss / max(n_batches, 1)
    acc = mean(y_true .== y_pred) |> Float32
    return (loss=avg_loss, acc=acc, y_true=y_true, y_pred=y_pred)
end

# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

function main()
    # Reactant on CPU for now. To switch to GPU on a CUDA box, change
    # "cpu" to "gpu" — the rest of the training code is identical.
    Reactant.set_default_backend("cpu")
    dev = reactant_device()
    @info "Using device: $dev (Reactant backend: $(Reactant.XLA.default_backend()))"

    # --- 1) Load data ---
    println("=== Loading plastic data ===")
    data = IRSpectraML.cached_plastic_data(
        train_root = TRAIN_ROOT,
        test_root  = TEST_ROOT,
        lab_root   = LAB_ROOT,
        spec_len   = 3000,
        val_frac   = 0.15,
        seed       = 42,
    )
    spec_len = data.spec_len
    println("\nSpectrum length: $spec_len")
    println("Train: $(size(data.X_tr, 2))  Val: $(size(data.X_val, 2))  Test: $(length(data.Y_test))  Lab: $(length(data.Y_lab))")
    println("Classes: $(IRSpectraML.PLASTIC_TYPES)")

    # Class distribution (train)
    for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
        n_i = count(==(i-1), data.Y_tr)
        println("  $t: $n_i ($(round(100*n_i/length(data.Y_tr), digits=1))%)")
    end

    # --- 2) Fit normalizer on TRAIN only ---
    norm = fit_plastic_norm(data.X_tr)

    X_tr_n  = apply_norm(norm, data.X_tr)
    X_val_n = apply_norm(norm, data.X_val)
    X_test_n = size(data.X_test, 2) > 0 ? apply_norm(norm, data.X_test) : data.X_test
    X_lab_n  = size(data.X_lab, 2)  > 0 ? apply_norm(norm, data.X_lab)  : data.X_lab

    # --- 3) Build model ---
    model = IRSpectraML.build_plastic_model(spec_len; n_classes=IRSpectraML.N_PLASTIC, n_spatial=24)
    parameters, state = Lux.setup(Random.default_rng(), model) |> dev
    n_params = IRSpectraML.count_params(parameters)
    println("\nModel parameters: $n_params  (output: $(IRSpectraML.N_PLASTIC) classes)")

    # --- 4) Skip training if checkpoint exists ---
    if isfile(PLASTIC_MODEL)
        saved_arch = JLD2.load(PLASTIC_MODEL, "arch_version")
        if saved_arch != IRSpectraML.ARCH_VERSION
            println("\nCheckpoint arch '$saved_arch' ≠ current '$(IRSpectraML.ARCH_VERSION)' — retraining.")
            rm(PLASTIC_MODEL)
        end
    end

    best_val_acc = -1f0
    if isfile(PLASTIC_MODEL)
        println("\nLoading saved model from $PLASTIC_MODEL ...")
        parameters = fmap(x -> x |> dev, JLD2.load(PLASTIC_MODEL, "params"))
        st_cpu     = JLD2.load(PLASTIC_MODEL, "states")
        state      = fmap(x -> x isa AbstractArray ? x |> dev : x, st_cpu)
        best_val_acc = Float32(JLD2.load(PLASTIC_MODEL, "best_val_acc"))
        println("  Loaded. best_val_acc=$(round(100*best_val_acc, digits=2))%. Delete $PLASTIC_MODEL to retrain.")
    else
        parameters, state, best_val_acc = IRSpectraML.train_plastic_model!(
            model, parameters, state,
            X_tr_n, data.Y_tr,
            X_val_n, data.Y_val;
            X_te      = isempty(data.Y_test) ? nothing : X_test_n,
            Y_te      = isempty(data.Y_test) ? nothing : data.Y_test,
            epochs    = 40,
            lr_start  = 1f-3,
            lr_min    = 1f-6,
            batchsize = 64,
            patience  = 8,
            resume    = true,
            label_names = IRSpectraML.PLASTIC_TYPES,
            dev       = dev,
        )

        println("\nSaving model → $PLASTIC_MODEL")
        cpu_dev = cpu_device()
        JLD2.save(PLASTIC_MODEL,
            "params",        fmap(cpu_dev, parameters),
            "states",        fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, state),
            "arch_version",  IRSpectraML.ARCH_VERSION,
            "best_val_acc",  best_val_acc,
            "label_names",   IRSpectraML.PLASTIC_TYPES,
            "spec_len",      spec_len,
        )
        JLD2.save(NORM_PATH, "mu", norm.μ, "sigma", norm.σ)
        println("  Saved.")
    end

    # --- 5) Final evaluation ---
    state = Lux.testmode(state)

    println("\n=== TRAIN (sanity check) ===")
    tr_eval = evaluate(model, parameters, state, X_tr_n, data.Y_tr; dev=dev)
    println("  acc=$(round(100*tr_eval.acc, digits=2))%  loss=$(round(tr_eval.loss, digits=4))")

    println("\n=== VAL ===")
    val_eval = evaluate(model, parameters, state, X_val_n, data.Y_val; dev=dev)
    println("  acc=$(round(100*val_eval.acc, digits=2))%  loss=$(round(val_eval.loss, digits=4))")
    cm_val = confusion_matrix(val_eval.y_true, val_eval.y_pred, IRSpectraML.N_PLASTIC)
    metrics_val = per_class_metrics(cm_val)
    println("  Macro F1: $(round(100*metrics_val.macro_f1, digits=2))%")
    for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
        @printf("    %-6s acc=%5.2f%%  prec=%5.2f%%  rec=%5.2f%%  F1=%5.2f%%\n",
                t, 100*metrics_val.acc[i], 100*metrics_val.prec[i],
                100*metrics_val.rec[i], 100*metrics_val.f1[i])
    end

    if size(X_test_n, 2) > 0
        println("\n=== TEST (held-out, same domain) ===")
        te_eval = evaluate(model, parameters, state, X_test_n, data.Y_test; dev=dev)
        println("  acc=$(round(100*te_eval.acc, digits=2))%  loss=$(round(te_eval.loss, digits=4))")
        cm_te = confusion_matrix(te_eval.y_true, te_eval.y_pred, IRSpectraML.N_PLASTIC)
        m_te = per_class_metrics(cm_te)
        println("  Macro F1: $(round(100*m_te.macro_f1, digits=2))%")
        for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
            @printf("    %-6s acc=%5.2f%%  prec=%5.2f%%  rec=%5.2f%%  F1=%5.2f%%\n",
                    t, 100*m_te.acc[i], 100*m_te.prec[i],
                    100*m_te.rec[i], 100*m_te.f1[i])
        end
    end

    if size(X_lab_n, 2) > 0
        println("\n=== LAB DATA (cross-domain) ===")
        lab_eval = evaluate(model, parameters, state, X_lab_n, data.Y_lab; dev=dev)
        println("  acc=$(round(100*lab_eval.acc, digits=2))%  loss=$(round(lab_eval.loss, digits=4))")
        cm_lab = confusion_matrix(lab_eval.y_true, lab_eval.y_pred, IRSpectraML.N_PLASTIC)
        m_lab = per_class_metrics(cm_lab)
        println("  Macro F1: $(round(100*m_lab.macro_f1, digits=2))%")
        for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
            @printf("    %-6s acc=%5.2f%%  prec=%5.2f%%  rec=%5.2f%%  F1=%5.2f%%\n",
                    t, 100*m_lab.acc[i], 100*m_lab.prec[i],
                    100*m_lab.rec[i], 100*m_lab.f1[i])
        end
    end
end

main()
