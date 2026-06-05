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
const REPO_ROOT = @__DIR__
const TRAIN_ROOT = joinpath(REPO_ROOT, "online-data", "train")
const TEST_ROOT = joinpath(REPO_ROOT, "online-data", "test")
const LAB_ROOT = joinpath(REPO_ROOT, "lab-data")
const POSEIDON_ROOT = joinpath(REPO_ROOT, "Poseidon_files_V0.1.1", "Data", "IR_Spectra")
const POSEIDON_LABELS = joinpath(REPO_ROOT, "Poseidon_files_V0.1.1", "Data", "IR_References", "D4_4_publication.csv")
const PLASTIC_MODEL = joinpath(REPO_ROOT, "plastic_model.jld2")
const NORM_PATH = joinpath(REPO_ROOT, "plastic_norm.jld2")

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
    σ = vec(std(X, dims=2)) .+ 1f-6
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
    acc = zeros(Float32, n_classes)
    prec = zeros(Float32, n_classes)
    rec = zeros(Float32, n_classes)
    f1 = zeros(Float32, n_classes)
    for c in 1:n_classes
        tp = cm[c, c]
        fn = sum(cm[c, :]) - tp
        fp = sum(cm[:, c]) - tp
        support = sum(cm[c, :])
        acc[c] = support > 0 ? Float32(tp / support) : 0f0
        prec[c] = (tp + fp) > 0 ? Float32(tp / (tp + fp)) : 0f0
        rec[c] = (tp + fn) > 0 ? Float32(tp / (tp + fn)) : 0f0
        f1[c] = (prec[c] + rec[c]) > 0 ? Float32(2 * prec[c] * rec[c] / (prec[c] + rec[c])) : 0f0
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
    # Pull parameters and state to CPU for evaluation. The Reactant-device
    # model parameters can't be used in a non-JIT'd Lux call (no pointer()
    # support for ConcretePJRTArray). We don't pay the device transfer cost
    # during training (the @compile step handles it), only at eval time.
    ps_cpu = fmap(x -> x isa AbstractArray ? Array(x) : x, ps)
    st_cpu = fmap(x -> x isa AbstractArray ? Array(x) : x, state)
    for i in 1:batchsize:n
        idx = i:min(i + batchsize - 1, n)
        Xb = X[:, idx]
        Yb_int = Y[idx]
        Yb_oh = onehotbatch(Yb_int, 0:(n_classes-1))
        y_logits, _ = model(Xb, ps_cpu, st_cpu)
        total_loss += loss_fn(y_logits, Yb_oh)
        append!(y_true, Yb_int)
        # argmax per column returns a 1-based index; shift to 0-based
        # to match PLASTIC_TYPE_TO_IDX.
        append!(y_pred, [argmax(view(y_logits, :, j)) - 1
                         for j in 1:size(y_logits, 2)])
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

    # --- 0) CLI args ---
    # `--epochs=N` (or `--epochs N`) overrides the default 40 epochs and
    # disables early stopping (best-val-loss checkpointing still happens).
    # `--entropy=λ` adds an entropy regularizer to the cross-entropy loss
    # (pushes softmax outputs toward uniform; λ=0 is pure CE).
    epochs_override = nothing
    entropy_override = 0f0
    for (i, a) in enumerate(ARGS)
        if startswith(a, "--epochs=")
            epochs_override = parse(Int, a[length("--epochs=")+1:end])
        elseif a == "--epochs" && i < length(ARGS)
            epochs_override = parse(Int, ARGS[i+1])
        elseif startswith(a, "--entropy=")
            entropy_override = parse(Float32, a[length("--entropy=")+1:end])
        elseif a == "--entropy" && i < length(ARGS)
            entropy_override = parse(Float32, ARGS[i+1])
        end
    end
    force_run = epochs_override !== nothing
    if force_run
        println("CLI: --epochs=$epochs_override  →  forcing full run, early stop disabled")
    end
    if entropy_override > 0
        println("CLI: --entropy=$entropy_override  →  entropy regularizer on (calibration)")
    end

    # --- 1) Load data ---
    println("=== Loading plastic data ===")
    data = IRSpectraML.cached_plastic_data(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        lab_root=LAB_ROOT,
        spec_len=3000,
        val_frac=0.15,
        seed=42,
    )
    spec_len = data.spec_len
    println("\nSpectrum length: $spec_len")
    println("Train: $(size(data.X_tr, 2))  Val: $(size(data.X_val, 2))  Test: $(length(data.Y_test))  Lab: $(length(data.Y_lab))")
    println("Classes: $(IRSpectraML.PLASTIC_TYPES)")

    # Class distribution (train)
    for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
        n_i = count(==(i - 1), data.Y_tr)
        println("  $t: $n_i ($(round(100*n_i/length(data.Y_tr), digits=1))%)")
    end

    # --- 2) Savitzky–Golay smoothing (uniform across all splits) ---
    # Applies identically to train/val/test/lab so the model's input
    # distribution at eval matches training. Tunable below.
    # window=5, order=2: ~6 cm⁻¹ window, mild smoothing.
    # window=11, order=3 (default): ~13 cm⁻¹ window, stronger smoothing.
    SG_WINDOW = 11
    SG_ORDER = 3
    function smooth_split(X::Matrix{Float32})
        Xs = similar(X)
        for j in axes(X, 2)
            Xs[:, j] = IRSpectraML.smooth_spectrum(view(X, :, j);
                window=SG_WINDOW, order=SG_ORDER)
        end
        return Xs
    end
    X_tr_s = smooth_split(data.X_tr)
    X_val_s = smooth_split(data.X_val)
    X_test_s = size(data.X_test, 2) > 0 ? smooth_split(data.X_test) : data.X_test
    X_lab_s = size(data.X_lab, 2) > 0 ? smooth_split(data.X_lab) : data.X_lab
    println("\nApplied Savitzky–Golay smoothing (window=$SG_WINDOW, order=$SG_ORDER) to all splits.")

    # --- 3) SNV (per-spectrum mean/std) ---
    # Removes intensity scale + baseline slope, leaving relative shape.
    # Applied to all splits so eval distribution matches training.
    X_tr_sn = IRSpectraML.apply_snv(X_tr_s)
    X_val_sn = IRSpectraML.apply_snv(X_val_s)
    X_test_sn = size(X_test_s, 2) > 0 ? IRSpectraML.apply_snv(X_test_s) : X_test_s
    X_lab_sn = size(X_lab_s, 2) > 0 ? IRSpectraML.apply_snv(X_lab_s) : X_lab_s
    println("Applied SNV (per-spectrum mean-centering + std-scaling) to all splits.")

    # --- 4) Per-bin z-score, fit on TRAIN only ---
    norm = fit_plastic_norm(X_tr_sn)

    X_tr_n = apply_norm(norm, X_tr_sn)
    X_val_n = apply_norm(norm, X_val_sn)
    X_test_n = size(X_test_sn, 2) > 0 ? apply_norm(norm, X_test_sn) : X_test_sn
    X_lab_n = size(X_lab_sn, 2) > 0 ? apply_norm(norm, X_lab_sn) : X_lab_sn

    # --- 4) Build model ---
    model = IRSpectraML.build_plastic_model(spec_len; n_classes=IRSpectraML.N_PLASTIC, n_spatial=24)
    parameters, state = Lux.setup(Random.default_rng(), model) |> dev
    n_params = IRSpectraML.count_params(parameters)
    println("\nModel parameters: $n_params  (output: $(IRSpectraML.N_PLASTIC) classes)")

    # --- 5) Skip training if checkpoint exists ---
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
        st_cpu = JLD2.load(PLASTIC_MODEL, "states")
        state = fmap(x -> x isa AbstractArray ? x |> dev : x, st_cpu)
        best_val_acc = Float32(JLD2.load(PLASTIC_MODEL, "best_val_acc"))
        println("  Loaded. best_val_acc=$(round(100*best_val_acc, digits=2))%. Delete $PLASTIC_MODEL to retrain.")
    else
        parameters, state, best_val_acc = IRSpectraML.train_plastic_model!(
            model, parameters, state,
            X_tr_n, data.Y_tr,
            X_val_n, data.Y_val;
            X_te=isempty(data.Y_test) ? nothing : X_test_n,
            Y_te=isempty(data.Y_test) ? nothing : data.Y_test,
            epochs=force_run ? epochs_override : 40,
            lr_start=1f-3,
            lr_min=1f-6,
            batchsize=64,
            patience=8,
            resume=true,
            force=force_run,
            entropy=entropy_override,
            label_names=IRSpectraML.PLASTIC_TYPES,
            dev=dev,
        )

        println("\nSaving model → $PLASTIC_MODEL")
        cpu_dev = cpu_device()
        JLD2.save(PLASTIC_MODEL,
            "params", fmap(cpu_dev, parameters),
            "states", fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, state),
            "arch_version", IRSpectraML.ARCH_VERSION,
            "best_val_acc", best_val_acc,
            "label_names", IRSpectraML.PLASTIC_TYPES,
            "spec_len", spec_len,
        )
        JLD2.save(NORM_PATH, "mu", norm.μ, "sigma", norm.σ)
        println("  Saved.")
    end

    # --- 6) Final evaluation ---
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
            t, 100 * metrics_val.acc[i], 100 * metrics_val.prec[i],
            100 * metrics_val.rec[i], 100 * metrics_val.f1[i])
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
                t, 100 * m_te.acc[i], 100 * m_te.prec[i],
                100 * m_te.rec[i], 100 * m_te.f1[i])
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
                t, 100 * m_lab.acc[i], 100 * m_lab.prec[i],
                100 * m_lab.rec[i], 100 * m_lab.f1[i])
        end
    end

    # --- 7) Poseidon (marine microplastics, third domain) ---
    # Loaded as a third eval set. Spectra are interpolated onto the same grid
    # as the rest of the data, but with FLAT extrapolation (no linear guess
    # outside the measured 600-4000 range). Split by quality (clean / fouling /
    # " like") and reported separately so per-bucket behavior is visible.
    if isdir(POSEIDON_ROOT) && isfile(POSEIDON_LABELS)
        println("\n=== POSEIDON (marine microplastics) ===")
        try
            poseidon = IRSpectraML.load_poseidon_dataset(POSEIDON_ROOT, POSEIDON_LABELS;
                spec_len=spec_len,
                wn_min=400.0, wn_max=4000.0)
            for bucket in (:clean, :fouling, :like)
                Xb = getfield(poseidon, Symbol("X_", bucket))
                Yb = getfield(poseidon, Symbol("Y_", bucket))
                if size(Xb, 2) == 0
                    println("  $bucket: no samples (skipped)")
                    continue
                end
                Xb_s = smooth_split(Xb)
                Xb_sn = IRSpectraML.apply_snv(Xb_s)
                Xb_n = apply_norm(norm, Xb_sn)
                ev = evaluate(model, parameters, state, Xb_n, Yb; dev=dev)
                println("  $bucket (n=$(size(Xb, 2)))  acc=$(round(100*ev.acc, digits=2))%  loss=$(round(ev.loss, digits=4))")
                cm = confusion_matrix(ev.y_true, ev.y_pred, IRSpectraML.N_PLASTIC)
                m = per_class_metrics(cm)
                println("    Macro F1: $(round(100*m.macro_f1, digits=2))%")
                for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
                    @printf("      %-6s acc=%5.2f%%  prec=%5.2f%%  rec=%5.2f%%  F1=%5.2f%%\n",
                        t, 100 * m.acc[i], 100 * m.prec[i],
                        100 * m.rec[i], 100 * m.f1[i])
                end
            end
            if poseidon.skipped > 0
                println("  ($(poseidon.skipped) spectra skipped during load)")
            end
        catch e
            @warn "Poseidon eval failed: $e"
        end
    end

    # --- 7) BN running-stat refresh (lab + online mix) ---
    # Calibrate BatchNorm stats to a mix of online + lab data, then
    # re-evaluate. Refresh is eval-time only — the refreshed state is
    # not saved back to the checkpoint.
    #=
    if size(X_lab_n, 2) > 0
        println("\n=== BN running-stat refresh ===")
        # Mix: all 60 lab + 60 stratified online (10 per class).
        rng = Random.MersenneTwister(0)
        on_idx = Int[]
        for c in 0:(IRSpectraML.N_PLASTIC - 1)
            append!(on_idx, randperm(count(==(c), data.Y_tr))[1:10])
        end
        X_mix = hcat(X_tr_n[:, on_idx], X_lab_n)
        Y_mix = vcat(data.Y_tr[on_idx], data.Y_lab)
        # Shuffle the mix so the order isn't stratified.
        mix_perm = randperm(rng, length(Y_mix))
        X_mix = X_mix[:, mix_perm]
        Y_mix = Y_mix[mix_perm]
        println("  Adapting BN on $(length(Y_mix)) spectra (60 online + 60 lab) for 1 pass ...")
        refreshed_state = IRSpectraML.adapt_bn_stats!(
            model, parameters, state, X_mix, Y_mix;
            batchsize=32, n_passes=1,
        )
        # Use the refreshed state for re-evaluation. It is already in
        # testmode-of-the-new-stats, so we pass it straight to evaluate.
        println("\n=== AFTER BN REFRESH ===")
        if size(X_test_n, 2) > 0
            te_eval2 = evaluate(model, parameters, refreshed_state, X_test_n, data.Y_test; dev=dev)
            println("  TEST  acc=$(round(100*te_eval2.acc, digits=2))%  loss=$(round(te_eval2.loss, digits=4))")
        end
        if size(X_lab_n, 2) > 0
            lab_eval2 = evaluate(model, parameters, refreshed_state, X_lab_n, data.Y_lab; dev=dev)
            println("  LAB   acc=$(round(100*lab_eval2.acc, digits=2))%  loss=$(round(lab_eval2.loss, digits=4))")
            cm_lab2 = confusion_matrix(lab_eval2.y_true, lab_eval2.y_pred, IRSpectraML.N_PLASTIC)
            m_lab2 = per_class_metrics(cm_lab2)
            println("  Macro F1: $(round(100*m_lab2.macro_f1, digits=2))%")
            for (i, t) in enumerate(IRSpectraML.PLASTIC_TYPES)
                @printf("    %-6s acc=%5.2f%%  prec=%5.2f%%  rec=%5.2f%%  F1=%5.2f%%\n",
                        t, 100*m_lab2.acc[i], 100*m_lab2.prec[i],
                        100*m_lab2.rec[i], 100*m_lab2.f1[i])
            end
        end
    end
    =#
end

main()
