############################################################
# TRAINING LOOP — Lux.jl + Reactant (CPU backend for now)
#
# Two task heads supported:
#   * Plastic classification (mutually exclusive, N_PLASTIC classes)
#   * (Kept commented) Functional-group multi-label prediction
#
# Public entry points:
#   * train_plastic_model!(...)    — classification, Reactant-CPU
#   * train_model!(...)            — legacy multi-label (commented out)
#
# Backend: Reactant on CPU. To switch to GPU on a CUDA box, change
# `Reactant.set_default_backend("cpu")` to "gpu" in Main.jl — the rest of
# the code is identical (the model and loss are backend-agnostic).
#
# IMPORTANT: Lux's `CrossEntropyLoss` expects one-hot encoded targets
# (n_classes, batch), not integer class indices. We use OneHotArrays and
# JIT the onehotbatch call alongside the model so Reactant can trace
# through the dtype conversion (see LuxDL/Lux.jl#1556).
############################################################
using Functors: fmap
using Lux, Reactant, Optimisers, OneHotArrays
using DispatchDoctor
using Random: shuffle as _shuffle

const CHECKPOINT_PATH = "checkpoint.jld2"
const PLASTIC_CHECKPOINT_PATH = "plastic_model.jld2"

const ARCH_VERSION = "plastic-v1-3k-classifier-reactant-cpu"

# ──────────────────────────────────────────────────────────
# Checkpointing (version-gated)
# ──────────────────────────────────────────────────────────

function save_plastic_checkpoint(train_state, epoch, best_val_acc, label_names)
    println("  Saving plastic checkpoint at epoch $epoch (val acc = $(round(100*best_val_acc, digits=2))%)")
    cpu_dev = cpu_device()
    JLD2.save(PLASTIC_CHECKPOINT_PATH,
        "params", fmap(cpu_dev, train_state.parameters),
        "states", fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, train_state.states),
        "epoch", epoch,
        "best_val_acc", best_val_acc,
        "arch_version", ARCH_VERSION,
        "label_names", label_names,
    )
end

function maybe_load_plastic_checkpoint(model, dev, lr_start)
    !isfile(PLASTIC_CHECKPOINT_PATH) && return nothing

    saved_arch = JLD2.load(PLASTIC_CHECKPOINT_PATH, "arch_version")
    if saved_arch != ARCH_VERSION
        println("Plastic checkpoint arch '$saved_arch' ≠ current '$ARCH_VERSION' — ignoring checkpoint.")
        return nothing
    end

    println("Resuming plastic model from $PLASTIC_CHECKPOINT_PATH ...")

    ps_cpu = JLD2.load(PLASTIC_CHECKPOINT_PATH, "params")
    st_cpu = JLD2.load(PLASTIC_CHECKPOINT_PATH, "states")
    epoch = JLD2.load(PLASTIC_CHECKPOINT_PATH, "epoch")
    best_acc = JLD2.load(PLASTIC_CHECKPOINT_PATH, "best_val_acc")

    ps = fmap(x -> x |> dev, ps_cpu)
    st = fmap(x -> x isa AbstractArray ? x |> dev : x, st_cpu)

    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lr_start))
    println("  Resuming from epoch $(epoch+1), best val acc = $(round(100*best_acc, digits=2))%")
    return (train_state, epoch, Float32(best_acc))
end

# ──────────────────────────────────────────────────────────
# Accuracy metric (mutually exclusive classification)
# ──────────────────────────────────────────────────────────

"""
    classification_accuracy(logits, y_true) -> Float32

`logits` is (n_classes, batch) Float32, `y_true` is 0-indexed Int batch vector.
Returns mean accuracy.
"""
function classification_accuracy(logits::AbstractMatrix, y_true::AbstractVector{<:Integer})
    # argmax on a column returns a 1-based index; labels are 0-indexed,
    # so shift down by 1 to compare apples-to-apples.
    pred = [argmax(view(logits, :, i)) - 1 for i in 1:size(logits, 2)]
    return Float32(mean(pred .== y_true))
end


# ──────────────────────────────────────────────────────────
# BN running-stat refresh (unsupervised domain adaptation)
# ──────────────────────────────────────────────────────────

#=
"""
    adapt_bn_stats!(model, ps, st, X, Y; batchsize=32, n_passes=3, dev=cpu_device())

Recalibrate BatchNorm `running_mean` / `running_var` to a target
distribution. Sets the model to `trainmode` (so BN updates its running
stats from batch statistics via the standard momentum), runs `n_passes`
forward passes over `(X, Y)` with no gradient, then returns the
updated state.

Use this to bridge a small distribution shift between training and
target domains (e.g., recalibrate to a different spectrometer). No
labels are used for the optimization itself — they're just passed
through for shape consistency.

Data should be a *mix* of source and target domains if you want the
refreshed stats to remain usable on both; pure target-domain data
will make the model lab-specific and hurt source-domain performance.

NOTE: currently disabled — Reactant intercepts the trainmode forward
path in this setup. See Main.jl comment block for the call site that
this would feed back into.
"""
function adapt_bn_stats!(model, ps, st,
                        X::Matrix{Float32}, Y::Vector{Int};
                        batchsize::Int=32, n_passes::Int=3)
    # Reactant can't trace a plain non-JIT forward in trainmode (the
    # conv im2col path needs CPU pointers), so we work on plain CPU
    # arrays. Lux's BatchNorm in trainmode without an autodiff context
    # takes a slow path; we pay the JIT cost once via @compile and
    # reuse the compiled function for all batches.
    ps_cpu = fmap(x -> x isa AbstractArray ? Array(x) : x, ps)
    st_cpu = fmap(x -> x isa AbstractArray ? Array(x) : x, st)
    st_train = Lux.trainmode(st_cpu)

    # JIT-compile a sample forward so subsequent calls are fast.
    sample_x = X[:, 1:min(batchsize, size(X, 2))]
    compiled = @compile sync=true model(sample_x, ps_cpu, st_train)
    _, st_train = compiled(sample_x, ps_cpu, st_train)   # one warmup

    n = size(X, 2)
    for pass in 1:n_passes
        perm = randperm(n)
        for i in 1:batchsize:n
            idx = perm[i:min(i+batchsize-1, n)]
            xb = X[:, idx]
            _, st_train = compiled(xb, ps_cpu, st_train)   # forward only
        end
    end
    return st_train
end
=#


# ──────────────────────────────────────────────────────────
# Plastic classification training (Reactant-CPU)
# ──────────────────────────────────────────────────────────

"""
    train_plastic_model!(model, ps, st, X_tr, Y_tr, X_val, Y_val;
                         X_te=nothing, Y_te=nothing,
                         epochs=30, lr_start=1f-3, lr_min=1f-6,
                         batchsize=64, patience=8, resume=true,
                         label_names=PLASTIC_TYPES, dev=reactant_device())

Train a 6-way plastic classifier. Uses cross-entropy loss with logits,
cosine LR decay, early stopping on validation accuracy, and Reactant for
XLA compilation (currently on the CPU backend).

Device: defaults to `reactant_device()`. The model + loss + onehot are
JIT-compiled once at the start of training via `@compile sync=true`, then
reused across all batches.

To switch to GPU on a CUDA box:
  Reactant.set_default_backend("gpu")   # before calling main()
  ... rest of the code is identical
"""
function train_plastic_model!(model, ps, st,
    X_tr::Matrix{Float32}, Y_tr::Vector{Int},
    X_val::Matrix{Float32}, Y_val::Vector{Int};
    X_te::Union{Matrix{Float32},Nothing}=nothing,
    Y_te::Union{Vector{Int},Nothing}=nothing,
    epochs::Int=30,
    lr_start::Float32=1f-3,
    lr_min::Float32=1f-6,
    batchsize::Int=64,
    patience::Int=8,
    resume::Bool=true,
    force::Bool=false,
    entropy::Float32=0f0,
    label_names::Vector{String}=PLASTIC_TYPES,
    dev::Any=reactant_device())

    best_val_loss = Inf32
    min_delta = 5f-5
    epochs_no_improve = 0

    ps = fmap(dev, ps)
    st = fmap(x -> x isa AbstractArray ? dev(x) : x, st)

    # Cross-entropy on raw logits, optionally with an entropy regularizer
    # (pushes the softmax output toward uniform, dampens overconfidence).
    # `entropy` (λ) is captured by the closure so it is baked into the
    # JIT-compiled loss at compile time. λ=0 → pure cross-entropy, identical
    # to the previous behavior.
    loss_fn = CrossEntropyLoss(; logits=true)
    λ = Float32(entropy)
    # loss_fn = if λ > 0
    #    (logits, y_oh) -> begin
    #        ce = ce_fn(logits, y_oh)
    #        # Numerically stable softmax: subtract max before exp
    #        m    = maximum(logits; dims=1)
    #        z    = exp.(logits .- m)
    #        p    = z ./ sum(z; dims=1)
    #        # Mean per-sample entropy over the batch: H = -Σ p log p
    #        # ε floor keeps log defined when a class probability is 0.
    #        # Note: don't wrap with Float32(...) — that fails inside the
    #        # XLA trace (Float32(::TracedRNumber) is undefined).
    #        H    = -mean(sum(p .* log.(p .+ 1f-8); dims=1))
    #        ce + λ * H
    #    end
    # else
    #    ce_fn
    # end
    opt = Optimisers.AdamW(lr_start, (0.01f0, 0.01f0))
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    # --- Resume from checkpoint ---
    start_epoch, best_val_acc = 0, -1f0
    epochs_no_improve = 0
    if resume
        ckpt = maybe_load_plastic_checkpoint(model, dev, lr_start)
        if ckpt !== nothing
            train_state, start_epoch, best_val_acc = ckpt
        end
    end

    n_train = size(X_tr, 2)
    n_val = size(X_val, 2)
    n_classes = length(label_names)

    # --- JIT-compile model + loss + onehot once on a sample input ---
    sample_x = X_tr[:, 1:min(batchsize, n_train)] |> dev
    sample_y_int = Y_tr[1:min(batchsize, n_train)] |> dev
    @info "JIT-compiling model + loss + onehot with Reactant (one-time cost)..."
    t_compile = time()

    # Wrap onehot in a closure so the UnitRange is captured (XLA traces the
    # signature of a single argument — see LuxDL/Lux.jl#1556).
    onehot_fn = let classes = 0:(n_classes-1)
        y -> onehotbatch(y, classes)
    end
    onehot_compiled = @compile sync = true onehot_fn(sample_y_int)

    sample_y_oh = onehot_compiled(sample_y_int)
    model_compiled = @compile sync = true train_state.model(sample_x, train_state.parameters, Lux.testmode(train_state.states))
    _pred, _ = model_compiled(sample_x, train_state.parameters, Lux.testmode(train_state.states))
    loss_fn_compiled = @compile sync = true loss_fn(_pred, sample_y_oh)
    println("  compiled in $(round(time() - t_compile, digits=1))s")

    println("Training: $n_train samples, Val: $n_val samples, batch=$batchsize, epochs=$epochs, dev=$(typeof(dev))")

    # Last-batch-safe loop: drop the tail batch if smaller than batchsize so
    # the JIT'd onehot/model don't see a different shape than they were
    # compiled for. We require batchsize to divide both splits cleanly; if
    # it doesn't, we auto-truncate to the largest multiple of batchsize.
    if n_train % batchsize != 0
        @warn "n_train ($n_train) not divisible by batchsize ($batchsize); truncating last batch"
    end
    if n_val % batchsize != 0
        @warn "n_val ($n_val) not divisible by batchsize ($batchsize); truncating last batch"
    end
    last_full_train = (n_train ÷ batchsize) * batchsize
    last_full_val = (n_val ÷ batchsize) * batchsize
    last_full_val = max(last_full_val, 0)
    n_train = last_full_train
    n_val = last_full_val
    if n_val == 0
        @warn "Validation set is empty after truncation; skipping val eval"
    end

    for e in (start_epoch+1):epochs
        GC.gc(true)

        # --- Cosine LR decay ---
        t = Float32(e - 1) / Float32(epochs - 1)
        lr = Float32(lr_min + 0.5f0 * (lr_start - lr_min) * (1f0 + cos(Float32(π) * t)))
        Optimisers.adjust!(train_state.optimizer_state, lr)

        # --- Train one epoch (single pass over the data, batched) ---
        perm = randperm(n_train)
        epoch_loss = 0f0
        n_batches = 0
        for i in 1:batchsize:n_train
            idx = perm[i:min(i + batchsize - 1, n_train)]
            xb = X_tr[:, idx] |> dev
            yb_int = Y_tr[idx] |> dev
            yb = onehot_compiled(yb_int)  # one-hot on the Reactant device

            # Single train step: forward + backward + apply update (Reactant handles AD)
            _, loss_val, _, train_state = Lux.Training.single_train_step!(
                AutoReactant(),
                loss_fn,
                (xb, yb),
                train_state
            )

            epoch_loss += loss_val
            n_batches += 1
        end
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation (uses pre-compiled model + onehot) ---
        st_val = Lux.testmode(train_state.states)
        val_loss = 0f0
        val_correct = 0
        val_count = 0
        if n_val > 0
            for i in 1:batchsize:n_val
                idx = i:min(i + batchsize - 1, n_val)
                Xb = X_val[:, idx] |> dev
                Yb_int = Y_val[idx] |> dev
                Yb_oh = onehot_compiled(Yb_int)
                y_logits, _ = model_compiled(Xb, train_state.parameters, st_val)
                val_loss += loss_fn_compiled(y_logits, Yb_oh)
                # Pull logits to a host Array, argmax per column, then shift
                # from 1-based (Julia) to 0-based (PLASTIC_TYPE_TO_IDX).
                y_logits_cpu = Array(y_logits)
                y_pred_int = [argmax(view(y_logits_cpu, :, j)) - 1
                              for j in 1:size(y_logits_cpu, 2)]
                val_correct += sum(y_pred_int .== Y_val[idx])
                val_count += length(Y_val[idx])
            end
            val_loss /= max(ceil(Int, n_val / batchsize), 1)
        end
        val_acc = val_count > 0 ? Float32(val_correct / val_count) : -1f0

        @printf("Epoch %2d | train_loss=%.4f | val_loss=%.4f | val_acc=%5.2f%% | lr=%.2e\n",
            e, train_loss, val_loss, 100 * val_acc, lr)

        # --- Checkpoint if improved ---
        # if val_acc > best_val_acc
        #    best_val_acc = val_acc
        #    epochs_no_improve = 0
        #    save_plastic_checkpoint(train_state, e, best_val_acc, label_names)
        # else
        #     epochs_no_improve += 1
        #     println("  No improvement ($epochs_no_improve/$patience)")
        # end
        improved = (n_val > 0) && (val_loss < best_val_loss - min_delta)
        if improved
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_plastic_checkpoint(train_state, e, val_loss, label_names)
        else
            epochs_no_improve += 1
            println("  No improvement ($epochs_no_improve/$patience)")

        end


        if !force && epochs_no_improve >= patience
            println("\nEarly stopping at epoch $e (no improvement for $patience epochs)")
            break
        end
    end

    # --- Restore best weights ---
    if isfile(PLASTIC_CHECKPOINT_PATH)
        println("\nRestoring best plastic model weights from checkpoint...")
        best_ps = fmap(x -> x |> dev, JLD2.load(PLASTIC_CHECKPOINT_PATH, "params"))
        best_st = fmap(x -> x isa AbstractArray ? x |> dev : x,
            JLD2.load(PLASTIC_CHECKPOINT_PATH, "states"))
        return (best_ps, best_st, Float32(best_val_acc))
    end

    return (train_state.parameters, train_state.states, Float32(best_val_acc))
end

# ──────────────────────────────────────────────────────────
# Legacy multi-label training loop (SMILES / functional groups)
# Kept for the contrastive pretraining pipeline (Phase 1) — will be
# re-enabled on a CUDA machine with the Reactant GPU backend.
# ──────────────────────────────────────────────────────────
#=
function save_checkpoint(train_state, epoch, best_val_loss)
    println("  Saving checkpoint at epoch $epoch (val loss = $(round(best_val_loss, digits=4)))")
    cpu_dev = cpu_device()
    JLD2.save(CHECKPOINT_PATH,
        "params",        fmap(cpu_dev, train_state.parameters),
        "states",        fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, train_state.states),
        "epoch",         epoch,
        "best_val_loss", best_val_loss,
        "arch_version",  ARCH_VERSION,
    )
end

function maybe_load_checkpoint(model, dev, lr_start)
    !isfile(CHECKPOINT_PATH) && return nothing

    saved_arch = JLD2.load(CHECKPOINT_PATH, "arch_version")
    if saved_arch != ARCH_VERSION
        println("Checkpoint arch '$saved_arch' ≠ current '$ARCH_VERSION' — ignoring checkpoint.")
        return nothing
    end

    println("Resuming from checkpoint at $CHECKPOINT_PATH ...")

    ps_cpu     = JLD2.load(CHECKPOINT_PATH, "params")
    st_cpu     = JLD2.load(CHECKPOINT_PATH, "states")
    epoch      = JLD2.load(CHECKPOINT_PATH, "epoch")
    best_loss  = JLD2.load(CHECKPOINT_PATH, "best_val_loss")

    ps = fmap(x -> x |> dev, ps_cpu)
    st = fmap(x -> x isa AbstractArray ? x |> dev : x, st_cpu)

    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lr_start))

    println("  Resuming from epoch $(epoch+1), best val loss = $(round(best_loss, digits=4))")
    return (train_state, epoch, Float32(best_loss))
end

function train_model!(model, ps, st, chunk_paths::Vector{String},
                      norm::Normalizer,
                      Xv::Matrix{Float32}, Yv::Matrix{Float32};
                      epochs=30,
                      lr_start=1f-3,
                      lr_min=1f-6,
                      patience=5,
                      resume=true)

    dev = reactant_device()
    ps = fmap(dev, ps)
    st = fmap(x -> x isa AbstractArray ? dev(x) : x, st)

    focal_loss = Lux.BinaryFocalLoss(gamma = 0.6)
    loss_fn = GenericLossFunction((y_pred, y) -> focal_loss(sigmoid(clamp.(y_pred, Float32(-10), Float32(10))), y))

    opt = Optimisers.AdamW(lr_start)
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    start_epoch, best_val_loss = 0, Inf32
    epochs_no_improve = 0
    if resume
        ckpt = maybe_load_checkpoint(model, dev, lr_start)
        if ckpt !== nothing
            train_state, start_epoch, best_val_loss = ckpt
        end
    end

    val_loader = DataLoader((Xv, Yv), batchsize=512, partial=false, parallel=true)

    (x0, y0) = first(val_loader) |> dev
    model_compiled = @compile sync=true train_state.model(x0, train_state.parameters, Lux.testmode(train_state.states))
    print("done compiling thingy")
    prediction, _ = model_compiled(x0, train_state.parameters, Lux.testmode(train_state.states))
    loss_fn_compiled = @compile sync=true loss_fn(prediction, y0)

    for e in (start_epoch+1):epochs
        GC.gc(true)
        t = Float32(e - 1) / Float32(epochs - 1)
        lr = Float32(lr_min + 0.5f0 * (lr_start - lr_min) * (1f0 + cos(Float32(π) * t)))
        Optimisers.adjust!(train_state.optimizer_state, lr)

        for path in _shuffle(chunk_paths)
            X, Y, s = cached_load_chunk(path)
            tr_idx = findall(sv -> sv < 8, s)
            Xtr = apply_normalizer(norm, X[:, tr_idx])
            Ytr = Y[:, tr_idx]
            perm = randperm(size(Xtr, 2))
            train_loader = DataLoader((Xtr[:, perm], Ytr[:, perm]),
                                      batchsize=64, shuffle=false,
                                      partial=false, parallel=true) |> dev

            for (x, y) in train_loader
                _, loss_val, _, train_state = Lux.Training.single_train_step!(
                    AutoReactant(),
                    loss_fn,
                    (x, y),
                    train_state
                )
            end
        end

        st_val = Lux.testmode(train_state.states)
        val_loss = 0f0
        n_val = 0
        for (Xb, Yb) in dev(val_loader)
            Xb = Xb |> dev
            Yb = Yb |> dev
            y_pred, _ = Reactant.@time model_compiled(Xb, train_state.parameters, st_val)
            y_pred = y_pred |> dev
            val_loss += loss_fn_compiled(y_pred, Yb)
            n_val += 1
        end
        val_loss /= n_val

        @printf("Epoch %2d | val loss = %.4f | lr = %.2e\n", e, val_loss, lr)

        if val_loss < best_val_loss
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            save_checkpoint(train_state, e, best_val_loss)
        else
            epochs_no_improve += 1
            println("  No improvement ($epochs_no_improve/$patience)")
        end

        if !force && epochs_no_improve >= patience
            println("\nEarly stopping at epoch $e (no improvement for $patience epochs)")
            break
        end
    end

    if isfile(CHECKPOINT_PATH)
        println("\nRestoring best model weights from checkpoint...")
        best_ps = fmap(x -> x |> dev, JLD2.load(CHECKPOINT_PATH, "params"))
        best_st = fmap(x -> x isa AbstractArray ? x |> dev : x,
                       JLD2.load(CHECKPOINT_PATH, "states"))
        return best_ps, best_st
    end

    return train_state.parameters, train_state.states
end
=#
