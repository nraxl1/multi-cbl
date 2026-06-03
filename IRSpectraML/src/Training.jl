############################################################
# TRAINING LOOP — Lux.jl + Reactant + Enzyme
#
# Two task heads supported:
#   * Plastic classification (mutually exclusive, N_PLASTIC classes)
#   * (Kept commented) Functional-group multi-label prediction
#
# Public entry points:
#   * train_plastic_model!(...)    — classification
#   * train_model!(...)            — legacy multi-label (commented out)
############################################################
using Functors: fmap
using Lux, Reactant, Optimisers
using DispatchDoctor

const CHECKPOINT_PATH = "checkpoint.jld2"
const PLASTIC_CHECKPOINT_PATH = "plastic_model.jld2"

const ARCH_VERSION = "plastic-v1-3k-classifier"

# ──────────────────────────────────────────────────────────
# Checkpointing (shared, version-gated)
# ──────────────────────────────────────────────────────────

function save_plastic_checkpoint(train_state, epoch, best_val_acc, label_names)
    println("  Saving plastic checkpoint at epoch $epoch (val acc = $(round(100*best_val_acc, digits=2))%)")
    cpu_dev = cpu_device()
    JLD2.save(PLASTIC_CHECKPOINT_PATH,
        "params",        fmap(cpu_dev, train_state.parameters),
        "states",        fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, train_state.states),
        "epoch",         epoch,
        "best_val_acc",  best_val_acc,
        "arch_version",  ARCH_VERSION,
        "label_names",   label_names,
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

    ps_cpu     = JLD2.load(PLASTIC_CHECKPOINT_PATH, "params")
    st_cpu     = JLD2.load(PLASTIC_CHECKPOINT_PATH, "states")
    epoch      = JLD2.load(PLASTIC_CHECKPOINT_PATH, "epoch")
    best_acc   = JLD2.load(PLASTIC_CHECKPOINT_PATH, "best_val_acc")

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
    pred = [argmax(view(logits, :, i)) for i in 1:size(logits, 2)]
    return Float32(mean(pred .== y_true))
end


# ──────────────────────────────────────────────────────────
# Plastic classification training
# ──────────────────────────────────────────────────────────

"""
    train_plastic_model!(model, ps, st, X_tr, Y_tr, X_val, Y_val;
                         X_te=nothing, Y_te=nothing,
                         epochs=30, lr_start=1f-3, lr_min=1f-6,
                         batchsize=64, patience=8, resume=true, label_names=PLASTIC_TYPES)

Train a 6-way plastic classifier. Uses cross-entropy loss with logits,
cosine LR decay, early stopping on validation accuracy, and a Reactant-compiled
inner loop.
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
                              label_names::Vector{String}=PLASTIC_TYPES)

    dev = reactant_device()
    ps = fmap(dev, ps)
    st = fmap(x -> x isa AbstractArray ? dev(x) : x, st)

    # Cross-entropy on raw logits
    loss_fn = CrossEntropyLoss(; logits=true)

    opt = Optimisers.AdamW(lr_start)
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

    val_loader = DataLoader((X_val, Y_val), batchsize=512, partial=false, parallel=true)

    # Compile model and loss once with a sample input
    (x0, y0) = first(val_loader) |> dev
    model_compiled = @compile sync=true train_state.model(x0, train_state.parameters, Lux.testmode(train_state.states))
    _pred, _ = model_compiled(x0, train_state.parameters, Lux.testmode(train_state.states))
    loss_fn_compiled = @compile sync=true loss_fn(_pred, y0)
    print("done compiling thingy\n")

    n_train = size(X_tr, 2)
    n_val   = size(X_val, 2)
    println("Training: $n_train samples, Val: $n_val samples, batch=$batchsize, epochs=$epochs")

    for e in (start_epoch+1):epochs
        GC.gc(true)

        # --- Cosine LR decay ---
        t = Float32(e - 1) / Float32(epochs - 1)
        lr = Float32(lr_min + 0.5f0 * (lr_start - lr_min) * (1f0 + cos(Float32(π) * t)))
        Optimisers.adjust!(train_state.optimizer_state, lr)

        # --- Train one epoch (single pass over the data, batched) ---
        perm = randperm(n_train)
        train_loader = DataLoader((X_tr[:, perm], Y_tr[perm]),
                                  batchsize=batchsize, shuffle=false,
                                  partial=false, parallel=true) |> dev

        epoch_loss = 0f0
        n_batches = 0
        for (x, y) in train_loader
            _, loss_val, _, train_state = Lux.Training.single_train_step!(
                AutoReactant(),
                loss_fn,
                (x, y),
                train_state
            )
            epoch_loss += loss_val
            n_batches += 1
        end
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        st_val = Lux.testmode(train_state.states)
        val_loss = 0f0
        val_correct = 0
        val_count = 0
        for (Xb, Yb) in dev(val_loader)
            Xb = Xb |> dev
            Yb = Yb |> dev
            y_pred, _ = model_compiled(Xb, train_state.parameters, st_val)
            val_loss += loss_fn_compiled(y_pred, Yb)
            # accuracy on GPU
            val_correct += Int(sum(argmax.(eachcol(y_pred)) .== Yb))
            val_count  += length(Yb)
        end
        val_loss /= max(length(val_loader), 1)
        val_acc   = val_count > 0 ? Float32(val_correct / val_count) : 0f0

        @printf("Epoch %2d | train_loss=%.4f | val_loss=%.4f | val_acc=%5.2f%% | lr=%.2e\n",
                e, train_loss, val_loss, 100 * val_acc, lr)

        # --- Checkpoint if improved ---
        if val_acc > best_val_acc
            best_val_acc     = val_acc
            epochs_no_improve = 0
            save_plastic_checkpoint(train_state, e, best_val_acc, label_names)
        else
            epochs_no_improve += 1
            println("  No improvement ($epochs_no_improve/$patience)")
        end

        if epochs_no_improve >= patience
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
# Kept for the contrastive pretraining pipeline (Phase 1).
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

    val_loader = DataLoader((Xv, Yv), batchsize=512,
                            partial=false, parallel=true)

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

        for path in shuffle(chunk_paths)
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
            show(typeof(y_pred))
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

        if epochs_no_improve >= patience
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
