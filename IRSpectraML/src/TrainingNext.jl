############################################################
# TRAINING LOOP — Lux.jl + Reactant + Enzyme
############################################################
using Functors: fmap
using Lux
using DispatchDoctor

const CHECKPOINT_PATH = "checkpoint.jld2"

# ──────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────

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

    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr_start))

    println("  Resuming from epoch $(epoch+1), best val loss = $(round(best_loss, digits=4))")
    return (train_state, epoch, Float32(best_loss))
end

# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────

function train_model!(model, ps, st, chunk_paths::Vector{String},
                      norm::Normalizer,
                      Xv::Matrix{Float32}, Yv::Matrix{Float32};
                      epochs=30,
                      lr_start=1f-3,
                      lr_min=1f-6,
                      patience=5,
                      resume=true)

    dev = reactant_device()

    loss_fn = Lux.BinaryFocalLoss(; gamma = 0.6) # BinaryFocalLoss(; γ=0.6f0, ϵ=1f-6)

    opt = Optimisers.Adam(lr_start)
    train_state = Lux.Training.TrainState(model, ps, st, opt)

    # --- Resume from checkpoint ---
    start_epoch, best_val_loss = 0, Inf32
    epochs_no_improve = 0
    if resume
        ckpt = maybe_load_checkpoint(model, dev, lr_start)
        if ckpt !== nothing
            train_state, start_epoch, best_val_loss = ckpt
        end
    end

    val_loader = DataLoader((Xv, Yv), batchsize=32,
                            partial=false, parallel=true)

    for e in (start_epoch+1):epochs

        # --- Cosine LR decay ---
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
                                      batchsize=32, shuffle=false,
                                      partial=false, parallel=true)

            for (x, y) in dev(train_loader)
                # x = x |> dev
                # y = y |> dev

                # Compute gradients (updates st internally via Lux)
                gs, loss_val, _, train_state = Lux.Training.compute_gradients(
                    AutoEnzyme(),
                    loss_fn,
                    (x, y),
                    train_state
                )

                # Parameter update
                train_state = Lux.Training.apply_gradients!(train_state, gs)
            end
        end

        # --- Validation ---
        st_val = Lux.testmode(train_state.states)
        val_loss = 0f0
        n_val = 0
        for (Xb, Yb) in dev(val_loader)
            # Xb = Xb |> dev
            # Yb = Yb |> dev
            y_pred, _ = @jit Lux.apply(train_state.model, Xb, train_state.parameters, st_val)
            val_loss += loss_fn(y_pred, Yb)
            n_val += 1
        end
        val_loss /= n_val

        @printf("Epoch %2d | val loss = %.4f | lr = %.2e\n", e, val_loss, lr)

        # --- Checkpoint if improved ---
        if val_loss < best_val_loss
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            save_checkpoint(train_state, e, best_val_loss)
        else
            epochs_no_improve += 1
            println("  No improvement ($epochs_no_improve/$patience)")
        end

        # --- Early stopping ---
        if epochs_no_improve >= patience
            println("\nEarly stopping at epoch $e (no improvement for $patience epochs)")
            break
        end
    end

    # --- Restore best weights ---
    if isfile(CHECKPOINT_PATH)
        println("\nRestoring best model weights from checkpoint...")
        best_ps = fmap(x -> x |> dev, JLD2.load(CHECKPOINT_PATH, "params"))
        best_st = fmap(x -> x isa AbstractArray ? x |> dev : x,
                       JLD2.load(CHECKPOINT_PATH, "states"))
        return best_ps, best_st
    end

    return train_state.parameters, train_state.states
end
