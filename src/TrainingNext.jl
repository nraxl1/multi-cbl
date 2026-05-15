############################################################
# TRAINING LOOP — Lux.jl port
############################################################

using Functors: fmap

const CHECKPOINT_PATH = "checkpoint.jld2"

# ──────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────

"""
    binary_focal_loss(logits, targets; γ=0.6, ϵ=1e-6)

Binary focal loss accepting raw logits (applies sigmoid internally).
Matches the original Flux `Flux.binary_focal_loss(clamp.(sigmoid.(…), …), …)`.
"""
function binary_focal_loss(logits, targets; γ=0.6f0, ϵ=1f-6)
    p = clamp.(sigmoid.(logits), ϵ, 1f0 - ϵ)
    p_t = targets .* p .+ (1f0 .- targets) .* (1f0 .- p)
    ce = -targets .* log.(p .+ ϵ) .- (1f0 .- targets) .* log.(1f0 .- p .+ ϵ)
    return mean(((1f0 .- p_t) .^ γ) .* ce)
end

# ──────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────

function save_checkpoint(ps, st, opt_state, epoch, best_val_loss)
    println("  Saving checkpoint at epoch $epoch (val loss = $(round(best_val_loss, digits=4)))")
    cpu_dev = cpu_device()
    JLD2.save(CHECKPOINT_PATH,
        "params",        fmap(x -> cpu_dev(x), ps),
        "states",        fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, st),
        "opt_state",     fmap(x -> x isa AbstractArray ? cpu_dev(x) : x, opt_state),
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

    ps = fmap(x -> dev(x), ps_cpu)
    st = fmap(x -> x isa AbstractArray ? dev(x) : x, st_cpu)

    # Restore optimiser state if available
    opt_state = try
        opt_cpu = JLD2.load(CHECKPOINT_PATH, "opt_state")
        fmap(x -> x isa AbstractArray ? dev(x) : x, opt_cpu)
    catch
        println("  Could not restore optimiser state (old checkpoint?), resetting Adam.")
        Optimisers.setup(Optimisers.Adam(lr_start), ps)
    end

    println("  Resuming from epoch $(epoch+1), best val loss = $(round(best_loss, digits=4))")
    return (ps, st, opt_state, epoch, Float32(best_loss))
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

    dev = gpu_device()

    # --- Optimiser ---
    opt_state = Optimisers.setup(Optimisers.Adam(lr_start), ps)

    # --- Resume from checkpoint ---
    if resume
        ckpt = maybe_load_checkpoint(model, dev, lr_start)
        if ckpt !== nothing
            ps, st, opt_state, start_epoch, best_val_loss = ckpt
        else
            start_epoch, best_val_loss = 0, Inf32
        end
    else
        start_epoch, best_val_loss = 0, Inf32
    end
    epochs_no_improve = 0

    val_loader = DataLoader((Xv, Yv), batchsize=256,
                            partial=false, parallel=true) |> dev

    @showprogress desc="Epochs" for e in (start_epoch+1):epochs

        # --- Cosine LR decay ---
        t = Float32(e - 1) / Float32(epochs - 1)
        lr = Float32(lr_min + 0.5f0 * (lr_start - lr_min) * (1f0 + cos(Float32(π) * t)))
        Optimisers.adjust!(opt_state, lr)

        @showprogress desc="Chunks" offset=1 for path in shuffle(chunk_paths)
            X, Y, s = cached_load_chunk(path)
            tr_idx = findall(sv -> sv < 8, s)
            Xtr = apply_normalizer(norm, X[:, tr_idx])
            Ytr = Y[:, tr_idx]
            perm = randperm(size(Xtr, 2))
            train_loader = DataLoader((Xtr[:, perm], Ytr[:, perm]),
                                      batchsize=256, shuffle=false,
                                      partial=false, parallel=true) |> dev

            for (x, y) in train_loader
                # Forward pass — updates st (BatchNorm running stats, Dropout RNG)
                y_pred, st = model(x, ps, st)
                loss_val  = binary_focal_loss(y_pred, y)

                # Gradient via Enzyme (source-to-source AD — lower memory, no tape)
                grads = Enzyme.gradient(Enzyme.Reverse, ps) do p
                    yp, _ = model(x, p, st)
                    return binary_focal_loss(yp, y)
                end

                # Parameter update
                ps = Optimisers.update!(opt_state, ps, grads)
            end
        end

        # --- Validation ---
        st_val = Lux.test_mode(st)
        val_loss = 0f0
        n_val = 0
        for (Xb, Yb) in val_loader
            y_pred, _ = model(Xb, ps, st_val)
            val_loss += binary_focal_loss(y_pred, Yb)
            n_val += 1
        end
        val_loss /= n_val

        @printf("Epoch %2d | val loss = %.4f | lr = %.2e\n", e, val_loss, lr)

        # --- Checkpoint if improved ---
        if val_loss < best_val_loss
            best_val_loss      = val_loss
            epochs_no_improve  = 0
            save_checkpoint(ps, st, opt_state, e, best_val_loss)
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
        best_ps = fmap(x -> dev(x), JLD2.load(CHECKPOINT_PATH, "params"))
        best_st = fmap(x -> x isa AbstractArray ? dev(x) : x,
                       JLD2.load(CHECKPOINT_PATH, "states"))
        return best_ps, best_st
    end

    return ps, st
end
