############################################################
# TRAINING LOOP
############################################################

function train_model!(model, chunk_paths::Vector{String},
    norm::Normalizer,
    Xv::Matrix{Float32}, Yv::Matrix{Float32};
    epochs=10)
    dev = MLDataDevices.gpu_device()
    opt_state = Flux.setup(Adam(1e-3), model)
    val_loader = DataLoader((Xv, Yv), batchsize=256,
     partial=false, parallel=true) |> dev
    # dev = MLDataDevices.gpu_device() idk why i had 2 of these
    loss_fn(m, x, y) = Flux.binary_focal_loss(clamp.(sigmoid(m(x)), 1f-6, 1f0 - 1f-6), y, gamma=2)
    # loss_fn(m, x, y) = Flux.logitbinarycrossentropy(m(x), y)
    @showprogress desc="Epochs" for e in 1:epochs
        Flux.trainmode!(model)
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
                loss, grads = Flux.withgradient(m -> loss_fn(m, x, y), model)
                Flux.update!(opt_state, model, grads[1])
            end
            # Force GPU memory release between chunks
            # GC.gc(true)
            # CUDA.reclaim()
        end
        Flux.testmode!(model)
        val_loss = 0f0
        n_val = 0
        for (Xb, Yb) in val_loader
            # Xb, Yb = dev(Xb), dev(Yb)
            val_loss += loss_fn(model, Xb, Yb)
            n_val += 1
        end
        @printf("Epoch %2d | val loss = %.4f\n", e, val_loss / n_val)
    end
end