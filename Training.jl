############################################################
# TRAINING LOOP
############################################################


function train_model!(model, chunk_paths::Vector{String},
                      norm::Normalizer,
                      Xv::Matrix{Float32}, Yv::Matrix{Float32};
                      epochs=10)

    opt_state  = Flux.setup(Adam(1e-3), model)
    val_loader = DataLoader((Xv, Yv), batchsize=128)

    for e in 1:epochs
        Flux.trainmode!(model)
        train_loss = 0f0
        n_batches  = 0

        for path in shuffle(chunk_paths)
            X, Y, s = cached_load_chunk(path)

            tr_idx = findall(sv -> sv < 8, s)
            Xtr = apply_normalizer(norm, X[:, tr_idx])
            Ytr = Y[:, tr_idx]

            train_loader = DataLoader((Xtr, Ytr), batchsize=128, shuffle=true)

            for (Xb, Yb) in train_loader
                if CUDA.functional()
                    Xb, Yb = gpu(Xb), gpu(Yb)
                end
                loss, grads = Flux.withgradient(model) do m
                    Flux.logitbinarycrossentropy(m(Xb), Yb)
                end
                Flux.update!(opt_state, model, grads[1])
                train_loss += loss
                n_batches  += 1
            end

            X = Y = Xtr = Ytr = nothing
            GC.gc()
        end

        Flux.testmode!(model)
        val_loss = 0f0
        n_val    = 0
        for (Xb, Yb) in val_loader
            if CUDA.functional()
                Xb, Yb = gpu(Xb), gpu(Yb)
            end
            val_loss += Flux.logitbinarycrossentropy(model(Xb), Yb)
            n_val    += 1
        end

        @printf("Epoch %2d | train loss = %.4f | val loss = %.4f\n",
                e, train_loss/n_batches, val_loss/n_val)
    end
end
