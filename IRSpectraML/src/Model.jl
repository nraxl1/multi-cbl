using Functors: @functor

############################################################
# MODEL — Lux.jl port of the Flux ResBlock architecture
#
# NOTE: strided convs (Conv(stride=2)) are replaced with
# stride-1 conv + MaxPool due to a Reactant bug with tracing
# strided convs across parallel paths (#1990).
# The architecture is FUNCTIONALLY the same — same channel
# progression, same skip connections, same spatial reduction.
############################################################

# --- Stateless helpers ---

struct SpecReshapeLayer <: Lux.AbstractLuxLayer
    spec_len::Int
end

Lux.initialparameters(::AbstractRNG, ::SpecReshapeLayer) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::SpecReshapeLayer)     = NamedTuple()
Lux.parameterlength(::SpecReshapeLayer)                  = 0
Lux.statelength(::SpecReshapeLayer)                      = 0

(m::SpecReshapeLayer)(x, ps, st) = (reshape(x, m.spec_len, 1, :), st)


struct FlattenLayer <: Lux.AbstractLuxLayer end

Lux.initialparameters(::AbstractRNG, ::FlattenLayer) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::FlattenLayer)     = NamedTuple()
Lux.parameterlength(::FlattenLayer)                  = 0
Lux.statelength(::FlattenLayer)                      = 0

(m::FlattenLayer)(x, ps, st) = (reshape(x, :, size(x, ndims(x))), st)


# --- ResBlock with MaxPool for spatial reduction ---
#
# Main path:  Conv(s=1) → BN → relu → MaxPool(s=2) → Conv(s=1) → BN
# Skip path:  Conv(1×1, s=1) → MaxPool(s=2) → BN   (or identity)
# Merge:      h + s → relu

# All ResBlocks use stride=2 → ALL have skip layers.
# Keeping skip path inside the same struct avoids type-instability
# from Union{} types in state/parameter NamedTuples.

struct ResBlock <: Lux.AbstractLuxLayer
    conv1;      bn1
    pool_main   # MaxPool(stride=2)
    conv2;      bn2
    skip_conv   # Conv(1×1, stride=1) for channel matching
    skip_pool   # MaxPool(stride=2) for spatial matching
    skip_bn     # BatchNorm
end

@functor ResBlock

function Lux.initialparameters(rng::AbstractRNG, b::ResBlock)
    return (conv1      = Lux.initialparameters(rng, b.conv1),
            bn1        = Lux.initialparameters(rng, b.bn1),
            pool_main  = Lux.initialparameters(rng, b.pool_main),
            conv2      = Lux.initialparameters(rng, b.conv2),
            bn2        = Lux.initialparameters(rng, b.bn2),
            skip_conv  = Lux.initialparameters(rng, b.skip_conv),
            skip_pool  = Lux.initialparameters(rng, b.skip_pool),
            skip_bn    = Lux.initialparameters(rng, b.skip_bn))
end

function Lux.initialstates(rng::AbstractRNG, b::ResBlock)
    return (conv1      = Lux.initialstates(rng, b.conv1),
            bn1        = Lux.initialstates(rng, b.bn1),
            pool_main  = Lux.initialstates(rng, b.pool_main),
            conv2      = Lux.initialstates(rng, b.conv2),
            bn2        = Lux.initialstates(rng, b.bn2),
            skip_conv  = Lux.initialstates(rng, b.skip_conv),
            skip_pool  = Lux.initialstates(rng, b.skip_pool),
            skip_bn    = Lux.initialstates(rng, b.skip_bn))
end

function ResBlock(ch_in::Int, ch_out::Int; stride::Int=1)
    # Main path: Conv(s=1) → BN → relu → MaxPool → Conv(s=1) → BN
    # Skip path: Conv(1×1, s=1) → MaxPool → BN
    # SamePad ensures MaxPool output matches stride-2 conv output
    # (e.g., ceil(375/2) = 188 vs floor(375/2) = 187)
    return ResBlock(
        Lux.Conv((7,), ch_in => ch_out; stride=1, pad=3),
        Lux.BatchNorm(ch_out),
        Lux.MaxPool((2,); stride=2, pad=Lux.SamePad()),
        Lux.Conv((7,), ch_out => ch_out; pad=3),
        Lux.BatchNorm(ch_out),
        Lux.Conv((1,), ch_in => ch_out; stride=1),
        Lux.MaxPool((2,); stride=2, pad=Lux.SamePad()),
        Lux.BatchNorm(ch_out),
    )
end

function (b::ResBlock)(x, ps, st)
    # Main path
    h, st_c1   = b.conv1(x,     ps.conv1,     st.conv1)
    h, st_b1   = b.bn1(h,       ps.bn1,       st.bn1)
    h           = NNlib.relu(h)
    h, st_pm   = b.pool_main(h, ps.pool_main, st.pool_main)
    h, st_c2   = b.conv2(h,     ps.conv2,     st.conv2)
    h, st_b2   = b.bn2(h,       ps.bn2,       st.bn2)

    # Skip path
    s, st_sc   = b.skip_conv(x, ps.skip_conv, st.skip_conv)
    s, st_sp   = b.skip_pool(s, ps.skip_pool, st.skip_pool)
    s, st_sb   = b.skip_bn(s,   ps.skip_bn,   st.skip_bn)

    # Merge
    out = NNlib.relu(h + s)

    st_new = (conv1 = st_c1, bn1 = st_b1, pool_main = st_pm,
              conv2 = st_c2, bn2 = st_b2,
              skip_conv = st_sc, skip_pool = st_sp, skip_bn = st_sb)
    return out, st_new
end


# --- Model builder ---
#
# Spatial progression:  12000 → 3000 → 1500 → 750 → 375 → 188 → 94
#                       3000 → 750  → 375 → 188 → 94  → 47  → 24  (for 3K input)
# Channel progression:   1 → 32 → 32 → 64 → 64 → 128 → 128
# Flatten:              128 × 94 = 12032  (12K input)
#                       128 × 24 = 3072   (3K input)
# Dense:                → 256 → n_outputs
#
# `n_outputs` is task-specific:
#   - 5 for the legacy functional-group multi-label head (smiles pipeline)
#   - N_PLASTIC (= 6) for the mutually-exclusive plastic classifier
# The model itself doesn't care — just the loss head and label format.

function build_model(spec_len::Int, n_outputs::Int; n_spatial::Int=94)
    model = Lux.Chain(
        SpecReshapeLayer(spec_len),

        # Initial stride-4 reduction: Conv(stride=1, pad=15) + MaxPool(stride=4)
        # SamePad ensures ceil(input/4) matches what a stride-4 conv would give
        Lux.Conv((31,), 1 => 32; stride=1, pad=15),
        Lux.MaxPool((4,); stride=4, pad=Lux.SamePad()),
        Lux.BatchNorm(32),
        Lux.WrappedFunction(relu),

        # 5 ResBlocks with skip connections
        ResBlock(32, 32;   stride=2),
        ResBlock(32, 64;   stride=2),
        ResBlock(64, 64;   stride=2),
        ResBlock(64, 128;  stride=2),
        ResBlock(128, 128; stride=2),

        FlattenLayer(),
        Lux.Dense(128 * n_spatial => 256, relu),
        Lux.Dropout(0.3f0),
        Lux.Dense(256 => n_outputs),
    )
    return model
end

# Backwards-compatible convenience builder: 12K input, multi-label FG
# (Kept so the contrastive/smiles pipeline can be re-enabled without rewriting)
function build_model_fg(spec_len::Int, n_fg::Int)
    return build_model(spec_len, n_fg; n_spatial=94)
end

# Plastic classifier builder: 3K input, mutually exclusive 6-class output
# Final dense has no activation; softmax is applied inside the loss.
function build_plastic_model(spec_len::Int=3000; n_classes::Int=N_PLASTIC, n_spatial::Int=24)
    return build_model(spec_len, n_classes; n_spatial=n_spatial)
end


# --- Utilities ---

"""
Count the number of scalar parameters in a Lux model's parameter tree.
Handles Arrays, NamedTuples, and Tuples recursively.
"""
function count_params(x)
    if x isa AbstractArray
        return length(x)
    elseif x isa NamedTuple
        return isempty(x) ? 0 : sum(count_params, values(x))
    elseif x isa Tuple
        return isempty(x) ? 0 : sum(count_params, x)
    else
        return 0
    end
end
