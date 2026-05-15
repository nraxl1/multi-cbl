############################################################
# MODEL — Lux.jl port
############################################################

# --- Identity layer for path routing (replaces Flux identity) ---

struct IdentityLayer end
(i::IdentityLayer)(x, ps, st) = (x, st)
Lux.@layer IdentityLayer

# --- Type-stable custom layers ---

"""
    ReshapeLayer(spec_len)

Replaces `x -> reshape(x, spec_len, 1, :)`. A named struct avoids
the boxing that happens when a closure captures `spec_len`.
"""
struct ReshapeLayer
    spec_len::Int
end
(m::ReshapeLayer)(x, ps, st) = (reshape(x, m.spec_len, 1, :), st)
Lux.@layer ReshapeLayer

"""
    GlobalMaxPool

Replaces `x -> dropdims(maximum(x; dims=1); dims=1)`.
"""
struct GlobalMaxPool end
(m::GlobalMaxPool)(x, ps, st) = (dropdims(maximum(x; dims=1); dims=1), st)
Lux.@layer GlobalMaxPool

# --- ResBlock ---

struct ResBlock
    conv1
    bn1
    conv2
    bn2
    skip
end

Lux.@layer ResBlock

function ResBlock(ch_in::Int, ch_out::Int; stride::Int=1)
    conv1 = Conv((7,), ch_in => ch_out; stride=stride, pad=3)
    bn1 = BatchNorm(ch_out)
    conv2 = Conv((7,), ch_out => ch_out; pad=3)
    bn2 = BatchNorm(ch_out)

    if stride == 1 && ch_in == ch_out
        skip = IdentityLayer()
    else
        skip = Chain(
            Conv((1,), ch_in => ch_out; stride=stride),
            BatchNorm(ch_out),
        )
    end

    return ResBlock(conv1, bn1, conv2, bn2, skip)
end

function (b::ResBlock)(x, ps, st)
    h, st_c1 = b.conv1(x, ps.conv1, st.conv1)
    h, st_b1 = b.bn1(h, ps.bn1, st.bn1)
    h = relu.(h)
    h, st_c2 = b.conv2(h, ps.conv2, st.conv2)
    h, st_b2 = b.bn2(h, ps.bn2, st.bn2)

    h_skip, st_skip = b.skip(x, ps.skip, st.skip)

    return relu.(h .+ h_skip), (conv1=st_c1, bn1=st_b1, conv2=st_c2, bn2=st_b2, skip=st_skip)
end

# --- Model builder ---
#
# Spatial dimension progression (spec_len = 12000):
#   Conv(stride=4)           → 3000
#   ResBlock(stride=2) × 5  → 1500 → 750 → 375 → 188 → 94
#   flattens to 128 × 94 = 12032

function build_model(spec_len::Int, n_fg::Int)
    model = Chain(
        ReshapeLayer(spec_len),
        # Stride 4 instead of 2 halves the first intermediate (98→49 MB at batch 128)
        # Kernel 31 maintains enough receptive field for IR band detection
        Conv((31,), 1 => 32; stride=4, pad=15),
        BatchNorm(32),
        relu,
        ResBlock(32, 32; stride=2),
        ResBlock(32, 64; stride=2),
        ResBlock(64, 64; stride=2),
        ResBlock(64, 128; stride=2),
        ResBlock(128, 128; stride=2),
        FlattenLayer(),
        Dense(128 * 94 => 256, relu),
        Dropout(0.3f0),
        Dense(256 => n_fg),
    )
    return model
end
