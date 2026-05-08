############################################################
# MODEL
############################################################

const ARCH_VERSION = "hybrid-cnn-transformer-v1"

# ---- Type-stable custom layers ----

struct ReshapeLayer
    spec_len::Int
end
(m::ReshapeLayer)(x) = reshape(x, m.spec_len, 1, :)

# Reshape from CNN output (spatial, channels, batch)
# to Transformer input (channels, spatial, batch)
struct PermuteLayer end
(m::PermuteLayer)(x) = permutedims(x, (2, 1, 3))

# ---- ResBlock ----

struct ResBlock
    conv1
    bn1
    conv2
    bn2
    skip
end
Flux.@layer ResBlock

function ResBlock(ch_in::Int, ch_out::Int; stride::Int=1)
    conv1 = Conv((7,), ch_in => ch_out; stride=stride, pad=3)
    bn1   = BatchNorm(ch_out)
    conv2 = Conv((7,), ch_out => ch_out; pad=3)
    bn2   = BatchNorm(ch_out)
    if stride == 1 && ch_in == ch_out
        skip = identity
    else
        skip = Chain(
            Conv((1,), ch_in => ch_out; stride=stride),
            BatchNorm(ch_out),
        )
    end
    return ResBlock(conv1, bn1, conv2, bn2, skip)
end

function (b::ResBlock)(x)
    h = relu.(b.bn1(b.conv1(x)))
    h = b.bn2(b.conv2(h))
    return relu.(h .+ b.skip(x))
end

# ---- Transformer Encoder Block ----
# Operates on (d_model, seq_len, batch)
# MultiHeadAttention in Flux expects (d_model, seq_len, batch)

struct TransformerBlock
    attn
    norm1
    ff
    norm2
end
Flux.@layer TransformerBlock

function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int; dropout=0.1f0)
    attn  = MultiHeadAttention(d_model; nheads=n_heads, dropout_prob=dropout)
    norm1 = LayerNorm(d_model)
    ff    = Chain(
        Dense(d_model, d_ff, relu),
        Dropout(dropout),
        Dense(d_ff, d_model),
    )
    norm2 = LayerNorm(d_model)
    return TransformerBlock(attn, norm1, ff, norm2)
end

function (b::TransformerBlock)(x)
    # x shape: (d_model, seq_len, batch)
    # Self attention + residual
    attn_out, _ = b.attn(x, x, x)
    x = b.norm1(x .+ attn_out)
    # Feedforward + residual
    # LayerNorm and Dense operate on first dim so this works directly
    ff_out = b.ff(x)
    x = b.norm2(x .+ ff_out)
    return x
end

# ---- Full Model ----
function build_model(spec_len::Int, n_fg::Int)
    dev = MLDataDevices.gpu_device()

    backbone = Chain(
        ReshapeLayer(spec_len),
        Conv((31,), 1 => 32; stride=4, pad=15),
        BatchNorm(32), relu,
        ResBlock(32, 32;  stride=2),
        ResBlock(32, 64;  stride=2),
        ResBlock(64, 64;  stride=2),
        ResBlock(64, 128; stride=2),
        ResBlock(128, 128; stride=1),
        PermuteLayer(),
        TransformerBlock(128, 4, 256; dropout=0.1f0),
        TransformerBlock(128, 4, 256; dropout=0.1f0),
        Flux.flatten,
    )

    dummy = zeros(Float32, spec_len, 1)
    flat_dim = size(Flux.testmode!(backbone)(dummy), 1)
    @info "Inferred flat_dim: $flat_dim"

    Chain(
        backbone,
        Dense(flat_dim, 256, relu),
        Dropout(0.3f0),
        Dense(256, n_fg),
    ) |> dev
end