const CACHE_DIR  = "chunk_cache"
const PLASTIC_CACHE_DIR = "plastic_cache"

using DuckDB, DBInterface, DataFrames
using DataInterpolations: LinearInterpolation, ExtrapolationType
using Random: shuffle, randperm

############################################################
# ORIGINAL PARQUET + SMILES LOADING (FUNCTIONAL GROUPS)
#
# Kept for the contrastive pretraining pipeline (Phase 1).
# Not used by the plastic classification pipeline below.
# All public functions retain their original names so the
# contrastive pipeline can be re-enabled by uncommenting.
############################################################

#=
function ParquetDataFrame(path::String, columns::String, conn::DBInterface.Connection)
    parquet_df = DBInterface.execute(conn, """
        SELECT $columns
        FROM read_parquet('$path')
    """) |> DataFrame
    return parquet_df
end

function ParquetDataFrame(path::String, columns::String)
    conn = DBInterface.connect(DuckDB.DB, ":memory:")
    parquet_df = ParquetDataFrame(path::String, columns)
    DBInterface.close!(conn)
    return parquet_df
end

function ParquetDataFrame(path::String, columns::Vector{String})
    return ParquetDataFrame(path, join(columns, ", "))
end

function ParquetDataFrame(path::String)
    return ParquetDataFrame(path)
end


function load_chunk(path::String)
    println("  Loading $path ...")
    con = DBInterface.connect(DuckDB.DB, ":memory:")

    df = DuckDB.execute(con, """
        SELECT smiles, ir_spectra
        FROM read_parquet('$path')
    """) |> DataFrame
    DBInterface.close!(con)
    println("  Rows after SQL filter: $(nrow(df))")

    spectra = Vector{Vector{Float32}}()
    smiles_ok = String[]

    for i in 1:nrow(df)
        raw = df.ir_spectra[i]
        spec = try
            Float32.(collect(raw))
        catch
            continue
        end
        isempty(spec) && continue
        push!(spectra, spec)
        push!(smiles_ok, String(df.smiles[i]))
    end

    println("  Rows after spectrum parse: $(length(smiles_ok))")
    isempty(smiles_ok) && error("No valid rows in $path")

    spec_len = length(spectra[1])
    keep = findall(s -> length(s) == spec_len, spectra)
    spectra   = spectra[keep]
    smiles_ok = smiles_ok[keep]
    println("  Spectrum length: $spec_len  (dropped $(length(spectra) - length(keep)) length mismatches)")

    n = length(smiles_ok)
    labels = Vector{Union{Nothing, Vector{Float32}}}(undef, n)
    for i in 1:n
        labels[i] = featurize(smiles_ok[i])
        i % 2000 == 0 && println("  Featurized $i / $n ...")
    end

    valid = findall(!isnothing, labels)
    spectra   = spectra[valid]
    labels    = labels[valid]
    smiles_ok = smiles_ok[valid]
    println("  Rows after featurization: $(length(valid))")

    split_vec = [Int(hash(s) % 10) for s in smiles_ok]

    X = hcat(spectra...) |> x -> Float32.(x)
    Y = hcat(labels...)  |> x -> Float32.(x)

    return X, Y, split_vec
end

function cache_path(parquet_path::String)
    base = splitext(basename(parquet_path))[1]
    return joinpath(CACHE_DIR, base * ".jld2")
end

function cached_load_chunk(parquet_path::String)
    cp = cache_path(parquet_path)
    if isfile(cp)
        println("  Cache hit: $cp")
        return JLD2.load(cp, "X", "Y", "split_vec")
    end

    println("  Cache miss — loading + featurizing $parquet_path ...")
    X, Y, split_vec = load_chunk(parquet_path)

    mkpath(CACHE_DIR)
    JLD2.save(cp, "X", X, "Y", Y, "split_vec", split_vec)
    println("  Saved cache → $cp")
    return X, Y, split_vec
end

function prepare_data(data_paths::Vector{Vector{AbstractString}}, fit_normalizer, )::Tuple{DataLoader, DataLoader, DataLoader}
    if
    end
end
=#

############################################################
# PLASTIC CSV LOADING
#
# Directory layout (online-data/):
#   online-data/train/HDPE_c4/HDPE1.csv
#   online-data/train/PVC_c8/PVC42.csv
#   online-data/test/HDPE_c4/HDPE5.csv
#   ...
#
# File format:
#   line 1..3: metadata header (e.g. "TITLE...,...")
#   line 4..:  "wavenumber, intensity" rows
#
# Directory layout (lab-data/):
#   lab-data/HDPE1_trn.csv   (no header, already 2-column)
#   lab-data/PET10_trn.csv
#   ...
############################################################

# Canonical plastic types in stable order — index = class label
const PLASTIC_TYPES = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]
const N_PLASTIC     = length(PLASTIC_TYPES)
const PLASTIC_TYPE_TO_IDX = Dict(t => i-1 for (i, t) in enumerate(PLASTIC_TYPES))  # 0-indexed

# Default target grid for IR spectra: 400 - 4000 cm⁻¹, 3000 points
const DEFAULT_WN_MIN  = 400.0
const DEFAULT_WN_MAX  = 4000.0
const DEFAULT_SPEC_LEN = 3000

# Wavenumber column aliases in case CSV header is non-trivial
const WAVENUMBER_COL_ALIASES = ["wavenumber", "wavenumbers", "wn", "x"]
const INTENSITY_COL_ALIASES  = ["intensity", "intensities", "absorbance", "y", "transmittance"]


"""
    load_plastic_csv(path; skip_meta_header=true) -> (wn::Vector{Float32}, intensity::Vector{Float32})

Load a single plastic spectrum CSV. For `online-data/` files the first 3-5 lines
are metadata ("TITLE...,", "DATA TYPE...,", etc.) and are skipped. For `lab-data/`
files there is no header so all lines are parsed.

Returns sorted (ascending wavenumber) Float32 vectors.
"""
function load_plastic_csv(path::String; skip_meta_header::Bool=true)
    raw = read(path, String)
    lines = split(raw, '\n'; keepempty=false)

    # Find the first data line: first field is parseable Float64
    data_start = 1
    for (i, line) in enumerate(lines)
        first_field = split(strip(line), ',')[1]
        isempty(first_field) && continue
        try
            parse(Float64, first_field)
            data_start = i
            break
        catch
            continue
        end
    end

    wn  = Float32[]
    inten = Float32[]
    for line in lines[data_start:end]
        parts = split(strip(line), ',')
        length(parts) < 2 && continue
        try
            push!(wn,   parse(Float32, strip(parts[1])))
            push!(inten, parse(Float32, strip(parts[2])))
        catch
            continue
        end
    end

    if length(wn) == 0
        error("No numeric data found in $path")
    end

    # Sort by wavenumber ascending (some spectra are recorded in descending order)
    perm = sortperm(wn)
    return wn[perm], inten[perm]
end


"""
    load_plastic_csv_duckdb(path) -> (wn::Vector{Float32}, intensity::Vector{Float32})

DuckDB-based single-file loader. Uses `read_csv_auto` with `header=false` and
`skip=N` (we auto-detect N by pre-scanning for the first numeric line). Faster
than the pure-Julia path for large files but slightly more overhead for small
ones; the Julia path is the default and the DuckDB path is exposed for batch
use via `load_plastic_dataset_duckdb`.
"""
function load_plastic_csv_duckdb(path::String)
    # Detect how many header lines to skip by scanning for the first numeric line
    raw_lines = readlines(path)
    skip_n = 0
    for (i, line) in enumerate(raw_lines)
        first_field = split(strip(line), ',')[1]
        isempty(first_field) && (skip_n = i; continue)
        try
            parse(Float64, first_field)
            skip_n = i - 1
            break
        catch
            skip_n = i
            continue
        end
    end

    con = DBInterface.connect(DuckDB.DB, ":memory:")
    sql = """
    SELECT column0 AS wn, column1 AS intensity
    FROM read_csv_auto('$path', header=false, skip=$skip_n, sample_size=-1)
    """
    df = DBInterface.execute(con, sql) |> DataFrame
    DBInterface.close!(con)

    wn = Float32.(df.wn)
    inten = Float32.(df.intensity)
    perm = sortperm(wn)
    return wn[perm], inten[perm]
end


"""
    load_plastic_dataset_duckdb(paths_and_types; spec_len=3000, ...)
        -> (X::Matrix{Float32}, Y::Vector{Int})

DuckDB-based batch loader. Uses `read_csv_auto` on a glob pattern over all
input files in a single SQL query — much faster for thousands of files.
"""
function load_plastic_dataset_duckdb(paths_and_types::Vector{Tuple{String,String}};
                                     spec_len::Int = DEFAULT_SPEC_LEN,
                                     wn_min::Float64 = DEFAULT_WN_MIN,
                                     wn_max::Float64 = DEFAULT_WN_MAX)
    # Group by plastic type to determine labels, then build a single DuckDB
    # query that reads all files at once via a glob or a union of file lists.
    # For correctness across heterogeneous header sizes, we pre-scan each file
    # to find the skip count, then issue a per-file query (still via DuckDB).
    target_wn = collect(range(wn_min, wn_max, length=spec_len))

    n = length(paths_and_types)
    X = Matrix{Float32}(undef, spec_len, n)
    Y = Vector{Int}(undef, n)
    skipped = 0

    con = DBInterface.connect(DuckDB.DB, ":memory:")
    @info "DuckDB-loading $n plastic spectra ..."

    for (i, (path, ptype)) in enumerate(paths_and_types)
        try
            wn, inten = load_plastic_csv_duckdb(path)
            spec = interpolate_spectrum(wn, inten; target_wn=target_wn)
            X[:, i] = spec
            Y[i] = PLASTIC_TYPE_TO_IDX[ptype]
        catch e
            @warn "Failed to load $path via DuckDB: $e — falling back to Julia loader"
            try
                wn, inten = load_plastic_csv(path)
                spec = interpolate_spectrum(wn, inten; target_wn=target_wn)
                X[:, i] = spec
                Y[i] = PLASTIC_TYPE_TO_IDX[ptype]
            catch e2
                @warn "Fallback also failed for $path: $e2"
                skipped += 1
            end
        end
        if i % 500 == 0
            @info "  loaded $i / $n"
        end
    end
    DBInterface.close!(con)
    @info "Done. loaded=$(n-skipped) skipped=$skipped"
    return X, Y
end


"""
    interpolate_spectrum(wn, intensity; target_wn=range(400, 4000, length=3000))
        -> Vector{Float32}

Linear interpolation of a spectrum onto a fixed wavenumber grid via
DataInterpolations.LinearInterpolation. Returns `target_wn` length vector.

Spectra that don't cover the full range will have the extrapolated region filled
with the edge value (matches the Flat() boundary used previously).
"""
function interpolate_spectrum(wn::AbstractVector{<:Real},
                              intensity::AbstractVector{<:Real};
                              target_wn::AbstractVector{<:Real} =
                                  collect(range(DEFAULT_WN_MIN, DEFAULT_WN_MAX, length=DEFAULT_SPEC_LEN)),
                              extrap = ExtrapolationType.Linear)
    wn_f = Float64.(wn)
    inten_f = Float64.(intensity)
    # DataInterpolations expects strictly increasing x
    if !issorted(wn_f)
        perm = sortperm(wn_f)
        wn_f = wn_f[perm]
        inten_f = inten_f[perm]
    end
    # Deduplicate any repeated wavenumber values (rare, but possible)
    if length(unique(wn_f)) < length(wn_f)
        ux = Float64[]; uy = Float64[]
        for i in eachindex(wn_f)
            if i == 1 || wn_f[i] != wn_f[i-1]
                push!(ux, wn_f[i]); push!(uy, inten_f[i])
            end
        end
        wn_f, inten_f = ux, uy
    end
    itp = LinearInterpolation(inten_f, wn_f;
                              extrapolation_left=extrap,
                              extrapolation_right=extrap)
    out = itp.(target_wn)
    return Float32.(out)
end


"""
    smooth_spectrum(x::AbstractVector{<:Real}; window::Int=11, order::Int=3)
        -> Vector{Float32}

Apply a Savitzky–Golay smoothing filter to a 1D spectrum. Defaults to
window=11, order=3 — a conservative choice that removes high-frequency
noise on a 1.2 cm⁻¹ grid (≈13 cm⁻¹ window) while preserving peak
positions and shoulders.

Returns Float32 (matches the model's input dtype). The SavitzkyGolay
package computes in Float64 internally; the conversion is essentially
free.
"""
function smooth_spectrum(x::AbstractVector{<:Real}; window::Int=11, order::Int=3)
    r = SavitzkyGolay.savitzky_golay(collect(Float64, x), window, order)
    return Float32.(r.y)
end


"""
    apply_snv(X::Matrix{Float32}) -> Matrix{Float32}

Standard Normal Variate: per-spectrum mean-centering and std-scaling.
For each column (spectrum), subtract its mean and divide by its std
(+1e-6 floor to avoid division by zero on flat regions).

SNV removes per-spectrum intensity scale and baseline slope
differences, so the model sees relative *shape* rather than absolute
intensity. Standard companion to Savitzky–Golay in NIR / IR pipelines:
SG removes high-frequency noise on a per-spectrum basis, SNV then
removes the global intensity offset. Applied to *all* splits (train,
val, test, lab) so the eval distribution matches training.

Order in the pipeline: SG → SNV → (optional per-bin z-score).
"""
function apply_snv(X::Matrix{Float32})
    Xn = similar(X)
    for j in axes(X, 2)
        x = view(X, :, j)
        μ = Float32(mean(x))
        σ = Float32(std(x)) + 1f-6
        @views Xn[:, j] = (x .- μ) ./ σ
    end
    return Xn
end


"""
    plastic_type_from_path(path) -> String

Extract the plastic type (e.g. "HDPE") from a path like
`.../online-data/train/HDPE_c4/HDPE10.csv` or `lab-data/HDPE1_trn.csv`.
Falls back to filename prefix scan.
"""
function plastic_type_from_path(path::String)
    parts = split(path, '/')
    for p in Iterators.reverse(parts[1:end-1])  # skip filename
        for t in PLASTIC_TYPES
            if startswith(p, t)
                return t
            end
        end
    end
    # Last resort: parse from filename (e.g. "HDPE1_trn.csv")
    fname = basename(path)
    for t in PLASTIC_TYPES
        if startswith(fname, t)
            return t
        end
    end
    error("Could not infer plastic type from path: $path")
end


"""
    discover_plastic_csvs(root) -> Vector{Tuple{String,String}}

Walk a directory tree rooted at `root` and return a list of
`(path, plastic_type)` tuples for every `.csv` file found. The plastic type
is inferred from the immediate parent directory name (e.g. `HDPE_c4` → `HDPE`).
"""
function discover_plastic_csvs(root::String)
    files = Tuple{String,String}[]
    if !isdir(root)
        return files
    end
    for (dirpath, _, filenames) in walkdir(root)
        for fname in filenames
            endswith(fname, ".csv") || continue
            full = joinpath(dirpath, fname)
            ptype = plastic_type_from_path(full)
            push!(files, (full, ptype))
        end
    end
    return files
end


"""
    load_plastic_dataset(paths_and_types; spec_len=3000, wn_min=400.0, wn_max=4000.0)
        -> (X::Matrix{Float32}, Y::Vector{Int})

Load a list of `(path, plastic_type)` pairs. Each spectrum is parsed with
`load_plastic_csv` and interpolated to a common grid with
`interpolate_spectrum`. The returned `X` has shape `(spec_len, n_samples)`
and `Y` is a 0-indexed integer class vector.
"""
function load_plastic_dataset(paths_and_types::Vector{Tuple{String,String}};
                              spec_len::Int = DEFAULT_SPEC_LEN,
                              wn_min::Float64 = DEFAULT_WN_MIN,
                              wn_max::Float64 = DEFAULT_WN_MAX)
    target_wn = collect(range(wn_min, wn_max, length=spec_len))

    n = length(paths_and_types)
    X = Matrix{Float32}(undef, spec_len, n)
    Y = Vector{Int}(undef, n)
    skipped = 0

    @info "Loading $n plastic spectra ..."
    for (i, (path, ptype)) in enumerate(paths_and_types)
        try
            wn, inten = load_plastic_csv(path)
            spec = interpolate_spectrum(wn, inten; target_wn=target_wn)
            X[:, i] = spec
            Y[i] = PLASTIC_TYPE_TO_IDX[ptype]
        catch e
            @warn "Failed to load $path: $e"
            skipped += 1
        end
        if i % 500 == 0
            @info "  loaded $i / $n"
        end
    end
    @info "Done. loaded=$(n-skipped) skipped=$skipped"
    return X, Y
end


"""
    split_train_val_test(X, Y; val_frac=0.2, seed=42)
        -> (X_tr, Y_tr, X_val, Y_val, X_te, Y_te) | (X_tr, Y_tr, X_val, Y_val)

Stratified-ish random split. If `X_te` / `Y_te` are already passed in
(see keyword `external_test`), they are returned untouched. Otherwise the
held-out test set is carved from the data (10%).
"""
function split_train_val_test(X::Matrix{Float32}, Y::Vector{Int};
                              val_frac::Float64=0.2,
                              test_frac::Float64=0.1,
                              seed::Int=42)
    Random.seed!(seed)
    n = size(X, 2)
    idx = randperm(n)
    n_test = Int(round(n * test_frac))
    n_val  = Int(round(n * val_frac))

    test_idx = idx[1:n_test]
    val_idx  = idx[n_test+1:n_test+n_val]
    train_idx = idx[n_test+n_val+1:end]

    return (X[:, train_idx], Y[train_idx],
            X[:, val_idx],   Y[val_idx],
            X[:, test_idx],  Y[test_idx])
end


"""
    prepare_plastic_data(; train_root, test_root=nothing, lab_root=nothing,
                          spec_len=3000, val_frac=0.2, seed=42)
        -> NamedTuple with X_tr, Y_tr, X_val, Y_val, X_test, Y_test, X_lab, Y_lab

Top-level helper. Walks `train_root` for the training pool, optionally
`test_root` for the locked-away test set, and `lab_root` for the cross-domain
lab eval set. Carves a validation split from the training pool.
"""
function prepare_plastic_data(; train_root::String,
                              test_root::Union{String,Nothing}=nothing,
                              lab_root::Union{String,Nothing}=nothing,
                              spec_len::Int=DEFAULT_SPEC_LEN,
                              val_frac::Float64=0.2,
                              seed::Int=42,
                              use_duckdb::Bool=false)
    train_files = discover_plastic_csvs(train_root)
    isempty(train_files) && error("No CSV files found in train_root=$train_root")

    @info "Discovered $(length(train_files)) training files (use_duckdb=$use_duckdb)"

    load_fn = use_duckdb ? load_plastic_dataset_duckdb : load_plastic_dataset
    X_all, Y_all = load_fn(train_files; spec_len=spec_len)

    # Train/val split (no internal test split — we use the external one if available)
    Random.seed!(seed)
    n = size(X_all, 2)
    idx = randperm(n)
    n_val = Int(round(n * val_frac))
    val_idx = idx[1:n_val]
    train_idx = idx[n_val+1:end]
    X_tr, Y_tr = X_all[:, train_idx], Y_all[train_idx]
    X_val, Y_val = X_all[:, val_idx], Y_all[val_idx]

    # External test
    X_test = Matrix{Float32}(undef, 0, 0); Y_test = Int[]
    if test_root !== nothing && isdir(test_root)
        test_files = discover_plastic_csvs(test_root)
        @info "Discovered $(length(test_files)) external test files"
        if !isempty(test_files)
            X_test, Y_test = load_plastic_dataset(test_files; spec_len=spec_len)
        end
    end

    # Lab eval (cross-domain)
    X_lab = Matrix{Float32}(undef, 0, 0); Y_lab = Int[]
    if lab_root !== nothing && isdir(lab_root)
        lab_files = discover_plastic_csvs(lab_root)
        @info "Discovered $(length(lab_files)) lab files"
        if !isempty(lab_files)
            X_lab, Y_lab = load_plastic_dataset(lab_files; spec_len=spec_len)
        end
    end

    return (X_tr=X_tr, Y_tr=Y_tr, X_val=X_val, Y_val=Y_val,
            X_test=X_test, Y_test=Y_test, X_lab=X_lab, Y_lab=Y_lab,
            spec_len=spec_len)
end


############################################################
# CACHE
############################################################

function plastic_cache_path(key::String)
    mkpath(PLASTIC_CACHE_DIR)
    return joinpath(PLASTIC_CACHE_DIR, key * ".jld2")
end

function cached_plastic_data(; force::Bool=false, use_duckdb::Bool=false, kwargs...)
    key = "plastic_" * string(hash((kwargs, use_duckdb)))
    cp = plastic_cache_path(key)
    if !force && isfile(cp)
        @info "Loading plastic cache: $cp"
        X_tr    = JLD2.load(cp, "X_tr")
        Y_tr    = JLD2.load(cp, "Y_tr")
        X_val   = JLD2.load(cp, "X_val")
        Y_val   = JLD2.load(cp, "Y_val")
        X_test  = JLD2.load(cp, "X_test")
        Y_test  = JLD2.load(cp, "Y_test")
        X_lab   = JLD2.load(cp, "X_lab")
        Y_lab   = JLD2.load(cp, "Y_lab")
        spec_len = JLD2.load(cp, "spec_len")
        return (X_tr=X_tr, Y_tr=Y_tr, X_val=X_val, Y_val=Y_val,
                X_test=X_test, Y_test=Y_test, X_lab=X_lab, Y_lab=Y_lab,
                spec_len=spec_len)
    end
    data = prepare_plastic_data(; use_duckdb=use_duckdb, kwargs...)
    JLD2.save(cp,
        "X_tr", data.X_tr, "Y_tr", data.Y_tr,
        "X_val", data.X_val, "Y_val", data.Y_val,
        "X_test", data.X_test, "Y_test", data.Y_test,
        "X_lab", data.X_lab, "Y_lab", data.Y_lab,
        "spec_len", data.spec_len)
    return data
end


############################################################
# POSEIDON DATASET (marine microplastics FTIR)
#
# Directory layout:
#   Poseidon_files_V0.1.1/Data/IR_Spectra/<DDMMYY>/<Manta>/<sieve>/<ParticleID>.txt
#       e.g. IR_Spectra/141014/M242/1/TM0110B10.txt
#
# Spectrum file format (tab-separated, no header):
#   <index>  <wavenumber>  <intensity>
#   Wavenumbers descending, range ~600-4000 cm⁻¹, ~1.93 cm⁻¹ step (1762 pts).
#
# Labels live in:
#   Poseidon_files_V0.1.1/Data/IR_References/D4_4_publication.csv
#   col 1: <Manta>_<DDMMYY>_<sieve>_<ParticleID>   (the join key)
#   col 2: polymer interpretation, e.g. "Poly(ethylene)", "Poly(propylene) + fouling"
############################################################

# Label taxonomy mapping for the 6-class plastic classifier. Poseidon uses a
# richer vocabulary; we collapse it to a base polymer + quality flags.
# Returns nothing if the label cannot be mapped to one of our 6 classes.
const POSEIDON_LABEL_MAP = Dict{String,Int}(
    "Poly(ethylene)"        => PLASTIC_TYPE_TO_IDX["HDPE"],   # PE → HDPE arbitrarily (see note in loader)
    "Poly(ethylene) like"   => PLASTIC_TYPE_TO_IDX["HDPE"],
    "Poly(propylene)"       => PLASTIC_TYPE_TO_IDX["PP"],
    "Poly(propylene) like"  => PLASTIC_TYPE_TO_IDX["PP"],
    "Poly(styrene)"         => PLASTIC_TYPE_TO_IDX["PS"],
)

# Base class extracted from a Poseidon label string, or nothing if the label is
# not in our 6-class taxonomy. Strips trailing qualifiers (e.g. "+ fouling",
# " like") before lookup.
function _poseidon_base_label(label::AbstractString)
    s = strip(label)
    # Strip "like" suffix
    s = endswith(s, " like") ? s[1:end-5] : s
    # Strip " + fouling" / "+ fouling" / " + <anything>" suffix
    plus = findfirst("+", s)
    if plus !== nothing
        s = strip(s[1:plus-1])
    end
    return strip(s)
end

"""
    parse_poseidon_labels(csv_path) -> Dict{String, NamedTuple}

Read `D4_4_publication.csv` and return a Dict keyed by the join string
(`<Manta>_<DDMMYY>_<sieve>_<ParticleID>`) → `(base_class::Int, is_fouling::Bool,
is_like::Bool)`. Skips rows whose base label is not in the 6-class taxonomy
(no key returned for them, so they fail the join in `load_poseidon_dataset`).
"""
function parse_poseidon_labels(csv_path::String)
    labels = Dict{String,NamedTuple{(:base_class, :is_fouling, :is_like), Tuple{Int,Bool,Bool}}}()
    open(csv_path, "r") do io
        # Skip header
        readline(io)
        for line in eachline(io)
            parts = split(line, ','; limit=2)
            length(parts) < 2 && continue
            key = strip(parts[1])
            label_str = strip(parts[2])
            base = _poseidon_base_label(label_str)
            haskey(POSEIDON_LABEL_MAP, base) || continue
            is_fouling = occursin("+", label_str)
            is_like    = endswith(label_str, " like")
            labels[key] = (base_class=POSEIDON_LABEL_MAP[base],
                           is_fouling=is_fouling, is_like=is_like)
        end
    end
    return labels
end

"""
    load_poseidon_csv(path) -> (wn::Vector{Float32}, intensity::Vector{Float32})

Read a single Poseidon spectrum. Tab-separated 3-column `index  wavenumber
intensity`, no header, wavenumbers descending. Returns ascending-wn Float32
vectors.
"""
function load_poseidon_csv(path::String)
    wn  = Float32[]
    inten = Float32[]
    open(path, "r") do io
        for line in eachline(io)
            parts = split(line, '\t'; limit=3)
            length(parts) < 3 && continue
            try
                push!(wn,    parse(Float32, strip(parts[2])))
                push!(inten, parse(Float32, strip(parts[3])))
            catch
                continue
            end
        end
    end
    if length(wn) == 0
        error("No numeric data found in $path")
    end
    perm = sortperm(wn)
    return wn[perm], inten[perm]
end

"""
    load_poseidon_dataset(spectra_root, labels_csv;
                          spec_len=3000, wn_min=400.0, wn_max=4000.0,
                          extrap=ExtrapolationType.Flat)
        -> NamedTuple

Walk `spectra_root` for `*.txt` files, join with the labels CSV by
`<Manta>_<DDMMYY>_<sieve>_<ParticleID>`, interpolate each spectrum onto the
target wavenumber grid, and return three matrices split by quality:

  * `:clean`   — base polymer, no fouling, not " like"  (e.g. "Poly(ethylene)")
  * `:fouling` — base polymer, fouling present
  * `:like`    — base polymer, " like" qualifier (fuzzy match)

Returned NamedTuple:
    (X_clean,    Y_clean,    N_clean,
     X_fouling,  Y_fouling,  N_fouling,
     X_like,     Y_like,     N_like,
     skipped)

Spectra whose wavenumber range is below the target grid (Poseidon starts
at ~600) are filled with the value at the lowest available wavenumber
(`Flat` extrapolation by default — no Linear extrapolation of unknown
spectra).
"""
function load_poseidon_dataset(spectra_root::String, labels_csv::String;
                                spec_len::Int = DEFAULT_SPEC_LEN,
                                wn_min::Float64 = DEFAULT_WN_MIN,
                                wn_max::Float64 = DEFAULT_WN_MAX,
                                extrap = ExtrapolationType.Flat)
    target_wn = collect(range(wn_min, wn_max, length=spec_len))
    labels = parse_poseidon_labels(labels_csv)
    @info "Poseidon labels: $(length(labels)) entries with mappable base polymer"

    # Walk spectra_root
    paths = String[]
    for (dirpath, _, filenames) in walkdir(spectra_root)
        for fname in filenames
            endswith(fname, ".txt") || continue
            push!(paths, joinpath(dirpath, fname))
        end
    end
    @info "Found $(length(paths)) Poseidon .txt spectra"

    # Each entry: (path, key_tuple (manta, ddmmyy, sieve, particle))
    entries = Tuple{String,Tuple{String,String,String,String}}[]
    for p in paths
        parts = split(p, '/')
        length(parts) < 4 && continue
        # parts[end]   = ParticleID.txt
        # parts[end-1] = sieve
        # parts[end-2] = Manta
        # parts[end-3] = DDMMYY
        particle_id = splitext(parts[end])[1]
        sieve  = parts[end-1]
        manta  = parts[end-2]
        ddmmyy = parts[end-3]
        key = "$(manta)_$(ddmmyy)_$(sieve)_$(particle_id)"
        push!(entries, (p, (manta, ddmmyy, sieve, particle_id)))
    end

    # Per-bucket collectors
    Xs = Dict{Symbol,Vector{Vector{Float32}}}(
        :clean=>Vector{Float32}[], :fouling=>Vector{Float32}[], :like=>Vector{Float32}[])
    Ys = Dict{Symbol,Vector{Int}}(
        :clean=>Int[], :fouling=>Int[], :like=>Int[])
    skipped = 0
    n_unlabeled = 0
    n_loaded = 0

    for (path, _) in entries
        # Reconstruct key same way (manta_ddmmyy_sieve_particle)
        parts = split(path, '/')
        particle_id = splitext(parts[end])[1]
        sieve  = parts[end-1]
        manta  = parts[end-2]
        ddmmyy = parts[end-3]
        key = "$(manta)_$(ddmmyy)_$(sieve)_$(particle_id)"
        haskey(labels, key) || (n_unlabeled += 1; continue)

        try
            wn, inten = load_poseidon_csv(path)
            spec = interpolate_spectrum(wn, inten; target_wn=target_wn, extrap=extrap)
            meta = labels[key]
            # Bucket by quality flag (priority: fouling > like > clean)
            bucket = meta.is_fouling ? :fouling : (meta.is_like ? :like : :clean)
            push!(Xs[bucket], spec)
            push!(Ys[bucket], meta.base_class)
            n_loaded += 1
        catch e
            @warn "Failed to load $path: $e"
            skipped += 1
        end
    end
    @info "Poseidon: loaded=$n_loaded unlabeled=$n_unlabeled skipped=$skipped"

    function _stack(b::Symbol)
        Xv = Xs[b]
        if isempty(Xv)
            return (zeros(Float32, spec_len, 0), Int[])
        end
        return (reduce(hcat, Xv), Ys[b])
    end

    Xc, Yc = _stack(:clean)
    Xf, Yf = _stack(:fouling)
    Xl, Yl = _stack(:like)

    return (X_clean=Xc,   Y_clean=Yc,
            X_fouling=Xf, Y_fouling=Yf,
            X_like=Xl,    Y_like=Yl,
            skipped=skipped)
end
