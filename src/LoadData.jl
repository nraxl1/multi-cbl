const CACHE_DIR  = "chunk_cache"

using DuckDB, DBInterface

############################################################
# LOAD ONE CHUNK
############################################################


"""
Open parquet file at ```path``` and return the provided columns as a Julia ```DataFrame```.
Returns all columns if ```columns``` is not provided. Uses a provided DuckDB connection if available. 


WARNING: ```columns``` is passed directly into an SQL query with **NO SANITIZATION**
"""
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

# find out how to make kwags work here and it will be so clean 
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

############################################################
# CACHE
############################################################

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
