############################################################
# PARQUET DATASET VALIDATOR (MD5, ORDERED, STREAMING SAFE)
# Matches hashes.txt line-by-line with chunk files
############################################################
import Pkg; Pkg.add("PythonCall")
using PythonCall

# Python hashlib for MD5 (Julia stdlib does NOT include MD5)
hashlib = pyimport("hashlib")

############################################################
# CONFIG
############################################################

DATA_DIR = "parquet-files/data"
HASH_FILE = "parquet-files/hashes.txt"

############################################################
# 1. MD5 HASH FUNCTION (STREAMING)
############################################################

function hash_file_md5(path::String)

    md5 = hashlib.md5()

    open(path, "r") do io
        buf = Vector{UInt8}(undef, 1024^2)  # 1MB buffer

        while !eof(io)
            n = readbytes!(io, buf)
            md5.update(view(buf, 1:n))
        end
    end

    return string(md5.hexdigest())
end

############################################################
# 2. LOAD EXPECTED HASHES (ORDERED)
############################################################

function load_hashes(path::String)
    hashes = String[]

    for line in eachline(path)
        h = strip(line)
        isempty(h) && continue
        push!(hashes, h)
    end

    return hashes
end

############################################################
# 3. GET PARQUET FILES (SAFE + ORDERED)
############################################################

function get_parquet_files(dir::String)

    files = readdir(dir)

    parquet_files = filter(f ->
        startswith(f, "IR_data_chunk") &&
        endswith(f, ".parquet")
    , files)

    sort!(parquet_files)

    return joinpath.(dir, parquet_files)
end

############################################################
# 4. VALIDATION LOGIC
############################################################

function validate(files, expected_hashes)

    println("\n🔍 PARQUET MD5 VALIDATION START")
    println("Files:  ", length(files))
    println("Hashes: ", length(expected_hashes))

    n = min(length(files), length(expected_hashes))

    mismatches = 0

    for i in 1:n

        println("\nChecking chunk ", i, "...")

        actual = hash_file_md5(files[i])
        expected = expected_hashes[i]

        if actual != expected
            println("❌ MISMATCH DETECTED")
            println("file:     ", files[i])
            println("expected: ", expected)
            println("actual:   ", actual)
            mismatches += 1
        else
            println("✔ OK")
        end
    end

    # check size mismatch
    if length(files) != length(expected_hashes)
        println("\n⚠ SIZE MISMATCH")
        println("files  = ", length(files))
        println("hashes = ", length(expected_hashes))
    end

    println("\n-------------------------")

    if mismatches == 0
        println("✅ ALL FILES VALID")
    else
        println("❌ TOTAL MISMATCHES: ", mismatches)
    end
end

############################################################
# 5. MAIN
############################################################

function main()

    println("Loading dataset...")

    files = get_parquet_files(DATA_DIR)
    hashes = load_hashes(HASH_FILE)

    validate(files, hashes)
end

main()
