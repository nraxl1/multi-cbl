#!/usr/bin/env julia
"""
Initialize.jl — One-time project setup.

Installs core dependencies and the correct GPU backend for your hardware.
After running this, launch the project with: julia --project=. Main.jl
"""

using Pkg

project_dir = dirname(@__FILE__)
Pkg.activate(project_dir)

println("🔧 Setting up MultiCBL project at: $project_dir")
println("Julia version: $(VERSION)")

# --- Core dependencies ---
println("\n📦 Installing core dependencies...")
Pkg.add([
    "DuckDB", "DBInterface", "DataFrames",
    "Flux", "MLDataDevices", "Statistics", "Random", "Printf",
    "PythonCall", "MLUtils", "JLD2",
    "CondaPkg"
])

# --- Optional GPU backend (auto-detect) ---
function detect_gpu_backend()
    if Sys.isapple() && Sys.ARCH === :aarch64
        return :metal
    end
    if !Sys.isapple()
        Sys.which("nvidia-smi") !== nothing && return :cuda
        for path in [
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/wsl/lib/libcuda.so",
        ]
            isfile(path) && return :cuda
        end
    end
    if Sys.iswindows()
        windir = get(ENV, "WINDIR", "C:\\Windows")
        isfile(joinpath(windir, "System32", "nvml.dll")) && return :cuda
    end
    return :none
end

gpu = detect_gpu_backend()
if gpu === :metal
    println("\n🍎 Apple Silicon detected — installing Metal.jl...")
    Pkg.add("Metal")
    
    using Metal
    
elseif gpu === :cuda
    println("\n🎮 NVIDIA GPU detected — installing CUDA.jl + cuDNN.jl...")
    Pkg.add("CUDA")
    Pkg.add("cuDNN")
else
    println("\n💻 No supported GPU detected — CPU-only mode.")
    println("   Install CUDA.jl or Metal.jl later with: ] add CUDA  or  ] add Metal")
end

# --- Python dependency ---
using CondaPkg
CondaPkg.add("rdkit")

Pkg.instantiate()
Pkg.precompile()

println("\n" * "="^60)
println("✅ Setup complete! Run with: julia --project=. Main.jl")
println("="^60)
