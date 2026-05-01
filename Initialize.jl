using Pkg


Pkg.activate(".")
Pkg.instantiate()

Pkg.add([
    "DuckDB", "DBInterface", "DataFrames",
    "Flux", "CUDA", "Statistics", "Random",
    "PythonCall", "cuDNN", "MLUtils", "JLD2",
    "CondaPkg"
])

using CondaPkg
CondaPkg.add("rdkit")


