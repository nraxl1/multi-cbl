# Architecture Documentation

## Data Flow

```
Parquet Files → DuckDB Conversion → Feature Extraction → Normalization → Model Training → Evaluation
```

### 1. Data Loading (`src/parquet-files/`)
- Raw IR spectra stored as Parquet files
- Converted to DuckDB native format for performance
- Chunked by index (e.g., `IR_data_chunk001_of_009.parquet`)

### 2. Feature Extraction (`IRSpectraML/src/Featurization.jl`)
- RDKit chemical structure parsing
- Functional group detection via SMARTS patterns
- Feature vector generation (one-hot functional group presence)
- Min-max normalization

### 3. Model (`IRSpectraML/src/ModelNext.jl`)
- **Architecture**: ResCNN-v7 (Flux migration)
- **Layers**:
  - SpecReshapeLayer: Reshape spectrum to (spec_len, 1, 1)
  - ResBlock × 6: Conv(1) → BN → ReLU → MaxPool(2) → Conv(1) → BN
  - Skip connections with MaxPool
  - Flatten → Dense(12032→256) → Dense(256→5)
- **Output**: 5-class multi-label classification (5 functional groups)

### 4. Training (`IRSpectraML/src/TrainingNext.jl`)
- AdamW optimizer with cosine annealing
- Early stopping with patience
- GPU device management via Reactant

### 5. Evaluation (`Main.jl`)
- Batched inference (batch_size=16) to avoid VRAM OOM
- Per-label accuracy and F1 scores
- Macro-F1 aggregation

## Migration Notes

**Flux → Lux**: The main architectural change is replacing Flux layers with Lux equivalents. The ResBlock stride-2 MaxPool approach replaces Flux's strided Conv due to Reactant tracing limitations (issue #1990).

## Directory Structure Rationale

| File | Purpose |
|------|---------|
| `Main.jl` | Entry point — script-style notebook execution |
| `Initialize.jl` | One-time dependency setup |
| `DataTools.jl` | Shared data utilities and DuckDB management |
| `IRSpectraML/` | Package module for reusable ML components |
