# FTIR Plastic Classifier

Classifies FTIR infrared spectra into 6 plastic types using a 1D Convolutional Neural Network.

## Setup

```bash
C:\Users\ksili\Desktop\CBL\.venv\Scripts\activate # Change with your path obviously, but has to have the .venv\Scripts\activate
```

```bash
cd C:\Users\ksili\Desktop\CBL # change to your path
```

```bash
pip install -r requirements.txt
```

## Expected data layout

```
data/
├── HDPE_c4/
│   ├── sample_001.csv
│   ├── sample_002.csv
│   └── ...
├── PET_c4/
│   └── ...
└── ... (6 folders, one per plastic class)
```

Each CSV is an FTIR export with metadata headers followed by `wavenumber,transmittance` rows.

## Train

First split the data into train/test on disk (files are copied, originals stay intact):

```bash
python split_data.py --data_root ./online-data --test_ratio 0.15
```

This creates `data/train/` and `data/test/`, each mirroring the original class folder structure. The test set is never seen during training or validation.

Then train:

```bash
python train.py --data_root ./online-data --epochs 50 --model cnn # model will be generated in ./output
```

Key options:
- `--model cnn|resnet` — plain CNN (default) or residual variant
- `--lr 0.001` — learning rate
- `--patience 12` — early-stopping patience
- `--batch_size 64`

Outputs saved to `./output/`: model weights, scaler, class names, training curves, and confusion matrix.

## Predict

```bash
# Single file
python predict.py --input unknown_sample.csv

# Folder of unknowns
python predict.py --input ./unknown_samples/ --model cnn
```

## Architecture

The default CNN has three conv blocks (Conv1d → BatchNorm → ReLU → MaxPool → Dropout) followed by a two-layer classifier head. Input spectra are resampled onto a common 400–4000 cm⁻¹ grid (3601 points) and standardized using the training set statistics.

The `--model resnet` variant adds skip connections for slightly harder classification tasks.
