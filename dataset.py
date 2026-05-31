"""
dataset.py – Parse FTIR spectral CSV files and build PyTorch datasets.

Expected folder layout:
    data_root/
        HDPE_c4/
        HDPE_c8/       ← optional, merged with HDPE_c4 into class "HDPE"
        PET_c4/
        PET_c8/
        ...

Folders sharing the same base name (before _c4/_c8/etc.) are treated as
one class.  All spectra are resampled to a common wavenumber grid regardless
of their original resolution.

Each CSV has a metadata header followed by wavenumber,transmittance rows.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import pickle


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_spectrum_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a single FTIR CSV file.
    Returns (wavenumbers, transmittances) as 1-D numpy arrays,
    sorted by ascending wavenumber.
    """
    wavenumbers = []
    transmittances = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try to parse as a numeric data row (two comma-separated floats)
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    wn = float(parts[0])
                    tr = float(parts[1])
                    wavenumbers.append(wn)
                    transmittances.append(tr)
                    in_data = True
                    continue
                except ValueError:
                    pass

            # If we were already reading data and hit a non-numeric line, stop
            if in_data:
                break

    wn = np.array(wavenumbers, dtype=np.float32)
    tr = np.array(transmittances, dtype=np.float32)

    # Sort by ascending wavenumber for consistency
    order = np.argsort(wn)
    return wn[order], tr[order]


def resample_spectrum(
    wn: np.ndarray,
    tr: np.ndarray,
    target_wn: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a spectrum onto a common wavenumber grid."""
    return np.interp(target_wn, wn, tr).astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────

# Common wavenumber grid (399 → 4000 cm⁻¹, 3601 evenly spaced points)
COMMON_WN = np.linspace(400, 4000, 3601, dtype=np.float32)


def folder_to_class(folder_name: str) -> str:
    """Strip resolution suffix (_c4, _c8, etc.) to get the base class label."""
    return re.sub(r"_c\d+$", "", folder_name)


def load_all_spectra(data_root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk data_root, parse every CSV, resample to a common grid.
    Folders like HDPE_c4 and HDPE_c8 are merged into the same class (HDPE).

    Returns
    -------
    X : ndarray (N, len(COMMON_WN))   – transmittance spectra
    y : ndarray (N,)                   – integer labels
    class_names : list[str]            – ordered label names
    """
    data_root = Path(data_root)
    folders = sorted(
        [d for d in data_root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    # Group folders by base class name
    class_to_folders: dict[str, list[Path]] = {}
    for f in folders:
        cls = folder_to_class(f.name)
        class_to_folders.setdefault(cls, []).append(f)

    class_names = sorted(class_to_folders.keys())
    print(f"Found {len(class_names)} classes: {class_names}")
    for cls, dirs in sorted(class_to_folders.items()):
        print(f"  {cls} ← {[d.name for d in dirs]}")

    spectra, labels = [], []
    for label_idx, cls in enumerate(class_names):
        for folder in class_to_folders[cls]:
            csv_files = sorted(folder.glob("*.csv"))
            n_ok, n_fail = 0, 0
            for fp in csv_files:
                try:
                    wn, tr = parse_spectrum_file(str(fp))
                    if len(wn) < 100:
                        n_fail += 1
                        continue
                    resampled = resample_spectrum(wn, tr, COMMON_WN)
                    spectra.append(resampled)
                    labels.append(label_idx)
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
            print(f"    {folder.name}: {n_ok} loaded, {n_fail} skipped")

    X = np.stack(spectra, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names


class SpectraDataset(Dataset):
    """PyTorch dataset wrapping (X, y) numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, scaler=None, fit_scaler=False):
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None

        # Shape for 1-D CNN: (N, 1, seq_len)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(
    train_root: str,
    test_root: str,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], StandardScaler]:
    """
    Load physically separated train/test data, carve a validation set
    from train only, and return DataLoaders.
    """
    print("Loading TRAIN split...")
    X_trainval, y_trainval, class_names = load_all_spectra(train_root)
    print(f"  Total train+val samples: {X_trainval.shape[0]}")

    print("Loading TEST split...")
    X_test, y_test, test_classes = load_all_spectra(test_root)
    print(f"  Total test samples: {X_test.shape[0]}")

    assert class_names == test_classes, (
        f"Class mismatch between train and test folders!\n"
        f"  train: {class_names}\n  test: {test_classes}"
    )

    # Carve validation set from training data only
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio, stratify=y_trainval, random_state=seed,
    )
    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Fit scaler on training data only
    train_ds = SpectraDataset(X_train, y_train, fit_scaler=True)
    val_ds   = SpectraDataset(X_val,   y_val,   scaler=train_ds.scaler)
    test_ds  = SpectraDataset(X_test,  y_test,  scaler=train_ds.scaler)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, class_names, train_ds.scaler
