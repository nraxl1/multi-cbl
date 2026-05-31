"""
train.py – Train & evaluate the FTIR plastic classifier.

Usage
-----
    python train.py --data_root ./data --epochs 80 --model cnn

Outputs (in --out_dir, default ./output/):
    model.pt              – best model weights
    scaler.pkl            – fitted StandardScaler
    class_names.json      – ordered label list
    training_curves.png   – loss / accuracy over epochs
    confusion_matrix.png  – test-set confusion matrix
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import build_dataloaders, COMMON_WN
from model import SpectraCNN, SpectraResNet


def parse_args():
    p = argparse.ArgumentParser(description="Train FTIR plastic classifier")
    p.add_argument("--data_root", type=str, required=True,
                    help="Root dir containing train/ and test/ subdirectories "
                         "(created by split_data.py)")
    p.add_argument("--out_dir", type=str, default="./output",
                    help="Where to save model & plots")
    p.add_argument("--model", choices=["cnn", "resnet"], default="cnn",
                    help="Architecture to use")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12,
                    help="Early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Training / evaluation loops ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, all_preds, all_labels


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out_path}")


def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_root = Path(args.data_root) / "train"
    test_root  = Path(args.data_root) / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        print(f"ERROR: Expected train/ and test/ inside {args.data_root}")
        print(f"       Run split_data.py first to create them.")
        return

    train_dl, val_dl, test_dl, class_names, scaler = build_dataloaders(
        str(train_root), str(test_root),
        batch_size=args.batch_size, seed=args.seed,
    )
    n_classes = len(class_names)

    # Save metadata
    with open(out / "class_names.json", "w") as f:
        json.dump(class_names, f)
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Model
    seq_len = len(COMMON_WN)
    if args.model == "resnet":
        model = SpectraResNet(n_classes=n_classes, seq_len=seq_len).to(device)
    else:
        model = SpectraCNN(n_classes=n_classes, seq_len=seq_len).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop with early stopping
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':>5} | {'TrLoss':>7} {'TrAcc':>7} | {'VaLoss':>7} {'VaAcc':>7} | {'LR':>9}")
    print("-" * 62)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {tr_loss:7.4f} {tr_acc:7.3%} | {va_loss:7.4f} {va_acc:7.3%} | {lr:9.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best val acc: {best_val_acc:.3%})")
                break

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.0f}s  |  Best val acc: {best_val_acc:.3%}")

    # ── Test evaluation ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    te_loss, te_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)

    print(f"\n{'='*50}")
    print(f"TEST  Loss: {te_loss:.4f}  Accuracy: {te_acc:.3%}")
    print(f"{'='*50}\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plots
    plot_curves(history, out / "training_curves.png")
    plot_confusion(y_true, y_pred, class_names, out / "confusion_matrix.png")

    print(f"\nAll artifacts saved to {out.resolve()}/")


if __name__ == "__main__":
    main()
