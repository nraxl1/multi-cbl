"""
train.py – Train & evaluate the FTIR plastic classifier.

Usage
-----
    python train.py --data_root ./data --epochs 80 --model cnn
    python train.py --data_root ./data --model cnn --lab_data ./lab-data --plot_title "Lab grid, per-spectrum normalisation"

Outputs (in --out_dir, default ./output/):
    model.pt                     best model weights
    scaler.pkl                   fitted StandardScaler
    class_names.json             ordered label list
    training_curves.png          loss / accuracy over epochs
    confusion_matrix_online.png  online test-set confusion matrix
    metrics_table_online.png     online test-set per-class metrics
    confusion_matrix_lab.png     lab test-set confusion matrix       (requires --lab_data)
    metrics_table_lab.png        lab test-set per-class metrics      (requires --lab_data)
    combined_confusion.png       1x2 combined confusion matrices     (requires --lab_data)
    combined_metrics.png         1x2 combined metrics tables         (requires --lab_data)
    combined_2x2.png             2x2 combined overview               (requires --lab_data)
"""

import argparse
import io
import json
import pickle
import re
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from dataset import COMMON_WN, build_dataloaders, parse_spectrum_file, resample_spectrum
from model import SpectraCNN, SpectraResNet


def parse_args():
    p = argparse.ArgumentParser(description="Train FTIR plastic classifier")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root dir containing train/ and test/ subdirectories "
        "(created by split_data.py)",
    )
    p.add_argument(
        "--out_dir", type=str, default="./output", help="Where to save model & plots"
    )
    p.add_argument(
        "--model", choices=["cnn", "resnet"], default="cnn", help="Architecture to use"
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--patience", type=int, default=12, help="Early-stopping patience (epochs)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lab_data",
        type=str,
        default=None,
        help="Path to flat lab data folder for optional post-training evaluation",
    )
    p.add_argument(
        "--plot_title",
        type=str,
        default="Model Evaluation",
        help="Grand title for the 2x2 combined overview plot",
    )
    return p.parse_args()


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


def evaluate_lab_data(lab_folder, model, scaler, class_names, device):
    lab_folder = Path(lab_folder)
    csv_files = sorted(lab_folder.glob("*.csv"))
    y_true, y_pred = [], []

    for fp in csv_files:
        match = re.match(r"^([A-Za-z]+)\d+", fp.stem)
        if not match:
            continue
        raw_label = match.group(1).upper()

        if raw_label in ("HDPE", "LDPE") and "PE" in class_names:
            label = "PE"
        elif raw_label in class_names:
            label = raw_label
        else:
            continue

        try:
            wn, tr = parse_spectrum_file(str(fp))
            if len(wn) < 100:
                continue
            resampled = resample_spectrum(wn, tr, COMMON_WN)
            scaled = scaler.transform(resampled.reshape(1, -1))
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.no_grad():
                pred_idx = model(tensor).argmax(1).item()
            y_true.append(class_names.index(label))
            y_pred.append(pred_idx)
        except Exception:
            continue

    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def plot_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out_path}")


def _confusion_fig(y_true, y_pred, class_names, title):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    overall_acc = (np.array(y_true) == np.array(y_pred)).mean()
    macro_f1 = report["macro avg"]["f1-score"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(
        f"Overall Accuracy: {overall_acc:.3f}  |  Macro F1: {macro_f1:.3f}",
        fontsize=10,
        pad=8,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _metrics_fig(y_true, y_pred, class_names, title):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    col_labels = ["Class", "Accuracy", "F1 Score"]
    rows = [
        [cls, f"{report[cls]['recall']:.3f}", f"{report[cls]['f1-score']:.3f}"]
        for cls in class_names
    ]

    fig, ax = plt.subplots(figsize=(5, len(class_names) * 0.42 + 0.9))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.7)

    cmap = plt.cm.Blues
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2166AC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, start=1):
        tbl[i, 0].set_facecolor("#DEEBF7")
        for j in (1, 2):
            val = float(row[j])
            tbl[i, j].set_facecolor(cmap(0.15 + val * 0.65))

    fig.tight_layout(pad=0.3)
    return fig


def _fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    plt.close(fig)
    return img


def plot_confusion(y_true, y_pred, class_names, out_path, title="Confusion Matrix"):
    fig = _confusion_fig(y_true, y_pred, class_names, title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix → {out_path}")


def plot_metrics_table(
    y_true, y_pred, class_names, out_path, title="Performance Metrics"
):
    fig = _metrics_fig(y_true, y_pred, class_names, title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics table → {out_path}")


def plot_combined_confusion(
    online_true, online_pred, lab_true, lab_pred, class_names, out_path
):
    img_online = _fig_to_image(
        _confusion_fig(online_true, online_pred, class_names, "Online Spectra Test Set")
    )
    img_lab = _fig_to_image(
        _confusion_fig(lab_true, lab_pred, class_names, "Lab Spectra")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    for ax, img in zip(axes, [img_online, img_lab]):
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined confusion matrices → {out_path}")


def plot_combined_metrics(
    online_true, online_pred, lab_true, lab_pred, class_names, out_path
):
    img_online = _fig_to_image(
        _metrics_fig(online_true, online_pred, class_names, "Online Spectra Test Set")
    )
    img_lab = _fig_to_image(
        _metrics_fig(lab_true, lab_pred, class_names, "Lab Spectra")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(class_names) * 0.5 + 2)))
    fig.suptitle("Performance Metrics", fontsize=13, fontweight="bold")
    for ax, img in zip(axes, [img_online, img_lab]):
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined metrics tables → {out_path}")


def _draw_metrics_in_ax(ax, y_true, y_pred, class_names):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    col_labels = ["Class", "Accuracy", "F1 Score"]
    rows = [
        [cls, f"{report[cls]['recall']:.3f}", f"{report[cls]['f1-score']:.3f}"]
        for cls in class_names
    ]
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        bbox=[0.01, 0.06, 0.98, 0.88],
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2166AC")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor("#DEEBF7")


def plot_combined_2x2(
    online_true, online_pred, lab_true, lab_pred, class_names, out_path, title
):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.45, 1.0],
        hspace=0.22,
        wspace=0.25,
        left=0.09,
        right=0.94,
        top=0.91,
        bottom=0.04,
    )

    ax_cm_online = fig.add_subplot(gs[0, 0])
    ax_cm_lab = fig.add_subplot(gs[0, 1])
    ax_mt_online = fig.add_subplot(gs[1, 0])
    ax_mt_lab = fig.add_subplot(gs[1, 1])

    online_report = classification_report(
        online_true, online_pred, target_names=class_names, output_dict=True
    )
    lab_report = classification_report(
        lab_true, lab_pred, target_names=class_names, output_dict=True
    )
    online_acc = (np.array(online_true) == np.array(online_pred)).mean()
    lab_acc = (np.array(lab_true) == np.array(lab_pred)).mean()
    online_f1 = online_report["macro avg"]["f1-score"]
    lab_f1 = lab_report["macro avg"]["f1-score"]

    ConfusionMatrixDisplay(
        confusion_matrix(online_true, online_pred), display_labels=class_names
    ).plot(ax=ax_cm_online, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax_cm_online.set_title(
        f"Online Spectra Test Set\nOverall Accuracy: {online_acc:.3f}  |  Macro F1: {online_f1:.3f}",
        fontsize=9,
        fontweight="bold",
        pad=6,
    )

    ConfusionMatrixDisplay(
        confusion_matrix(lab_true, lab_pred), display_labels=class_names
    ).plot(ax=ax_cm_lab, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax_cm_lab.set_title(
        f"Lab Spectra Test\nOverall Accuracy: {lab_acc:.3f}  |  Macro F1: {lab_f1:.3f}",
        fontsize=9,
        fontweight="bold",
        pad=6,
    )

    _draw_metrics_in_ax(ax_mt_online, online_true, online_pred, class_names)
    _draw_metrics_in_ax(ax_mt_lab, lab_true, lab_pred, class_names)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2x2 combined plot → {out_path}")


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_root = Path(args.data_root) / "train"
    test_root = Path(args.data_root) / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        print(f"ERROR: Expected train/ and test/ inside {args.data_root}")
        print(f"       Run split_data.py first to create them.")
        return

    train_dl, val_dl, test_dl, class_names, scaler = build_dataloaders(
        str(train_root),
        str(test_root),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    n_classes = len(class_names)

    with open(out / "class_names.json", "w") as f:
        json.dump(class_names, f)
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

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

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(
        f"\n{'Epoch':>5} | {'TrLoss':>7} {'TrAcc':>7} | {'VaLoss':>7} {'VaAcc':>7} | {'LR':>9}"
    )
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
        print(
            f"{epoch:5d} | {tr_loss:7.4f} {tr_acc:7.3%} | {va_loss:7.4f} {va_acc:7.3%} | {lr:9.2e}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} (best val acc: {best_val_acc:.3%})"
                )
                break

    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.0f}s  |  Best val acc: {best_val_acc:.3%}")

    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    te_loss, te_acc, y_pred, y_true = evaluate(model, test_dl, criterion, device)

    print(f"\n{'=' * 50}")
    print(f"TEST  Loss: {te_loss:.4f}  Accuracy: {te_acc:.3%}")
    print(f"{'=' * 50}\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_curves(history, out / "training_curves.png")
    plot_confusion(
        y_true,
        y_pred,
        class_names,
        out / "confusion_matrix_online.png",
        title="Online Spectra Test Set Confusion Matrix",
    )
    plot_metrics_table(
        y_true,
        y_pred,
        class_names,
        out / "metrics_table_online.png",
        title="Online Spectra Test Set Performance Metrics",
    )

    if args.lab_data:
        lab_true, lab_pred = evaluate_lab_data(
            args.lab_data,
            model,
            scaler,
            class_names,
            device,
        )
        lab_acc = (lab_true == lab_pred).mean()
        print(f"\nLab evaluation — {len(lab_true)} samples  Accuracy: {lab_acc:.3%}")
        print(classification_report(lab_true, lab_pred, target_names=class_names))

        plot_confusion(
            lab_true,
            lab_pred,
            class_names,
            out / "confusion_matrix_lab.png",
            title="Lab Spectra Confusion Matrix",
        )
        plot_metrics_table(
            lab_true,
            lab_pred,
            class_names,
            out / "metrics_table_lab.png",
            title="Lab Spectra Performance Metrics",
        )
        plot_combined_confusion(
            y_true,
            y_pred,
            lab_true,
            lab_pred,
            class_names,
            out / "combined_confusion.png",
        )
        plot_combined_metrics(
            y_true,
            y_pred,
            lab_true,
            lab_pred,
            class_names,
            out / "combined_metrics.png",
        )
        plot_combined_2x2(
            y_true,
            y_pred,
            lab_true,
            lab_pred,
            class_names,
            out / "combined_2x2.png",
            title=args.plot_title,
        )

    print(f"\nAll artifacts saved to {out.resolve()}/")


if __name__ == "__main__":
    main()
