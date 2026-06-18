"""
predict.py – Classify unknown FTIR spectra using a trained model.

Usage
-----
    # Single file  → full probability breakdown
    python predict.py --model_dir ./output --input unknown_sample.csv

    # Flat folder of unknowns  → per-file table + distribution summary
    python predict.py --model_dir ./output --input ./unknown_samples/

    # Labeled folder (subfolders = ground truth)  → adds confusion matrix,
    # classification report, and misclassification details
    python predict.py --model_dir ./output --input ./data/test/

    # Save results CSV and plots to a directory
    python predict.py --model_dir ./output --input ./data/test/ --save_csv results.csv --plot_dir ./plots
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import parse_spectrum_file, resample_spectrum, COMMON_WN, folder_to_class
from model import SpectraCNN, SpectraResNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_from_filename(filename: str) -> str | None:
   
    match = re.match(r"^([A-Za-z]+)\d+", Path(filename).stem)
    return match.group(1).upper() if match else None

"""

def label_from_filename(filename: str) -> str | None:
    match = re.match(r"^([A-Za-z]+)\d+", Path(filename).stem)
    if not match:
        return None
    label = match.group(1).upper()
    if label in ("HDPE", "LDPE"):
        return "PE"
    return label

"""

def transmittance_to_absorbance(spectrum: np.ndarray) -> np.ndarray:
    t = np.clip(spectrum, 0.01, 100.0).astype(np.float64)
    return (-np.log10(t / 100.0)).astype(np.float32)

def load_model(model_dir: str, architecture: str = "cnn"):
    model_dir = Path(model_dir)

    with open(model_dir / "class_names.json") as f:
        class_names = json.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    n_classes = len(class_names)
    seq_len = len(COMMON_WN)

    if architecture == "resnet":
        model = SpectraResNet(n_classes=n_classes, seq_len=seq_len)
    else:
        model = SpectraCNN(n_classes=n_classes, seq_len=seq_len)

    state = torch.load(model_dir / "model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, scaler, class_names


def predict_file(filepath: str, model, scaler, class_names) -> dict:
    """
    Predict the plastic class for a single FTIR CSV file.

    Returns dict with keys: file, prediction, confidence, all_probabilities
    """
    wn, tr = parse_spectrum_file(filepath)
    if len(wn) < 100:
        return {"file": filepath, "error": "Too few data points"}

    resampled = resample_spectrum(wn, tr, COMMON_WN)
    # rmin, rmax = resampled.min(), resampled.max()
    # if rmax > rmin:
    #     resampled = (resampled - rmin) / (rmax - rmin)
    scaled = scaler.transform(resampled.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "file": str(filepath),
        "prediction": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {
            name: float(f"{p:.4f}") for name, p in zip(class_names, probs)
        },
    }


def predict_folder(folder: str, model, scaler, class_names,
                    save_csv: str | None = None,
                    plot_dir: str | None = None) -> list[dict]:
    """
    Predict every CSV in a folder (flat or with labeled subfolders).

    If the folder contains subfolders (e.g. HDPE_c4/, PET_c8/), they are
    treated as ground-truth labels and full evaluation metrics are produced:
    confusion matrix, classification report, and misclassification list.

    If the folder just contains CSVs (no subfolders), only predictions and
    a distribution summary are printed.
    """
    folder = Path(folder)

    # ── Detect labeled vs flat ───────────────────────────────────────
    subfolders = sorted([d for d in folder.iterdir() if d.is_dir()])
    has_labels = bool(subfolders) and any(
        list(sf.glob("*.csv")) for sf in subfolders
    )

    if has_labels:
        return _predict_labeled(folder, subfolders, model, scaler, class_names,
                                save_csv, plot_dir)
    else:
        return _predict_flat(folder, model, scaler, class_names, save_csv, plot_dir)


def _predict_flat(folder: Path, model, scaler, class_names,
                  save_csv: str | None, plot_dir: str | None = None) -> list[dict]:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return []

    # Check if filenames carry labels
    labels = [label_from_filename(fp.name) for fp in csv_files]
    has_labels = all(l is not None for l in labels)

    results = []
    errors = 0
    y_true, y_pred = [], []

    print(f"{'File':<45} {'True':<12} {'Predicted':<20} {'Conf':>8} {'':>3}")
    print("-" * 92)

    for fp, true_label in zip(csv_files, labels):
        result = predict_file(str(fp), model, scaler, class_names)
        result["true_label"] = true_label
        results.append(result)

        if "error" in result:
            print(f"{fp.name:<45} {true_label:<12} {'ERROR':<20}")
            errors += 1
        else:
            hit = "✓" if has_labels and result["prediction"] == true_label else "✗"
            print(f"{fp.name:<45} {true_label or '':<12} {result['prediction']:<20} "
                  f"{result['confidence']:>8.2%} {hit:>3}")
            if has_labels:
                y_true.append(true_label)
                y_pred.append(result["prediction"])

    # Accuracy summary
    if has_labels and y_true:
        n_correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = n_correct / len(y_true)
        print(f"\n{'='*92}")
        print(f"Overall accuracy: {n_correct}/{len(y_true)} = {acc:.2%}  ({errors} errors)")
        print(f"{'='*92}\n")

        present_labels = sorted(set(y_true) | set(y_pred))
        print(classification_report(y_true, y_pred, labels=present_labels, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=present_labels).plot(
            ax=ax, cmap="Blues", xticks_rotation=45)
        ax.set_title(f"Confusion matrix — accuracy {acc:.2%}")
        fig.tight_layout()
        cm_path = Path(plot_dir or ".") / "predict_confusion_matrix.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix → {cm_path}")
    else:
        _print_distribution(results, class_names, errors)

    if save_csv:
        _save_results_csv(results, class_names, save_csv, labeled=has_labels)

    return results


def _predict_labeled(folder: Path, subfolders: list[Path], model, scaler,
                     class_names, save_csv: str | None,
                     plot_dir: str | None) -> list[dict]:
    """Predict a labeled folder structure and produce evaluation metrics."""
    results = []
    errors = 0

    # Collect files with their ground-truth labels
    file_label_pairs: list[tuple[Path, str]] = []
    for sf in subfolders:
        true_class = folder_to_class(sf.name)
        for fp in sorted(sf.glob("*.csv")):
            file_label_pairs.append((fp, true_class))

    if not file_label_pairs:
        print(f"No CSV files found in subfolders of {folder}")
        return []

    print(f"Evaluating {len(file_label_pairs)} files across "
          f"{len(subfolders)} subfolders in {folder}/\n")
    print(f"{'File':<35} {'True':<15} {'Predicted':<15} {'Conf':>7} {'':>3}")
    print("-" * 78)

    y_true, y_pred = [], []

    for fp, true_class in file_label_pairs:
        result = predict_file(str(fp), model, scaler, class_names)
        result["true_label"] = true_class
        results.append(result)

        if "error" in result:
            print(f"{fp.name:<35} {true_class:<15} {'ERROR':<15} {'':>7} {'':>3}")
            errors += 1
        else:
            hit = "✓" if result["prediction"] == true_class else "✗"
            print(f"{fp.name:<35} {true_class:<15} {result['prediction']:<15} "
                  f"{result['confidence']:>7.2%} {hit:>3}")
            y_true.append(true_class)
            y_pred.append(result["prediction"])

    # ── Overall accuracy ─────────────────────────────────────────────
    n_correct = sum(t == p for t, p in zip(y_true, y_pred))
    n_total = len(y_true)
    acc = n_correct / n_total if n_total else 0

    print(f"\n{'='*78}")
    print(f"RESULTS  —  {n_total} evaluated, {errors} errors")
    print(f"Overall accuracy: {n_correct}/{n_total} = {acc:.2%}")
    print(f"{'='*78}\n")

    # ── Classification report ────────────────────────────────────────
    # Use the model's class_names for consistent label ordering
    present_labels = sorted(set(y_true) | set(y_pred))
    print(classification_report(y_true, y_pred, labels=present_labels,
                                zero_division=0))

    # ── Misclassifications ───────────────────────────────────────────
    misclassified = [r for r in results
                     if "error" not in r and r["prediction"] != r["true_label"]]
    if misclassified:
        print(f"\nMISCLASSIFIED FILES ({len(misclassified)}):")
        print(f"{'File':<35} {'True':<15} {'Predicted':<15} {'Conf':>7}")
        print("-" * 74)
        for r in sorted(misclassified, key=lambda x: x["confidence"]):
            fname = Path(r["file"]).name
            print(f"{fname:<35} {r['true_label']:<15} {r['prediction']:<15} "
                  f"{r['confidence']:>7.2%}")
        # Per-pair breakdown
        pair_counts = Counter((r["true_label"], r["prediction"]) for r in misclassified)
        print(f"\nConfusion pairs:")
        for (true, pred), count in pair_counts.most_common():
            print(f"  {true} → {pred}: {count}")
    else:
        print("\nNo misclassifications — perfect accuracy!")

    # ── Confusion matrix plot ────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=present_labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion matrix — accuracy {acc:.2%}")
    fig.tight_layout()

    cm_path = Path(plot_dir or ".") / "predict_confusion_matrix.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved confusion matrix → {cm_path}")

    # ── Per-class confidence distribution plot ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    conf_by_class = {}
    for r in results:
        if "error" not in r:
            conf_by_class.setdefault(r["true_label"], []).append(r["confidence"])

    labels_sorted = sorted(conf_by_class.keys())
    data = [conf_by_class[c] for c in labels_sorted]
    bp = ax.boxplot(data, labels=labels_sorted, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C9AFF")
        patch.set_alpha(0.6)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction confidence by true class")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    conf_path = Path(plot_dir or ".") / "predict_confidence_dist.png"
    fig.savefig(conf_path, dpi=150)
    plt.close(fig)
    print(f"Saved confidence distribution → {conf_path}")

    # ── CSV export ───────────────────────────────────────────────────
    if save_csv:
        _save_results_csv(results, class_names, save_csv, labeled=True)

    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_distribution(results, class_names, errors):
    successful = [r for r in results if "error" not in r]
    counts = Counter(r["prediction"] for r in successful)
    confidences = {}
    for r in successful:
        confidences.setdefault(r["prediction"], []).append(r["confidence"])

    print(f"\n{'='*77}")
    print(f"SUMMARY  —  {len(successful)} classified, {errors} errors\n")
    print(f"{'Class':<25} {'Count':>6} {'%':>7}   {'Mean conf.':>10} {'Min conf.':>10}")
    print("-" * 63)

    for cls in sorted(counts, key=counts.get, reverse=True):
        n = counts[cls]
        pct = n / len(successful)
        confs = confidences[cls]
        print(f"{cls:<25} {n:>6} {pct:>7.1%}   {np.mean(confs):>10.2%} {np.min(confs):>10.2%}")

    print("-" * 63)
    print(f"{'TOTAL':<25} {len(successful):>6}")


def _save_results_csv(results, class_names, save_csv, labeled=False):
    import csv as csv_mod
    save_path = Path(save_csv)
    with open(save_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        header = ["file"]
        if labeled:
            header.append("true_label")
        header += ["prediction", "confidence", "correct"] if labeled else ["prediction", "confidence"]
        header += class_names
        writer.writerow(header)
        for r in results:
            if "error" in r:
                row = [Path(r["file"]).name]
                if labeled:
                    row.append(r.get("true_label", ""))
                row += ["ERROR", ""]
                if labeled:
                    row.append("")
                row += [""] * len(class_names)
                writer.writerow(row)
            else:
                row = [Path(r["file"]).name]
                if labeled:
                    row.append(r.get("true_label", ""))
                row += [r["prediction"], f"{r['confidence']:.4f}"]
                if labeled:
                    row.append("1" if r["prediction"] == r.get("true_label") else "0")
                row += [f"{r['all_probabilities'].get(c, 0):.4f}" for c in class_names]
                writer.writerow(row)
    print(f"\nResults saved → {save_path}")


def main():
    p = argparse.ArgumentParser(description="Predict plastic type from FTIR spectra")
    p.add_argument("--model_dir", type=str, default="./output",
                    help="Directory containing model.pt, scaler.pkl, class_names.json")
    p.add_argument("--input", type=str, required=True,
                    help="Path to a single CSV or a directory of CSVs")
    p.add_argument("--model", choices=["cnn", "resnet"], default="cnn",
                    help="Architecture (must match training)")
    p.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to save results as CSV (folder mode)")
    p.add_argument("--plot_dir", type=str, default=None,
                    help="Directory to save plots (default: current dir)")
    args = p.parse_args()

    model, scaler, class_names = load_model(args.model_dir, args.model)
    print(f"Loaded model ({args.model}) on {DEVICE} — classes: {class_names}\n")

    input_path = Path(args.input)

    # ── Folder mode ──────────────────────────────────────────────────
    if input_path.is_dir():
        predict_folder(str(input_path), model, scaler, class_names,
                        save_csv=args.save_csv, plot_dir=args.plot_dir)
        return

    # ── Single-file mode ─────────────────────────────────────────────
    result = predict_file(str(input_path), model, scaler, class_names)
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"File:       {input_path.name}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nFull probability breakdown:")
        for name, prob in sorted(
            result["all_probabilities"].items(),
            key=lambda x: x[1], reverse=True,
        ):
            bar = "█" * int(prob * 40)
            print(f"  {name:<20} {prob:>7.2%}  {bar}")


if __name__ == "__main__":
    main()
