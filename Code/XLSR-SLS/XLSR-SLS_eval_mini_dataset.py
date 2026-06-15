#!/usr/bin/env python
"""
Evaluate a dataset of audio files (bonafide + spoof) and export scores.

Usage: python eval_mini_dataset_60.py
"""

import os, sys, argparse
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
BONAFIDE_DIR = Path.home() / "projets_bachelor/dataSETS/mini_dataset_Antton/dataset_10_10/VM_bonafide"
SPOOF_DIR    = Path.home() / "projets_bachelor/dataSETS/mini_dataset_Antton/dataset_10_10/VM_spoof"
SCORES_OUT   = BASE_DIR / "scores/scores_mini60.txt"

TARGET_LEN   = 64600
THRESH_REF   = -11.520376   # ITW reference EER threshold (score >= THRESH_REF -> spoof)

# ── MODEL ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args   = argparse.Namespace(track="DF")
model  = Model(args, device)

state_dict = torch.load(str(BASE_DIR / "models/best_model.pth"), map_location=device)
new_state  = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def collect_files(folder):
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in (".wav", ".flac", ".mp3")
    )

def load_audio(path):
    waveform, sr = torchaudio.load(str(path))
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    x = waveform.mean(dim=0).numpy()
    if len(x) >= TARGET_LEN:
        return x[:TARGET_LEN]
    repeats = TARGET_LEN // len(x) + 1
    return np.tile(x, repeats)[:TARGET_LEN]

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    auc = np.trapz(tpr, fpr)
    return (fpr[idx] + fnr[idx]) / 2 * 100, thresholds[idx], auc

# ── DATASET ───────────────────────────────────────────────────────────────────
bonafide_files = collect_files(BONAFIDE_DIR)
spoof_files    = collect_files(SPOOF_DIR)

entries = (
    [(p, 0, "bonafide") for p in bonafide_files] +
    [(p, 1, "spoof")    for p in spoof_files]
)
print(f"{len(entries)} files — {len(bonafide_files)} bonafide, {len(spoof_files)} spoof")

# ── INFERENCE ─────────────────────────────────────────────────────────────────
scores, labels, fnames = [], [], []
SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(SCORES_OUT, "w") as f_out:
    for path, label_int, label_str in tqdm(entries, desc="Inference"):
        try:
            torch.cuda.empty_cache()
            x   = load_audio(path)
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)

            with torch.no_grad():
                out   = model(x_t)
                # log-softmax: col 0 = spoof score (higher = more spoofy)
                # training labels: spoof=0, bonafide=1 → col 0 = P(spoof)
                score = out[0, 0].item()

            del x_t, out
            torch.cuda.empty_cache()

            fname = path.name
            f_out.write(f"{fname} {score:.6f}\n")
            scores.append(score)
            labels.append(label_int)
            fnames.append(fname)

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"\nOOM on {path.name} — retrying on CPU")
                x_cpu = torch.FloatTensor(load_audio(path)).unsqueeze(0)
                with torch.no_grad():
                    out   = model.to("cpu")(x_cpu)
                    score = out[0, 0].item()
                model.to(device)
                fname = path.name
                f_out.write(f"{fname} {score:.6f}\n")
                scores.append(score)
                labels.append(label_int)
                fnames.append(fname)
            else:
                print(f"\nERROR {path.name}: {e}")
        except Exception as e:
            print(f"\nERROR {path.name}: {e}")

# ── EER & ACCURACY ────────────────────────────────────────────────────────────
scores_arr = np.array(scores)
labels_arr = np.array(labels)

eer, thresh, auc = compute_eer(scores_arr, labels_arr)
preds   = (scores_arr >= thresh).astype(int)
acc     = np.mean(preds == labels_arr) * 100

# HTER (Half Total Error Rate) at the ITW reference threshold
# NB: this is NOT an EER. The EER is the unique point where FPR = FNR;
# at a fixed threshold FPR != FNR in general, so (FPR + FNR) / 2 is the HTER.
n_bonafide = np.sum(labels_arr == 0)
n_spoof    = np.sum(labels_arr == 1)
preds_ref  = (scores_arr >= THRESH_REF).astype(int)
fp_ref     = np.sum((preds_ref == 1) & (labels_arr == 0))
fn_ref     = np.sum((preds_ref == 0) & (labels_arr == 1))
fpr_ref    = fp_ref / n_bonafide if n_bonafide > 0 else 0
fnr_ref    = fn_ref / n_spoof    if n_spoof    > 0 else 0
hter_ref   = (fpr_ref + fnr_ref) / 2 * 100

wrong_files = [fnames[i] for i in range(len(labels)) if preds[i] != labels_arr[i]]

print(f"\n{'='*40}")
print(f"Files evaluated : {len(scores)}/{len(entries)}")
print(f"Corrects        : {int(np.sum(preds == labels_arr))}/{len(scores)}")
print(f"Incorrects      : {int(np.sum(preds != labels_arr))}/{len(scores)}")
print(f"Accuracy        : {acc:.1f}%")
print(f"AUC             : {auc:.3f}")
print(f"EER             : {eer:.2f}%")
print(f"Threshold       : {thresh:.4f}")
print(f"HTER @ITW       : {hter_ref:.2f}%  (FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)")
print(f"Threshold ITW   : {THRESH_REF}")
print(f"Scores saved to : {SCORES_OUT}")
if eer > 50.0:
    print(f"\n⚠  EER > 50% — le modèle est pire que le hasard sur ce dataset.")
    print(f"   Ce n'est PAS un bug de code : AUC={auc:.3f} < 0.5 signifie que le modèle")
    print(f"   assigne un score de spoof plus ÉLEVÉ aux fichiers bonafide qu'aux spoof.")
    print(f"   Cause probable : distribution des données trop éloignée du domaine d'entraînement")
    print(f"   (ASVspoof 2019 LA, anglais) — TTS non-anglais ou conditions micro atypiques.")
print('='*40)

if wrong_files:
    print("Misclassified files:")
    for f in wrong_files:
        print(f"  - {f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
PLOT_OUT = BASE_DIR / "scores/scores_mini60_plot.png"

order         = np.argsort(scores_arr)
sorted_scores = scores_arr[order]
sorted_labels = labels_arr[order]
sorted_fnames = [fnames[i] for i in order]
sorted_preds  = (sorted_scores >= thresh).astype(int)
is_wrong      = sorted_preds != sorted_labels

COLOR_BONAFIDE = "#2196F3"
COLOR_SPOOF    = "#F44336"
COLOR_THRESH   = "#FF9800"
COLOR_REF      = "#9C27B0"

n   = len(sorted_scores)
s_min, s_max = sorted_scores.min(), sorted_scores.max()
# Make sure the ITW reference threshold is visible even if outside the score range
view_min = min(s_min, THRESH_REF)
view_max = max(s_max, THRESH_REF)
margin = (view_max - view_min) * 0.05   # uncertainty zone width = 5% of visible range

for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
    try:
        plt.style.use(_style)
        break
    except OSError:
        pass
fig, ax = plt.subplots(figsize=(11, max(6, n * 0.45)))
fig.patch.set_facecolor("white")

for i in range(n):
    color  = COLOR_BONAFIDE if sorted_labels[i] == 0 else COLOR_SPOOF
    ax.hlines(i, view_min - margin, sorted_scores[i], color=color, linewidth=1.2, alpha=0.5)
    ax.scatter(sorted_scores[i], i, color=color,
               marker="D" if is_wrong[i] else "o", s=90, zorder=3,
               edgecolors="black" if is_wrong[i] else "none", linewidths=1.2)

ax.axvline(thresh, color=COLOR_THRESH, linewidth=2.0, linestyle="--",
           label=f"EER threshold = {thresh:.4f}")
ax.axvspan(thresh - margin, thresh + margin, alpha=0.08, color=COLOR_THRESH,
           label="Uncertainty zone (±5%)")

# ITW reference threshold with HTER annotation
ax.axvline(THRESH_REF, color=COLOR_REF, linewidth=1.8, linestyle=":",
           label=f"ITW reference threshold = {THRESH_REF}")
ax.text(THRESH_REF + (view_max - view_min) * 0.01, n * 0.5,
        f"HTER @ITW = {hter_ref:.2f}%\n(FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)",
        color=COLOR_REF, fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_REF, alpha=0.8))

ax.set_yticks(np.arange(n))
ax.set_yticklabels(sorted_fnames, fontsize=8)
ax.set_xlim(view_min - margin, view_max + margin)
ax.set_xlabel("Spoof score  →  lower = certain bonafide  |  higher = certain spoof", fontsize=10)
ax.set_ylabel("Audio file", fontsize=10)
ax.set_title(
    f"XLS-R SLS — Score per file\n"
    f"EER = {eer:.2f}%  (threshold {thresh:.4f})  |  "
    f"HTER @ITW threshold ({THRESH_REF}) = {hter_ref:.2f}%  |  Accuracy = {acc:.1f}%",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

ax.text((view_min + thresh) / 2, n - 0.5, "← predicted bonafide",
        ha="center", va="top", fontsize=8, color="#555")
ax.text((thresh + view_max) / 2, n - 0.5, "predicted spoof →",
        ha="center", va="top", fontsize=8, color="#555")

ax.legend(handles=[
    mpatches.Patch(color=COLOR_BONAFIDE, label="Bonafide (true)"),
    mpatches.Patch(color=COLOR_SPOOF,    label="Spoof (true)"),
    plt.scatter([], [], marker="D", color="gray",
                edgecolors="black", linewidths=1.2, label="Misclassified"),
    plt.Line2D([0], [0], color=COLOR_THRESH, linewidth=2, linestyle="--",
               label=f"EER threshold = {thresh:.4f}"),
    plt.Line2D([0], [0], color=COLOR_REF, linewidth=1.8, linestyle=":",
               label=f"ITW reference threshold = {THRESH_REF}"),
], fontsize=9, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to : {PLOT_OUT}")
plt.show()

# ── CONFUSION MATRICES ──────────────────────────────────────────────────────────
# Two matrices: one at the optimal EER threshold, one at the ITW reference threshold.
# Convention: positive class = spoof. Predicted spoof <=> score >= threshold.
#   FAR (False Acceptance Rate) = spoof accepted as bonafide = FN / n_spoof
#   FRR (False Rejection Rate)  = bonafide rejected as spoof = FP / n_bonafide
CM_OUT = BASE_DIR / "scores/confusion_mini60_plot.png"

preds_opt = (scores_arr >= thresh).astype(int)       # 1 = spoof
preds_itw = (scores_arr >= THRESH_REF).astype(int)   # 1 = spoof


def confusion_counts(preds, labels):
    """Return (TN, FP, FN, TP) with positive class = spoof (1)."""
    tn = int(np.sum((preds == 0) & (labels == 0)))  # bonafide correctly accepted
    fp = int(np.sum((preds == 1) & (labels == 0)))  # bonafide rejected as spoof
    fn = int(np.sum((preds == 0) & (labels == 1)))  # spoof accepted as bonafide
    tp = int(np.sum((preds == 1) & (labels == 1)))  # spoof correctly rejected
    return tn, fp, fn, tp


def draw_confusion(ax, preds, labels, title):
    tn, fp, fn, tp = confusion_counts(preds, labels)
    n_bona = tn + fp
    n_spf  = fn + tp
    far = fn / n_spf  if n_spf  > 0 else 0.0   # spoof accepted  (false acceptance)
    frr = fp / n_bona if n_bona > 0 else 0.0   # bonafide rejected (false rejection)
    acc_cm = (tn + tp) / (n_bona + n_spf) if (n_bona + n_spf) > 0 else 0.0

    color_grid = [["#2196F3", "#F44336"],
                  ["#F44336", "#2196F3"]]
    cell_text  = [[f"TN\n{tn}", f"FP (FRR)\n{fp}"],
                  [f"FN (FAR)\n{fn}", f"TP\n{tp}"]]

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                       facecolor=color_grid[r][c], alpha=0.35,
                                       edgecolor="black", linewidth=1.2))
            ax.text(c + 0.5, r + 0.5, cell_text[r][c],
                    ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred. bonafide", "Pred. spoof"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["True bonafide", "True spoof"], fontsize=9, rotation=90, va="center")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        f"{title}\nFAR = {far*100:.1f}%   |   FRR = {frr*100:.1f}%   |   "
        f"Accuracy = {acc_cm*100:.1f}%",
        fontsize=10, fontweight="bold", pad=14
    )


fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig_cm.patch.set_facecolor("white")

draw_confusion(axes[0], preds_opt, labels_arr,
               f"Optimal EER threshold = {thresh:.4f}")
draw_confusion(axes[1], preds_itw, labels_arr,
               f"ITW reference threshold = {THRESH_REF}")

fig_cm.suptitle("XLS-R SLS — Confusion matrices (positive class = spoof)",
                fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
print(f"Confusion matrices saved to : {CM_OUT}")
plt.show()
