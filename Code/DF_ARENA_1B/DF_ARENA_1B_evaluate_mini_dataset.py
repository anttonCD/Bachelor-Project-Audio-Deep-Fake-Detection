import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import pipeline
from sklearn.metrics import roc_curve
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = os.path.expanduser("~/projets_bachelor/dataSETS/mini_dataset_Antton/dataset_10_10")
BONAFIDE_DIR = os.path.join(BASE_DIR, "VM_bonafide")
SPOOF_DIR    = os.path.join(BASE_DIR, "VM_spoof")
SCORES_OUT   = os.path.expanduser("~/projets_bachelor/model_1B_DF_Arena/scores_1B_custom.txt")
DEVICE       = "cuda"

# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────
print("Chargement du modèle...")
pipe = pipeline(
    "antispoofing",
    model="Speech-Arena-2025/DF_Arena_1B_V_1",
    trust_remote_code=True,
    device=DEVICE
)

# ─────────────────────────────────────────────
# COLLECTE DES FICHIERS AUDIO
# ─────────────────────────────────────────────
def collect_files(bonafide_dir, spoof_dir):
    entries = []
    for folder, label_int, label_str in [
        (bonafide_dir, 0, "bonafide"),
        (spoof_dir,    1, "spoof"),
    ]:
        if not os.path.isdir(folder):
            print(f"⚠️  Dossier manquant : {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".wav", ".flac", ".mp3")):
                entries.append((os.path.join(folder, fname), label_int, label_str))
    return entries

entries = collect_files(BONAFIDE_DIR, SPOOF_DIR)
print(f"{len(entries)} fichiers trouvés "
      f"({sum(1 for _,l,_ in entries if l==1)} spoof, "
      f"{sum(1 for _,l,_ in entries if l==0)} bonafide)\n")

# ─────────────────────────────────────────────
# INFÉRENCE
# ─────────────────────────────────────────────
scores  = []
labels  = []
fnames  = []
errors  = []

with open(SCORES_OUT, "w") as f_out:
    for path, label_int, label_str in tqdm(entries, desc="Inférence"):
        try:
            audio, _ = librosa.load(path, sr=16000)
            result = pipe(audio)

            spoof_score = result["all_scores"]["spoof"]  # ∈ [0, 1]
            fname = os.path.basename(path)

            f_out.write(f"{fname} {label_str} {spoof_score:.6f}\n")

            scores.append(spoof_score)
            labels.append(label_int)
            fnames.append(fname)

        except Exception as e:
            errors.append((path, str(e)))
            print(f"\n  ✗ {os.path.basename(path)} : {e}")

# ─────────────────────────────────────────────
# CALCUL DE L'EER
# ─────────────────────────────────────────────
def compute_eer(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer * 100, thresholds[idx]

# ─────────────────────────────────────────────
# RÉSULTATS
# ─────────────────────────────────────────────
scores_arr = np.array(scores)
labels_arr = np.array(labels)
total      = len(labels)

eer, thresh = compute_eer(scores, labels)

# Prédictions au seuil EER : score >= thresh → spoof (1), sinon bonafide (0)
preds   = (scores_arr >= thresh).astype(int)
correct = np.sum(preds == labels_arr)
wrong   = total - correct
acc     = correct / total * 100

# HTER (Half Total Error Rate) at the ITW reference threshold
# NB: this is NOT an EER. The EER is the unique point where FPR = FNR;
# at a fixed threshold FPR != FNR in general, so (FPR + FNR) / 2 is the HTER.
THRESH_REF = 0.8013
preds_ref    = (scores_arr >= THRESH_REF).astype(int)
n_bonafide   = np.sum(labels_arr == 0)
n_spoof      = np.sum(labels_arr == 1)
fp_ref       = np.sum((preds_ref == 1) & (labels_arr == 0))
fn_ref       = np.sum((preds_ref == 0) & (labels_arr == 1))
fpr_ref      = fp_ref / n_bonafide if n_bonafide > 0 else 0
fnr_ref      = fn_ref / n_spoof    if n_spoof    > 0 else 0
hter_ref     = (fpr_ref + fnr_ref) / 2 * 100

wrong_files = [fnames[i] for i in range(total) if preds[i] != labels_arr[i]]

print("\n" + "=" * 40)
print(f"✓ CORRECTS  : {correct}/{total}")
print(f"✗ INCORRECTS: {wrong}/{total}")
print(f"Accuracy    : {acc:.1f}%")
print(f"EER         : {eer:.2f}%")
print(f"Threshold   : {thresh:.4f}")
print("=" * 40)

if wrong_files:
    print("Fichiers mal classifiés :")
    for f in wrong_files:
        print(f"  - {f}")

if errors:
    print(f"\n⚠️  {len(errors)} fichier(s) en erreur lors de l'inférence :")
    for path, msg in errors:
        print(f"  - {os.path.basename(path)} : {msg}")

# ─────────────────────────────────────────────
# VISUALIZATION — per-file score
# ─────────────────────────────────────────────
PLOT_OUT = os.path.expanduser("~/projets_bachelor/model_1B_DF_Arena/scores_1B_plot.png")

# Sort by ascending score for better readability
order = np.argsort(scores_arr)
sorted_scores  = scores_arr[order]
sorted_labels  = labels_arr[order]
sorted_fnames  = [fnames[i] for i in order]
sorted_preds   = (sorted_scores >= thresh).astype(int)
is_wrong       = sorted_preds != sorted_labels

COLOR_BONAFIDE = "#2196F3"   # blue
COLOR_SPOOF    = "#F44336"   # red
COLOR_THRESH   = "#FF9800"   # orange
COLOR_REF      = "#9C27B0"   # purple

n = len(sorted_scores)
y = np.arange(n)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, max(6, n * 0.45)))
fig.patch.set_facecolor("white")

for i in range(n):
    color  = COLOR_BONAFIDE if sorted_labels[i] == 0 else COLOR_SPOOF
    # horizontal stem from 0 to the score
    ax.hlines(i, 0, sorted_scores[i], color=color, linewidth=1.2, alpha=0.5)
    # main point
    marker = "D" if is_wrong[i] else "o"
    ax.scatter(sorted_scores[i], i, color=color,
               marker=marker, s=90, zorder=3,
               edgecolors="black" if is_wrong[i] else "none", linewidths=1.2)

# Threshold line
ax.axvline(thresh, color=COLOR_THRESH, linewidth=2.0, linestyle="--",
           label=f"EER threshold = {thresh:.4f}")

# Reference threshold with simulated EER annotation
ax.axvline(THRESH_REF, color=COLOR_REF, linewidth=1.8, linestyle=":",
           label=f"ITW reference threshold = {THRESH_REF}")
ax.text(THRESH_REF + 0.01, n * 0.5,
        f"HTER @ITW = {hter_ref:.2f}%\n(FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)",
        color=COLOR_REF, fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_REF, alpha=0.8))

# "Uncertainty" zone: ±0.05 around the threshold
ax.axvspan(max(0, thresh - 0.05), min(1, thresh + 0.05),
           alpha=0.08, color=COLOR_THRESH, label="Uncertainty zone (±0.05)")

# File labels on the Y axis
ax.set_yticks(y)
ax.set_yticklabels(sorted_fnames, fontsize=8)

# Manual legend
patch_b = mpatches.Patch(color=COLOR_BONAFIDE, label="Bonafide (true)")
patch_s = mpatches.Patch(color=COLOR_SPOOF,    label="Spoof (true)")
patch_w = plt.scatter([], [], marker="D", color="gray",
                      edgecolors="black", linewidths=1.2, label="Misclassified")
ax.legend(handles=[patch_b, patch_s, patch_w,
                   plt.Line2D([0], [0], color=COLOR_THRESH, linewidth=2, linestyle="--",
                              label=f"EER threshold = {thresh:.4f}"),
                   plt.Line2D([0], [0], color=COLOR_REF, linewidth=1.8, linestyle=":",
                              label=f"ITW reference threshold = {THRESH_REF}")],
          fontsize=9, loc="lower right", framealpha=0.9)

ax.set_xlabel("Spoof score  →  0 = certain bonafide  |  1 = certain spoof", fontsize=10)
ax.set_ylabel("Audio file", fontsize=10)
ax.set_xlim(0, 1)
ax.set_title(
    f"DF Arena 1B — Score per file\n"
    f"Optimal EER = {eer:.2f}%  (threshold {thresh:.4f})  |  "
    f"HTER @ITW threshold ({THRESH_REF}) = {hter_ref:.2f}%  |  Accuracy = {acc:.1f}%",
    fontsize=11, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

# Visual separator between the two regions (left = predicted bonafide, right = predicted spoof)
ax.text(thresh / 2, n - 0.5, "← predicted bonafide",
        ha="center", va="top", fontsize=8, color="#555")
ax.text((thresh + 1) / 2, n - 0.5, "predicted spoof →",
        ha="center", va="top", fontsize=8, color="#555")

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {PLOT_OUT}")
plt.show()

# ─────────────────────────────────────────────
# VISUALIZATION — confusion matrices
# Two matrices: one at the optimal EER threshold,
# one at the ITW reference threshold.
# Convention: positive class = spoof.
#   FAR (False Acceptance Rate) = spoof accepted as bonafide = FN / n_spoof
#   FRR (False Rejection Rate)  = bonafide rejected as spoof = FP / n_bonafide
# ─────────────────────────────────────────────
CM_OUT = os.path.expanduser("~/projets_bachelor/model_1B_DF_Arena/confusion_1B_plot.png")


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
    acc = (tn + tp) / (n_bona + n_spf) if (n_bona + n_spf) > 0 else 0.0

    # Matrix layout: rows = true class, cols = predicted class
    cm = np.array([[tn, fp],
                   [fn, tp]])
    # Color cells: correct (diagonal) in blue tones, errors (off-diagonal) in red tones
    color_grid = np.array([["#2196F3", "#F44336"],
                           ["#F44336", "#2196F3"]])
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
        f"Accuracy = {acc*100:.1f}%",
        fontsize=10, fontweight="bold", pad=14
    )


fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig_cm.patch.set_facecolor("white")

draw_confusion(axes[0], preds, labels_arr,
               f"Optimal EER threshold = {thresh:.4f}")
draw_confusion(axes[1], preds_ref, labels_arr,
               f"ITW reference threshold = {THRESH_REF}")

fig_cm.suptitle("DF Arena 1B — Confusion matrices (positive class = spoof)",
                fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
print(f"Confusion matrices saved: {CM_OUT}")
plt.show()

