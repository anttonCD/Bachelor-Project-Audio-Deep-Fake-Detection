import os
import glob
import warnings
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore", message=".*sequentially.*")

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

BASE_DIR      = os.path.expanduser("~/projets_bachelor/dataSETS/mini_dataset_Antton/dataset_50_50")
BONAFIDE_DIR  = os.path.join(BASE_DIR, "VM_bonafide")
SPOOF_DIR     = os.path.join(BASE_DIR, "VM_spoof")
THRESH_REF = 0.0014

# ──────────────────────────────────────────
# CHARGER LE MODÈLE
# ──────────────────────────────────────────

print("Chargement du modèle...")
pipe = pipeline(
    "antispoofing",
    model="Speech-Arena-2025/DF_Arena_500M_V_1",
    trust_remote_code=True,
    device='cuda'
)

# ──────────────────────────────────────────
# CONSTRUIRE LA LISTE DE FICHIERS + LABELS
# ──────────────────────────────────────────

# Cherche .wav et .flac dans les deux dossiers
bonafide_files = glob.glob(os.path.join(BONAFIDE_DIR, "*.wav")) + \
                 glob.glob(os.path.join(BONAFIDE_DIR, "*.flac"))
spoof_files    = glob.glob(os.path.join(SPOOF_DIR, "*.wav")) + \
                 glob.glob(os.path.join(SPOOF_DIR, "*.flac"))

# Associer chaque fichier à son label (0=bonafide, 1=spoof)
all_files  = [(f, 0) for f in bonafide_files] + \
             [(f, 1) for f in spoof_files]

print(f"Bonafide : {len(bonafide_files)} fichiers")
print(f"Spoof    : {len(spoof_files)} fichiers")
print(f"Total    : {len(all_files)} fichiers")

# ──────────────────────────────────────────
# INFÉRENCE
# ──────────────────────────────────────────

scores       = []
valid_labels = []
errors       = 0

for filepath, label in tqdm(all_files, desc="Inférence"):
    try:
        audio, _ = librosa.load(filepath, sr=16000, mono=True)
        result   = pipe(audio)
        scores.append(result["all_scores"]["bonafide"])
        valid_labels.append(label)
    except Exception as e:
        print(f"Erreur sur {filepath}: {e}")
        errors += 1
        continue

print(f"\n{len(scores)} fichiers traités, {errors} erreurs")

# ──────────────────────────────────────────
# CALCUL EER
# ──────────────────────────────────────────

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, [-s for s in scores], pos_label=1)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100, -thresholds[eer_idx]


eer, threshold = compute_eer(valid_labels, scores)

# HTER (Half Total Error Rate) at the ITW reference threshold
# (bonafide scores: score < THRESH_REF → predicted spoof)
# NB: this is NOT an EER. The EER is the unique point where FPR = FNR;
# at a fixed threshold FPR != FNR in general, so (FPR + FNR) / 2 is the HTER.
_scores_arr = np.array(scores)
_labels_arr = np.array(valid_labels)
_n_bonafide = np.sum(_labels_arr == 0)
_n_spoof    = np.sum(_labels_arr == 1)
_preds_ref  = (_scores_arr < THRESH_REF).astype(int)
_fp_ref     = np.sum((_preds_ref == 1) & (_labels_arr == 0))
_fn_ref     = np.sum((_preds_ref == 0) & (_labels_arr == 1))
fpr_ref     = _fp_ref / _n_bonafide if _n_bonafide > 0 else 0
fnr_ref     = _fn_ref / _n_spoof    if _n_spoof    > 0 else 0
hter_ref    = (fpr_ref + fnr_ref) / 2 * 100

print("\n" + "="*40)
print(f"Dataset    : Mini dataset 50/50")
print(f"EER        : {eer:.2f}%")
print(f"Threshold  : {threshold:.4f}")
print(f"HTER @ITW  : {hter_ref:.2f}%  (FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)")
print(f"Thresh ITW : {THRESH_REF}")
print(f"N bonafide : {valid_labels.count(0)}")
print(f"N spoof    : {valid_labels.count(1)}")
print(f"N erreurs  : {errors}")
print("="*40)
# ──────────────────────────────────────────
# DÉTAIL PAR FICHIER
# ──────────────────────────────────────────

print("\n" + "="*40)
print("DÉTAIL PAR FICHIER")
print("="*40)

corrects  = []
erreurs   = []

for (filepath, label), score in zip(all_files, scores):
    nom_fichier   = os.path.basename(filepath)
    dossier       = "bonafide" if label == 0 else "spoof"
    
    # Prédiction : score bonafide > threshold → prédit bonafide (0), sinon spoof (1)
    prediction    = 0 if score >= threshold else 1
    est_correct   = (prediction == label)
    
    pred_str = "bonafide" if prediction == 0 else "spoof"
    
    if est_correct:
        corrects.append(nom_fichier)
        statut = "✓"
    else:
        erreurs.append(nom_fichier)
        statut = "✗"
    
    print(f"{statut} [{dossier:8s}] → prédit: {pred_str:8s} | score: {score:.4f} | {nom_fichier}")

# ──────────────────────────────────────────
# RÉSUMÉ
# ──────────────────────────────────────────

print("\n" + "="*40)
print(f"✓ CORRECTS  : {len(corrects)}/{len(all_files)}")
print(f"✗ INCORRECTS: {len(erreurs)}/{len(all_files)}")
print(f"Accuracy    : {len(corrects)/len(all_files)*100:.1f}%")
print(f"EER         : {eer:.2f}%")
print(f"Threshold   : {threshold:.4f}")
print("="*40)

print("\nFichiers mal classifiés :")
for f in erreurs:
    print(f"  - {f}")

# ──────────────────────────────────────────
# EXPORT TXT
# ──────────────────────────────────────────

OUTPUT_TXT = os.path.join(os.path.dirname(__file__), "scores_par_fichier.txt")

with open(OUTPUT_TXT, "w") as f:
    f.write("="*70 + "\n")
    f.write("SCORES PAR FICHIER — DF_Arena_500M\n")
    f.write(f"EER threshold : {threshold:.4f}\n")
    f.write(f"EER           : {eer:.2f}%\n")
    f.write(f"HTER @ITW     : {hter_ref:.2f}%  (FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)\n")
    f.write(f"Thresh ITW    : {THRESH_REF}\n")
    f.write("="*70 + "\n\n")
    f.write(f"{'Statut':<4} {'Label réel':<12} {'Prédit':<12} {'Score':>8}  {'Fichier'}\n")
    f.write("-"*70 + "\n")
    for (filepath, label), score in zip(all_files, scores):
        nom          = os.path.basename(filepath)
        label_str    = "bonafide" if label == 0 else "spoof"
        prediction   = 0 if score >= threshold else 1
        pred_str     = "bonafide" if prediction == 0 else "spoof"
        statut       = "OK" if prediction == label else "ERR"
        f.write(f"{statut:<4} {label_str:<12} {pred_str:<12} {score:>8.4f}  {nom}\n")

print(f"\nScores exportés → {OUTPUT_TXT}")

# ─────────────────────────────────────────────
# VISUALIZATION — per-file score
# ─────────────────────────────────────────────
PLOT_OUT = os.path.join(os.path.dirname(__file__), "qualitative_scores.png")

scores_arr = np.array(scores)
labels_arr = np.array(valid_labels)
fnames     = [os.path.basename(fp) for fp, _ in all_files[:len(scores)]]
thresh     = threshold
acc        = len(corrects) / len(all_files) * 100

# Sort by ascending score for better readability
order = np.argsort(scores_arr)
sorted_scores  = scores_arr[order]
sorted_labels  = labels_arr[order]
sorted_fnames  = [fnames[i] for i in order]
sorted_preds   = (sorted_scores < thresh).astype(int)   # < thresh → spoof (1)
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

ax.axvline(THRESH_REF, color=COLOR_REF, linewidth=1.8, linestyle=":",
           label=f"ITW reference threshold = {THRESH_REF}")
ax.text(THRESH_REF + 0.01, n * 0.5,
        f"HTER @ITW = {hter_ref:.2f}%\n(FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)",
        color=COLOR_REF, fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_REF, alpha=0.8))

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

ax.set_xlabel("Bonafide score  →  (lower = more likely spoof)", fontsize=10)
ax.set_ylabel("Audio file", fontsize=10)

# Automatic zoom on the actual data range
data_min = sorted_scores.min()
data_max = sorted_scores.max()
data_span = data_max - data_min if data_max > data_min else thresh
margin = data_span * 0.15
x_lo = max(0, data_min - margin)
x_hi = data_max + margin
ax.set_xlim(x_lo, x_hi)

ax.set_title(
    f"DF Arena 500M — Score per file\n"
    f"Optimal EER = {eer:.2f}%  (threshold {thresh:.4f})  |  "
    f"HTER @ITW threshold ({THRESH_REF}) = {hter_ref:.2f}%  |  Accuracy = {acc:.1f}%",
    fontsize=11, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

# Uncertainty zone proportional to the visible range (±5% of the range)
hesi = data_span * 0.05
ax.axvspan(max(x_lo, thresh - hesi), min(x_hi, thresh + hesi),
           alpha=0.12, color=COLOR_THRESH, label="Uncertainty zone (±5% range)")

# Visual separator between the two regions
ax.text((x_lo + thresh) / 2, n - 0.5, "← predicted spoof",
        ha="center", va="top", fontsize=8, color="#555")
ax.text((thresh + x_hi) / 2, n - 0.5, "predicted bonafide →",
        ha="center", va="top", fontsize=8, color="#555")

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {PLOT_OUT}")
plt.close()

# ─────────────────────────────────────────────
# VISUALIZATION — confusion matrices
# Two matrices: one at the optimal EER threshold,
# one at the ITW reference threshold.
# Convention: positive class = spoof. Here predicted spoof <=> score < threshold.
#   FAR (False Acceptance Rate) = spoof accepted as bonafide = FN / n_spoof
#   FRR (False Rejection Rate)  = bonafide rejected as spoof = FP / n_bonafide
# ─────────────────────────────────────────────
CM_OUT = os.path.join(os.path.dirname(__file__), "confusion_matrices.png")

preds_opt = (scores_arr < thresh).astype(int)       # 1 = spoof
preds_itw = (scores_arr < THRESH_REF).astype(int)   # 1 = spoof


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

fig_cm.suptitle("DF Arena 500M — Confusion matrices (positive class = spoof)",
                fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
print(f"Confusion matrices saved: {CM_OUT}")
plt.close()