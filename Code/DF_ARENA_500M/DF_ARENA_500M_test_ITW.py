import os
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import roc_curve

# Supprimer le warning HuggingFace
warnings.filterwarnings("ignore", message=".*sequentially.*")

DATASET_DIR = "/home/antton/data/release_in_the_wild/"
META_CSV    = os.path.join(DATASET_DIR, "meta.csv")

print("Chargement du modèle...")
pipe = pipeline(
    "antispoofing",
    model="Speech-Arena-2025/DF_Arena_500M_V_1",
    trust_remote_code=True,
    device='cuda'
)

df = pd.read_csv(META_CSV)
print(f"Total : {len(df)} fichiers")
print(df["label"].value_counts())

scores       = []
valid_labels = []
errors       = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Inférence"):
    filepath = os.path.join(DATASET_DIR, row["file"])

    if not os.path.exists(filepath):
        errors += 1
        continue

    try:
        audio, _ = librosa.load(filepath, sr=16000, mono=True)
        result   = pipe(audio)
        scores.append(result["all_scores"]["bonafide"])
        valid_labels.append(0 if row["label"] == "bona-fide" else 1)
    except Exception as e:
        errors += 1
        continue

print(f"\n{len(scores)} fichiers traités, {errors} erreurs")

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, [-s for s in scores], pos_label=1)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer * 100, -thresholds[eer_idx]

eer, threshold = compute_eer(valid_labels, scores)

scores_arr = np.array(scores)
labels_arr = np.array(valid_labels)
n_bonafide = valid_labels.count(0)
n_spoof    = valid_labels.count(1)

# Prédictions : bonafide score < threshold → spoof (1), sinon bonafide (0)
preds   = (scores_arr < threshold).astype(int)
correct = np.sum(preds == labels_arr)
total   = len(labels_arr)
acc     = correct / total * 100

fp  = np.sum((preds == 1) & (labels_arr == 0))
fn  = np.sum((preds == 0) & (labels_arr == 1))
fpr_thresh = fp / n_bonafide if n_bonafide > 0 else 0
fnr_thresh = fn / n_spoof    if n_spoof    > 0 else 0

print("\n" + "="*40)
print(f"Dataset    : In-the-Wild")
print(f"EER        : {eer:.2f}%")
print(f"Threshold  : {threshold:.4f}")
print(f"Accuracy   : {acc:.1f}%")
print(f"FPR @seuil : {fpr_thresh*100:.2f}%")
print(f"FNR @seuil : {fnr_thresh*100:.2f}%")
print(f"N bonafide : {n_bonafide}")
print(f"N spoof    : {n_spoof}")
print(f"N erreurs  : {errors}")
print("="*40)
print(f"EER attendu (paper) : 1.76%")

# Sauvegarde des scores pour le calcul de pooled EER
results_df = pd.DataFrame({
    "file":  df["file"].values[:len(scores)],
    "score": scores,
    "label": df["label"].values[:len(scores)]
})
results_df.to_csv(
    "/home/antton/projets_bachelor/model_DF_arena/DF_Arena_ITW_scores.txt",
    sep=" ", index=False, header=False
)
print("Scores sauvegardés : DF_Arena_ITW_scores.txt")

# ─────────────────────────────────────────────
# VISUALISATION — distribution des scores
# ─────────────────────────────────────────────
PLOT_OUT = "/home/antton/projets_bachelor/model_DF_arena/DF_Arena_ITW_scores_plot.png"

COLOR_BONAFIDE = "#2196F3"
COLOR_SPOOF    = "#F44336"
COLOR_THRESH   = "#FF9800"

bonafide_scores = scores_arr[labels_arr == 0]
spoof_scores    = scores_arr[labels_arr == 1]

fig, ax = plt.subplots(figsize=(12, 6))

bins = np.linspace(0, 1, 60)
ax.hist(bonafide_scores, bins=bins, color=COLOR_BONAFIDE, alpha=0.6,
        label="Bonafide (vrai)", density=True)
ax.hist(spoof_scores,    bins=bins, color=COLOR_SPOOF,    alpha=0.6,
        label="Spoof (vrai)",    density=True)

ax.axvline(threshold, color=COLOR_THRESH, linewidth=2.0, linestyle="--",
           label=f"Threshold ITW (EER) = {threshold:.4f}")

ymax = ax.get_ylim()[1]
ax.text(threshold + 0.01, ymax * 0.85,
        f"EER = {eer:.2f}%\n(FPR={fpr_thresh*100:.1f}% / FNR={fnr_thresh*100:.1f}%)",
        color=COLOR_THRESH, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_THRESH, alpha=0.8))

ax.text(threshold / 2, ymax * 0.95, "← prédit spoof",
        ha="center", va="top", fontsize=8, color="#555")
ax.text((threshold + 1) / 2, ymax * 0.95, "prédit bonafide →",
        ha="center", va="top", fontsize=8, color="#555")

patch_b = mpatches.Patch(color=COLOR_BONAFIDE, label="Bonafide (vrai)")
patch_s = mpatches.Patch(color=COLOR_SPOOF,    label="Spoof (vrai)")
ax.legend(handles=[patch_b, patch_s,
                   plt.Line2D([0], [0], color=COLOR_THRESH, linewidth=2, linestyle="--",
                              label=f"Threshold ITW = {threshold:.4f}")],
          fontsize=9, loc="upper left")

ax.set_xlabel("Score bonafide  →  0 = certain spoof  |  1 = certain bonafide", fontsize=10)
ax.set_ylabel("Densité", fontsize=10)
ax.set_xlim(0, 1)
ax.set_title(
    f"DF Arena 500M — Distribution des scores (In-the-Wild)\n"
    f"EER = {eer:.2f}%  (seuil {threshold:.4f})  |  "
    f"FPR={fpr_thresh*100:.1f}%  FNR={fnr_thresh*100:.1f}%  |  Accuracy = {acc:.1f}%",
    fontsize=11, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
print(f"\nGraphique sauvegardé : {PLOT_OUT}")
plt.show()