"""
Évaluation du modèle Codecfake (W2V2-AASIST) sur le dataset personnel.
Les labels sont déduits du dossier parent : VM_bonafide → bonafide, VM_spoof → spoof.
"""

import os
import sys
import glob
import time
import torch
import types
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix

# pytorch_model_summary n'est pas installé dans cet env mais model.py en a besoin
# à l'import — on injecte un stub vide pour éviter le ModuleNotFoundError.
if "pytorch_model_summary" not in sys.modules:
    stub = types.ModuleType("pytorch_model_summary")
    stub.summary = lambda *_, **__: None
    sys.modules["pytorch_model_summary"] = stub

# ============================================================
# Configuration
# ============================================================

CODECFAKE_REPO_PATH = "/home/antton/projets_bachelor/model_pre-entraine_codecfake/Codecfake"
MODEL_PATH          = os.path.join(CODECFAKE_REPO_PATH,
                          "/home/antton/projets_bachelor/model_pre-entraine_codecfake/Codecfake/pretrained_model/codec_w2v2aasist/anti-spoofing_feat_model.pt")
DATASET_ROOT        = "/home/antton/projets_bachelor/dataSETS/mini_dataset_Antton/dataset_10_10"
BONAFIDE_DIR        = os.path.join(DATASET_ROOT, "VM_bonafide")
SPOOF_DIR           = os.path.join(DATASET_ROOT, "VM_spoof")

EXTRACT_LAYER = 5   # couche XLS-R utilisée pour l'extraction
FEAT_LEN      = 300  # nombre de frames (pad/crop)
THRESH_REF    = 0.572274  # ITW reference EER threshold (score >= THRESH_REF -> spoof)

if CODECFAKE_REPO_PATH not in sys.path:
    sys.path.insert(0, CODECFAKE_REPO_PATH)

# ============================================================
# GPU
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# Chargement XLS-R
# ============================================================

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

print("\nChargement de XLS-R (wav2vec2-xls-r-300m)...")
xlsr_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
xlsr_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
xlsr_model = xlsr_model.to(device)
xlsr_model.eval()
print(f"XLS-R chargé — extraction depuis la couche {EXTRACT_LAYER}")

# ============================================================
# Chargement du backend AASIST
# ============================================================

print(f"\nChargement du modèle AASIST depuis :\n  {MODEL_PATH}")
aasist_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
aasist_model = aasist_model.to(device)
aasist_model.eval()
print(f"Modèle chargé : {type(aasist_model).__name__}")

# ============================================================
# Fonctions utilitaires
# ============================================================

def extract_features(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)
    with torch.no_grad():
        inputs = xlsr_feature_extractor(
            waveform.numpy(), sampling_rate=target_sr, return_tensors="pt"
        )
        outputs = xlsr_model(
            inputs.input_values.to(device), output_hidden_states=True
        )
        hidden = outputs.hidden_states[EXTRACT_LAYER]  # (1, T, 1024)
    return hidden.cpu()


def pad_or_crop(features, length=FEAT_LEN):
    T = features.shape[1]
    if T > length:
        return features[:, :length, :]
    if T < length:
        return torch.nn.functional.pad(features, (0, 0, 0, length - T))
    return features


def infer_score(features):
    feat = pad_or_crop(features).float().to(device)
    feat = feat.permute(0, 2, 1)   # (1, 1024, T)
    feat = feat.unsqueeze(dim=0)   # (1, 1, 1024, T)
    with torch.no_grad():
        _feats, output = aasist_model(feat)
    return torch.nn.functional.softmax(output, dim=-1)[0, 1].item()


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, thresholds[idx]


# ============================================================
# Collecte des fichiers audio et labels
# ============================================================

audio_extensions = ("*.wav", "*.flac", "*.mp3", "*.ogg")

def collect_files(folder, label):
    files = []
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return [(f, label) for f in sorted(files)]

bonafide_files = collect_files(BONAFIDE_DIR, "bonafide")
spoof_files    = collect_files(SPOOF_DIR,    "spoof")
all_files      = bonafide_files + spoof_files

print(f"\nDataset : {DATASET_ROOT}")
print(f"  bonafide : {len(bonafide_files)} fichiers")
print(f"  spoof    : {len(spoof_files)} fichiers")
print(f"  total    : {len(all_files)} fichiers")

# ============================================================
# Inférence
# ============================================================

scores      = []
true_labels = []   # 0 = bonafide, 1 = spoof
file_names  = []
errors      = []

t_start = time.time()
for audio_path, label_str in tqdm(all_files, desc="Inférence"):
    try:
        feats = extract_features(audio_path)
        score = infer_score(feats)
        scores.append(score)
        true_labels.append(1 if label_str == "spoof" else 0)
        file_names.append(audio_path)
    except Exception as exc:
        errors.append((audio_path, str(exc)))
        print(f"\n  Erreur {os.path.basename(audio_path)}: {exc}")

elapsed = time.time() - t_start

# ============================================================
# Métriques
# ============================================================

scores_arr = np.array(scores)
labels_arr = np.array(true_labels)

eer, threshold = compute_eer(labels_arr, scores_arr)
pred_labels    = (scores_arr >= threshold).astype(int)
accuracy       = accuracy_score(labels_arr, pred_labels)
cm             = confusion_matrix(labels_arr, pred_labels)

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

print(f"\n{'='*55}")
print(f"  RESULTATS — dataset_50_50")
print(f"{'='*55}")
print(f"  Fichiers traités : {len(scores)} / {len(all_files)}")
print(f"  EER              : {eer*100:.2f}%")
print(f"  Seuil EER        : {threshold:.4f}")
print(f"  Accuracy         : {accuracy*100:.2f}%")
print(f"  Temps total      : {elapsed:.1f}s  ({elapsed/len(scores):.3f}s/fichier)")
print(f"\n  Matrice de confusion (lignes=vrai, col=prédit)")
print(f"              bonafide   spoof")
print(f"  bonafide    {cm[0,0]:>8}   {cm[0,1]:>5}")
print(f"  spoof       {cm[1,0]:>8}   {cm[1,1]:>5}")
if errors:
    print(f"\n  Erreurs : {len(errors)}")
print(f"{'='*55}")

# Détail par fichier
print(f"\n{'Fichier':<45} {'Label vrai':<12} {'Score':>8}  {'Prédit':<10}")
print("-" * 80)
for fpath, true_l, score in zip(file_names, true_labels, scores):
    true_str = "bonafide" if true_l == 0 else "spoof"
    pred_str = "bonafide" if score < threshold else "spoof"
    ok = "OK" if true_str == pred_str else "ERREUR"
    print(f"  {os.path.basename(fpath):<43} {true_str:<12} {score:>8.4f}  {pred_str:<10} {ok}")

# ============================================================
# Sauvegarde des scores
# ============================================================

output_dir  = os.path.join(CODECFAKE_REPO_PATH, "result")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "score_antton_dataset.txt")

with open(output_file, "w") as f:
    f.write("# filename score true_label predicted_label\n")
    for fpath, true_l, score in zip(file_names, true_labels, scores):
        true_str = "bonafide" if true_l == 0 else "spoof"
        pred_str = "bonafide" if score < threshold else "spoof"
        f.write(f"{os.path.basename(fpath)} {score:.6f} {true_str} {pred_str}\n")

print(f"\nScores sauvegardés dans : {output_file}")

# ============================================================
# Visualisation — score par fichier (style lollipop)
# ============================================================

PLOT_OUT = os.path.join(output_dir, "scores_w2v2aasist_plot.png")

COLOR_BONAFIDE = "#2196F3"   # blue
COLOR_SPOOF    = "#F44336"   # red
COLOR_THRESH   = "#FF9800"   # orange
COLOR_REF      = "#9C27B0"   # purple

# Sort by ascending score
fnames_short = [os.path.basename(p) for p in file_names]
order          = np.argsort(scores_arr)
sorted_scores  = scores_arr[order]
sorted_labels  = labels_arr[order]
sorted_fnames  = [fnames_short[i] for i in order]
sorted_preds   = (sorted_scores >= threshold).astype(int)
is_wrong       = sorted_preds != sorted_labels

n = len(sorted_scores)
y = np.arange(n)

# Margin for the uncertainty zone: 5% of the score range
score_range = sorted_scores.max() - sorted_scores.min()
hesitation  = score_range * 0.05

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(11, max(6, n * 0.45)))
fig.patch.set_facecolor("white")

for i in range(n):
    color  = COLOR_BONAFIDE if sorted_labels[i] == 0 else COLOR_SPOOF
    ax.hlines(i, sorted_scores.min(), sorted_scores[i],
              color=color, linewidth=1.2, alpha=0.5)
    marker = "D" if is_wrong[i] else "o"
    ax.scatter(sorted_scores[i], i, color=color,
               marker=marker, s=90, zorder=3,
               edgecolors="black" if is_wrong[i] else "none", linewidths=1.2)

ax.axvline(threshold, color=COLOR_THRESH, linewidth=2.0, linestyle="--",
           label=f"EER threshold = {threshold:.4f}")
ax.axvspan(threshold - hesitation, threshold + hesitation,
           alpha=0.08, color=COLOR_THRESH, label="Uncertainty zone (±5%)")

# ITW reference threshold with HTER annotation
ax.axvline(THRESH_REF, color=COLOR_REF, linewidth=1.8, linestyle=":",
           label=f"ITW reference threshold = {THRESH_REF}")
ax.text(THRESH_REF + score_range * 0.01, n * 0.5,
        f"HTER @ITW = {hter_ref:.2f}%\n(FPR={fpr_ref*100:.1f}% / FNR={fnr_ref*100:.1f}%)",
        color=COLOR_REF, fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_REF, alpha=0.8))

ax.set_yticks(y)
ax.set_yticklabels(sorted_fnames, fontsize=8)

patch_b = mpatches.Patch(color=COLOR_BONAFIDE, label="Bonafide (true)")
patch_s = mpatches.Patch(color=COLOR_SPOOF,    label="Spoof (true)")
patch_w = ax.scatter([], [], marker="D", color="gray",
                     edgecolors="black", linewidths=1.2, label="Misclassified")
ax.legend(
    handles=[
        patch_b, patch_s, patch_w,
        plt.Line2D([0], [0], color=COLOR_THRESH, linewidth=2, linestyle="--",
                   label=f"EER threshold = {threshold:.4f}"),
        plt.Line2D([0], [0], color=COLOR_REF, linewidth=1.8, linestyle=":",
                   label=f"ITW reference threshold = {THRESH_REF}"),
    ],
    fontsize=9, loc="lower right", framealpha=0.9,
)

ax.set_xlabel(
    "Spoof score  →  lower = certain bonafide  |  higher = certain spoof",
    fontsize=10,
)
ax.set_ylabel("Audio file", fontsize=10)
ax.set_title(
    f"W2V2-AASIST (Codecfake) — Score per file\n"
    f"EER = {eer*100:.2f}%  (threshold {threshold:.4f})  |  "
    f"HTER @ITW threshold ({THRESH_REF}) = {hter_ref:.2f}%  |  Accuracy = {accuracy*100:.1f}%",
    fontsize=12, fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)

ax.text((sorted_scores.min() + threshold) / 2, n - 0.5,
        "← predicted bonafide", ha="center", va="top", fontsize=8, color="#555")
ax.text((threshold + sorted_scores.max()) / 2, n - 0.5,
        "predicted spoof →",    ha="center", va="top", fontsize=8, color="#555")

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
print(f"Plot saved: {PLOT_OUT}")
plt.show()

# ============================================================
# Visualization — confusion matrices
# Two matrices: one at the optimal EER threshold,
# one at the ITW reference threshold.
# Convention: positive class = spoof. Predicted spoof <=> score >= threshold.
#   FAR (False Acceptance Rate) = spoof accepted as bonafide = FN / n_spoof
#   FRR (False Rejection Rate)  = bonafide rejected as spoof = FP / n_bonafide
# ============================================================

CM_OUT = os.path.join(output_dir, "confusion_matrices_w2v2aasist.png")

preds_opt = (scores_arr >= threshold).astype(int)    # 1 = spoof
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
               f"Optimal EER threshold = {threshold:.4f}")
draw_confusion(axes[1], preds_itw, labels_arr,
               f"ITW reference threshold = {THRESH_REF}")

fig_cm.suptitle("W2V2-AASIST (Codecfake) — Confusion matrices (positive class = spoof)",
                fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
print(f"Confusion matrices saved: {CM_OUT}")
plt.show()
