"""
Évaluation du modèle Codecfake (W2V2-AASIST) sur le dataset In-The-Wild (ITW).
Les labels sont lus depuis meta.csv : colonne 'label' vaut 'bona-fide' ou 'spoof'.
"""

import os
import sys
import csv
import time
import torch
import types
import numpy as np
import torchaudio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import eval_metrics as em

if "pytorch_model_summary" not in sys.modules:
    stub = types.ModuleType("pytorch_model_summary")
    stub.summary = lambda *_, **__: None
    sys.modules["pytorch_model_summary"] = stub

# ============================================================
# Configuration
# ============================================================

CODECFAKE_REPO_PATH = "/home/antton/projets_bachelor/model_pre-entraine_codecfake/Codecfake"
MODEL_PATH          = os.path.join(CODECFAKE_REPO_PATH,
                          "pretrained_model/cotrain_w2v2aasist_CSAM/anti-spoofing_feat_model.pt")
DATASET_DIR = "/home/antton/data/release_in_the_wild/"
META_CSV    = os.path.join(DATASET_DIR, "meta.csv")

EXTRACT_LAYER = 5
AUDIO_CUT     = 64600   # samples at 16 kHz — must match training pipeline

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

def pad_or_truncate(waveform, cut=AUDIO_CUT):
    """Truncate or tile-repeat waveform to exactly `cut` samples — mirrors pad_dataset() from generate_score.py."""
    n = waveform.shape[0]
    if n >= cut:
        return waveform[:cut]
    num_repeats = cut // n + 1
    return waveform.repeat(num_repeats)[:cut]


def extract_features(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    waveform = pad_or_truncate(waveform.squeeze(0))   # always 64600 samples
    with torch.no_grad():
        inputs = xlsr_feature_extractor(
            waveform.numpy(), sampling_rate=target_sr, return_tensors="pt"
        )
        outputs = xlsr_model(
            inputs.input_values.to(device), output_hidden_states=True
        )
        hidden = outputs.hidden_states[EXTRACT_LAYER]  # (1, ~202, 1024)
    return hidden.cpu()


def infer_score(features):
    feat = features.float().to(device)
    feat = feat.permute(0, 2, 1)   # (1, 1024, T)
    feat = feat.unsqueeze(dim=0)   # (1, 1, 1024, T)
    with torch.no_grad():
        _feats, output = aasist_model(feat)
    return torch.nn.functional.softmax(output, dim=-1)[0, 0].item()



# ============================================================
# Lecture du meta.csv
# ============================================================

all_files = []
with open(META_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename  = row["file"]
        label_str = row["label"]   # 'bona-fide' or 'spoof'
        full_path = os.path.join(DATASET_DIR, filename)
        if os.path.isfile(full_path):
            all_files.append((full_path, label_str))
        else:
            print(f"  [AVERTISSEMENT] fichier manquant : {full_path}")

bonafide_count = sum(1 for _, l in all_files if l == "bona-fide")
spoof_count    = sum(1 for _, l in all_files if l == "spoof")

print(f"\nDataset ITW : {DATASET_DIR}")
print(f"  bona-fide : {bonafide_count} fichiers")
print(f"  spoof     : {spoof_count} fichiers")
print(f"  total     : {len(all_files)} fichiers")

# ============================================================
# Inférence
# ============================================================

scores      = []
true_labels = []   # 0 = bona-fide, 1 = spoof
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

bona_scores  = scores_arr[labels_arr == 0]
spoof_scores = scores_arr[labels_arr == 1]

eer_fwd, threshold = em.compute_eer(bona_scores, spoof_scores)
eer_neg, _         = em.compute_eer(-bona_scores, -spoof_scores)
eer                = min(eer_fwd, eer_neg)

pred_labels = (scores_arr < threshold).astype(int)   # lower bona-fide prob → spoof
accuracy       = accuracy_score(labels_arr, pred_labels)
cm             = confusion_matrix(labels_arr, pred_labels)

print(f"\n{'='*55}")
print(f"  RESULTATS — ITW (In-The-Wild)")
print(f"{'='*55}")
print(f"  Fichiers traités : {len(scores)} / {len(all_files)}")
print(f"  EER              : {eer*100:.2f}%")
print(f"  Seuil EER        : {threshold:.4f}")
print(f"  Accuracy         : {accuracy*100:.2f}%")
print(f"  Temps total      : {elapsed:.1f}s  ({elapsed/max(len(scores),1):.3f}s/fichier)")
print(f"\n  Matrice de confusion (lignes=vrai, col=prédit)")
print(f"              bona-fide   spoof")
print(f"  bona-fide   {cm[0,0]:>9}   {cm[0,1]:>5}")
print(f"  spoof       {cm[1,0]:>9}   {cm[1,1]:>5}")
if errors:
    print(f"\n  Erreurs : {len(errors)}")
print(f"{'='*55}")

# ============================================================
# Sauvegarde des scores
# ============================================================

output_dir  = os.path.join(CODECFAKE_REPO_PATH, "result")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "score_itw.txt")

with open(output_file, "w") as f:
    f.write("# filename score true_label predicted_label\n")
    for fpath, true_l, score in zip(file_names, true_labels, scores):
        true_str = "bona-fide" if true_l == 0 else "spoof"
        pred_str = "spoof" if score < threshold else "bona-fide"
        f.write(f"{os.path.basename(fpath)} {score:.6f} {true_str} {pred_str}\n")

print(f"\nScores sauvegardés dans : {output_file}")
