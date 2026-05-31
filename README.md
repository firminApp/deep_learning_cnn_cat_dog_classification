# CNN Cats vs Dogs — From Scratch vs Transfer Learning

**Auteur :** Banigante Kpapou — Master 1 IA  
**Cours :** Deep Learning

## Objectif

Comparer un CNN entraîné **from scratch** (4 blocs Conv+BN+ReLU+Pool+Dropout) et un modèle en **transfert d'apprentissage** (ResNet18 pré-entraîné ImageNet) sur la tâche de classification chats/chiens. Montrer l'impact sur la convergence, la performance et la robustesse.

---

## Environnement

```bash
cd cnn-catsdogs-BaniganteKpapou
pip install -r requirements.txt
```

> GPU recommandé. Le code détecte automatiquement CUDA et bascule sur CPU si indisponible.

---

## Organisation des données

Le code attend la structure suivante dans `cnn-catsdogs-BaniganteKpapou/data/cats_vs_dogs/` :

```
data/cats_vs_dogs/
├── train/
│   ├── cat/   (11 250 images .jpg)
│   └── dog/   (11 250 images .jpg)
└── test/
    ├── cat/   (1 250 images .jpg)
    └── dog/   (1 250 images .jpg)
```

**Source des données :** dataset Kaggle "Dogs vs. Cats" ou version distribuée en cours.  
Les données ne sont **pas** poussées sur GitHub (voir `.gitignore`).

### Sur Google Colab

```bash
# Montez votre Drive ou uploadez les données, puis :
python train_cats_dogs.py --data-dir /content/data/cats_vs_dogs
```

---

## Commandes d'entraînement

### Expérience A — CNN from Scratch

```bash
# SGD
python train_cats_dogs.py \
  --data-dir data/cats_vs_dogs \
  --epochs 15 --optimizer sgd --lr 1e-2 \
  --batch-size 32 --scheduler-step 5 --scheduler-gamma 0.5 \
  --checkpoint-dir checkpoints

# Adam
python train_cats_dogs.py \
  --data-dir data/cats_vs_dogs \
  --epochs 15 --optimizer adam --lr 1e-3 \
  --batch-size 32 --scheduler-step 5 --scheduler-gamma 0.5 \
  --checkpoint-dir checkpoints
```

Arguments clés :
| Argument | Description |
|---|---|
| `--batch-size` | Taille de batch (défaut : 32) |
| `--lr` | Learning rate initial |
| `--epochs` | Nombre d'époques |
| `--optimizer` | `adam` ou `sgd` |
| `--scheduler-step` | Pas du StepLR |
| `--scheduler-gamma` | Facteur de réduction du LR |
| `--valid-ratio` | Fraction de validation (défaut : 0.1) |

### Expérience B — Transfer Learning (ResNet18)

```bash
# Adam — feature extraction (backbone gelé)
python train_cats_dogs.py \
  --data-dir data/cats_vs_dogs \
  --epochs 15 --optimizer adam --lr 1e-4 \
  --feature-extract \
  --checkpoint-dir checkpoints

# SGD — feature extraction
python train_cats_dogs.py \
  --data-dir data/cats_vs_dogs \
  --epochs 15 --optimizer sgd --lr 1e-3 \
  --feature-extract \
  --checkpoint-dir checkpoints
```

**`--feature-extract`** : gèle toutes les couches du backbone ResNet18, seule la tête fully-connected est entraînée.  
Sans ce flag : fine-tuning complet (toutes les couches entraînées).

### Mode debug rapide

```bash
python train_cats_dogs.py \
  --data-dir data/cats_vs_dogs \
  --epochs 1 \
  --max-train-batches 20 --max-valid-batches 5 --max-test-batches 5
```

---

## Évaluation / Rechargement du modèle

Le meilleur checkpoint (validation loss minimale) est sauvegardé automatiquement sous `checkpoints/`.  
Il est rechargé automatiquement en fin d'entraînement pour le test final.

Pour évaluer manuellement un checkpoint :

```python
import torch
from train_cats_dogs import init_model, evaluate, get_data_loaders
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, test_loader, classes = get_data_loaders(
    Path('data/cats_vs_dogs'), img_size=224, batch_size=32,
    valid_ratio=0.1, num_workers=4, seed=42)

model = init_model('resnet18', num_classes=2, feature_extract=True).to(device)
model.load_state_dict(torch.load('checkpoints/resnet18_transfer_adam.pth', map_location=device))
metrics = evaluate(model, test_loader, device, num_classes=2)
print(metrics)
```

---

## Résultats

> Les courbes et métriques ci-dessous sont issues d'un entraînement 15 époques sur GPU (CUDA).  
> Voir `notebook.ipynb` pour les figures complètes.

### Tableau de performance (jeu de test)

| Expérience | Loss | Accuracy | Precision | Recall |
|---|---|---|---|---|
| A1 – CNN Scratch / SGD    | ~0.45 | ~0.79 | ~0.79 | ~0.79 |
| A2 – CNN Scratch / Adam   | ~0.38 | ~0.83 | ~0.83 | ~0.83 |
| B1 – ResNet18 TL / Adam   | ~0.12 | ~0.96 | ~0.96 | ~0.96 |
| B2 – ResNet18 TL / SGD    | ~0.15 | ~0.94 | ~0.94 | ~0.94 |

*Valeurs indicatives — les résultats exacts sont affichés dans le notebook après exécution.*

### Analyse comparative

**Convergence :** Le transfert learning (ResNet18) converge nettement plus vite. Dès les 3 premières époques, la validation accuracy dépasse 90 %, contre ~70 % pour le CNN scratch. Les représentations bas-niveau (textures, bords) pré-apprises sur ImageNet sont directement réutilisables pour distinguer chats et chiens.

**Performance finale :** ResNet18 surpasse le CNN scratch d'environ 13 points d'accuracy sur le test. Avec seulement la tête fully-connected entraînée (~0.5 % des paramètres), le modèle de transfert obtient de meilleurs résultats en moins d'époques.

**Optimiseurs :** Adam converge plus régulièrement que SGD sur les premières époques. SGD avec momentum peut rattraper Adam mais nécessite un learning rate plus élevé et une décroissance bien calibrée. Sur le scratch, la différence Adam vs SGD est de ~4 points d'accuracy.

### Limites et pistes d'amélioration

- Fine-tuning progressif des couches profondes de ResNet18 (dégel graduel).
- Data augmentation plus agressive (CutMix, MixUp, RandAugment).
- Architecture scratch plus profonde avec residual connections.
- Recherche systématique d'hyperparamètres (Optuna ou grid search).
- Journalisation W&B pour le suivi temps réel.

---

## Structure du dépôt

```
cnn-catsdogs-BaniganteKpapou/
├── notebook.ipynb                   ← notebook principal (31 cellules)
├── train_cats_dogs.py               ← script CLI d'entraînement
├── train_cats_dogs_colab.ipynb      ← version Colab
├── requirements.txt
├── README.md
└── .gitignore
```

`.gitignore` exclut : `data/`, `checkpoints/`, `*.pth`, `*.pt`, `runs/`, `__pycache__/`.
