# CNN Cats vs Dogs — From Scratch vs Transfer Learning

## Objectif
Comparer un modèle CNN entraîné from scratch et un modèle basé sur le transfert learning (ResNet18) sur une tâche de classification Cats vs Dogs. Le carnet montre l’impact du transfert learning sur la convergence, la performance et la robustesse.

## Structure du rendu
- `devoir_rendu/cats_dogs_cnn_transfer.ipynb` : notebook principal d’entraînement et d’évaluation.
- `devoir_rendu/requirements.txt` : dépendances Python.
- `devoir_rendu/.gitignore` : fichiers et dossiers ignorés.

## Installation
```bash
cd devoir_rendu
python -m pip install -r requirements.txt
```

## Organisation des données
Le notebook attend un dossier `data/cats_vs_dogs/` au niveau de `devoir_rendu/`.

Structure recommandée :

```
devoir_rendu/data/cats_vs_dogs/train/cats/
devoir_rendu/data/cats_vs_dogs/train/dogs/
devoir_rendu/data/cats_vs_dogs/test/cats/
devoir_rendu/data/cats_vs_dogs/test/dogs/
```

Le notebook utilise le dossier `train/` pour créer un split train/validation avec `valid_ratio = 0.1`.

### Sources de données possibles
- Kaggle Dogs vs Cats
- un jeu de données équivalent de chats et chiens

> Important : ne pas versionner les données ni les checkpoints.

## Lancer le notebook
1. `cd devoir_rendu`
2. `jupyter notebook cats_dogs_cnn_transfer.ipynb`
3. Exécuter les cellules dans l’ordre.

## Expériences réalisées
- **Expérience A** : CNN entraîné from scratch avec blocs Conv2d + BatchNorm + Dropout.
- **Expérience B** : Transfert learning avec ResNet18 pré-entraîné et nouvelle tête de classification.

## Hyperparamètres clés
- Image size : `224x224`
- Batch size : `32`
- Optimiseur : `Adam`
- Loss : `CrossEntropyLoss`
- Scheduler : `StepLR`
- Seed fixé pour reproductibilité

## Résultats attendus
- Courbes d’entraînement et de validation pour loss et accuracy.
- Comparaison des performances entre les deux modèles.
- Métriques de test finales : accuracy, précision, recall.
- Matrices de confusion pour chaque modèle.

## Checkpoints
Les checkpoints sont sauvegardés dans `devoir_rendu/checkpoints/`.

## Bonnes pratiques
- Utiliser un GPU si disponible.
- Sauvegarder seulement les checkpoints locaux. Ne pas pousser `*.pt`, `*.pth`, `data/`, `checkpoints/`.
- Si vous changez l’organisation des données, ajustez `DATA_DIR` dans le notebook.
