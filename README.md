# COMP472 : réseau de neurones, kNN et ensembles (Kaggle, Fall 2026)

Deux tâches de classification, une seule soumission Kaggle de 10 000 lignes :
- **image** : 500 features (histogrammes de « mots visuels » normalisés L2), 50 classes ;
- **sentiment** : 768 features (embeddings de texte), 3 classes.

**Meilleure soumission : `results/submissions/iter9.csv`, Kaggle 0.661.** Candidate : `results/submissions/iter11.csv` (iter9 + label spreading, val. image 0.6515 contre 0.637), pas encore soumise (objectif 0.651, meilleur score de la compétition 0.671). L'historique complet des itérations est dans [COMPTE_RENDU_NN.md](COMPTE_RENDU_NN.md), et les figures dans [results/figures/](results/figures/).

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# placer les CSV Kaggle dans data/ : image_train.csv, image_test.csv,
# sentiment_train.csv, sentiment_test.csv, sample_submission.csv
```

## Utilisation

Tout passe par une seule ligne de commande : `python -m comp472 <commande>`.

| Commande | Rôle | Durée (CPU 4 cœurs) |
|---|---|---|
| `list` | liste les expériences, recherches, recettes et figures | — |
| `submit iter9` | reconstruit la meilleure soumission à partir des probabilités sauvegardées | ~20 s |
| `submit iter11` | iter9 + label spreading sur le graphe kNN | ~1 min |
| `verify` | vérifie que tous les résultats sauvegardés sont reproduits à l'identique | ~3 min |
| `plots` | génère les 15 figures dans `results/figures/` | ~20 s |
| `train iter8_nn` | entraîne un réseau (validation + 5 seeds), sauvegarde probabilités et CSV | 20 min à 2 h 30 |
| `search image-2` | recherche d'hyperparamètres (un facteur à la fois) | 30 min à 2 h |
| `seed-study` | ensembles de seeds / architectures sur le sentiment | ~20 min |
| `knn-grid` | premier kNN Hellinger (k × pondération) | ~1 min |
| `knn-search` | kNN affiné + pseudo-étiquetage, jugé dans l'ensemble | ~2 min |

Reproduire iter9 de zéro : `train iter7_nn`, puis `train iter8_nn`, puis `submit iter9` (environ 4 h de calcul).

Option globale : `--threads N` (8 par défaut, ou la variable `$NTHREADS`).

## Architecture

```
rendu_AI/
├── comp472/                 package Python
│   ├── paths.py             tous les chemins du projet
│   ├── config.py            hyperparamètres : MLPConfig, Experiment, registre EXPERIMENTS, KNNParams
│   ├── data.py              chargement des tâches (TaskData), split 80/20 commun, to_counts
│   ├── features.py          FeaturePipeline (entrée du MLP), géométries du kNN (Hellinger, PowerNormalize)
│   ├── models/
│   │   ├── mlp.py           MLP (PyTorch), predict / predict_proba
│   │   ├── trainer.py       MLPTrainer : early stopping, mixup, thinning, bruit, SWA, courbes
│   │   └── knn.py           voisins, vote pondéré, KNNClassifier, fusion avec les pseudo-étiquettes
│   ├── ensemble.py          mélange w·réseau + (1-w)·kNN, choix de w, PseudoLabelEnsemble (iter9)
│   ├── experiment.py        NNExperimentRunner : protocole validation → 5 seeds → soumission
│   ├── recipes.py           recettes de soumission iter6 … iter10 (registre RECIPES)
│   ├── search.py            recherches d'hyperparamètres (registre SEARCHES), étude des seeds, kNN
│   ├── store.py             ProbabilityStore : probabilités sauvegardées (val / test)
│   ├── submission.py        construction, contrôle et sauvegarde d'une soumission
│   ├── history.py           scores de chaque itération (validation et Kaggle)
│   ├── plots.py             15 figures matplotlib (registre FIGURES)
│   ├── verify.py            contrôle de reproductibilité
│   └── cli.py, __main__.py  ligne de commande
├── tests/test_equivalence.py  équivalence bit à bit avec les scripts d'origine
├── data/                    CSV Kaggle (non versionnés)
├── results/
│   ├── probabilities/       probabilités de validation et de test de chaque modèle (.npy)
│   ├── submissions/         iter1.csv … iter10.csv
│   ├── search/              tableaux des recherches (hyperparamètres, kNN, sous-ensembles)
│   ├── logs/                logs des runs, courbes d'apprentissage
│   └── figures/             figures du rapport
├── notebooks/               notebook du lab 7
└── COMPTE_RENDU_NN.md       compte rendu détaillé des itérations
```

### Choix de conception

- **Configuration déclarative** : chaque réseau est un `MLPConfig` immuable (dataclass gelée). Une variante se crée par copie (`IMAGE_ITER2.but(hidden=(8192,))`), sans dictionnaire libre ni clé mal orthographiée.
- **Registres** (`EXPERIMENTS`, `SEARCHES`, `RECIPES`, `FIGURES`) : ajouter une expérience, une recherche, une recette ou une figure revient à ajouter une entrée. La ligne de commande les découvre automatiquement.
- **Stratégie pour la géométrie du kNN** : `KNNClassifier` reçoit un objet `Hellinger()` ou `PowerNormalize(p, idf)`. Le vote et la recherche des voisins ne changent pas.
- **Séparation calcul / stockage** : les modèles produisent des probabilités, `ProbabilityStore` les sauvegarde, et les recettes d'ensemble ne lisent que ces fichiers. Reconstruire une soumission prend donc quelques secondes au lieu de plusieurs heures.
- **Un seul protocole** (`NNExperimentRunner`) pour tous les réseaux : split 80/20 stratifié (seed 42), early stopping, réentraînement sur 100 % pendant `best_epoch` epochs avec 5 seeds, moyenne des probabilités.

## Garantie : les résultats n'ont pas changé

La réorganisation a été vérifiée contre le code d'origine, conservé sur la branche git `sauvegarde-originale` :

- `python -m comp472 verify` reconstruit les soumissions iter6, iter7, iter8, iter9, iter10a et iter10. Elles sont **identiques à 100 %** aux fichiers soumis sur Kaggle. Le kNN Hellinger et la recherche kNN affinée redonnent **exactement** les mêmes probabilités et les mêmes tableaux.
- `python tests/test_equivalence.py` (ou `pytest tests`) extrait les scripts d'origine de la branche de sauvegarde. Il vérifie que les features, le split et **les poids des réseaux après entraînement** sont identiques bit à bit, pour chaque option : early stopping, epochs fixes, mixup, thinning, bruit, SWA, features sans standardisation, et le réseau d'iter1.

Les réseaux eux-mêmes n'ont pas été réentraînés en entier (plusieurs heures) : leurs probabilités sauvegardées sont celles produites par le code d'origine.

## Correspondance avec les anciens scripts

| Ancien script (branche `sauvegarde-originale`) | Nouvelle commande |
|---|---|
| `neural_network.py` (iter1) | `train iter1` |
| `nn_v2.py iterN.csv` | `train iterN` |
| `hp_search.py image` / `sentiment` | `search image-v1` / `search sentiment-v1` |
| `hp_search2.py image 2` | `search image-2` |
| `ens_eval.py sentiment` | `seed-study` |
| `knn_hellinger.py` | `knn-grid` |
| `knn_v2.py` | `knn-search`, puis `submit iter9` |
| `make_iter6.py` … `make_iter10.py` | `submit iter6` … `submit iter10` |
| `hp_progress.py` | supprimé (outil de suivi en direct) |

Note : les rounds 1 à 3 de la deuxième recherche utilisent la base historique `IMAGE_SCREENING` ([2048], dropout 0.5, sans dropout d'entrée), qui est celle des résultats enregistrés. Dans le script d'origine, cette base avait été modifiée depuis.
