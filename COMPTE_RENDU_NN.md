# Compte rendu : réseau de neurones (COMP472, Fall 2026)

Suivi des itérations du modèle réseau de neurones (MLP) pour les deux tâches :

- **image** : 500 features, 50 classes, 20 000 exemples d'entraînement, 5 000 de test ;
- **sentiment** : 768 features (embeddings de texte), 3 classes (négatif, neutre, positif), 12 000 exemples d'entraînement, 5 000 de test.

Une soumission contient 10 000 lignes (`id,label`). Le score Kaggle est l'accuracy sur l'ensemble des lignes, soit à peu près la moyenne des deux tâches.

**Méthode de validation, commune à toutes les itérations** : on découpe les données d'entraînement en 80 % / 20 % (stratifié, seed 42). Le prétraitement est ajusté sur les 80 % seulement. Le nombre d'epochs est choisi par arrêt anticipé (early stopping) sur les 20 %. On réentraîne ensuite sur 100 % des données pendant ce nombre d'epochs, avec 5 seeds, et on moyenne les probabilités.

## Récapitulatif

| Itération | Fichier | Val. image | Val. sentiment | Val. moyenne | Kaggle |
|---|---|---|---|---|---|
| iter1 | `iter1.csv` | 0.397 | 0.673 | 0.535 | **0.539** |
| iter2 | `iter2.csv` | 0.5085 | 0.6817 | 0.595 | **0.609** |
| iter3 | `iter3.csv` | 0.5075 | 0.6817 | 0.595 | **0.608** |
| iter4 | `iter4.csv` | 0.5155 | 0.6817 | 0.599 | **0.619** |
| iter5 | — | 0.511 (meilleure variante : SWA) | | | pas de gain, non soumis |
| iter6 | `iter6.csv` | **0.589** (kNN Hellinger) | 0.6817 (réseau iter4) | **0.636** | **0.637** |
| iter7 | `iter7.csv` | **0.627** (réseau + kNN) | 0.6817 | **0.654** | **0.657** |
| iter8 | `iter8.csv` | **0.630** (moyenne réseaux [4096]+[8192], + kNN) | 0.6817 | **0.656** | **0.659** |
| iter9 | `iter9.csv` | **0.637** (réseaux + kNN affiné + pseudo-étiquetage) | 0.6817 | **0.659** | **0.661** |
| iter10a | `iter10a.csv` | 0.6372 (iter9 + réseau a) | 0.6817 | 0.659 | **0.660** |
| iter10 | `iter10.csv` | 0.6385 (5 réseaux + kNN + pseudo-étiquetage) | 0.6817 | 0.660 | non soumis |

La moyenne de validation prédit bien le score Kaggle, en le sous-estimant légèrement : 0.535 contre 0.539 pour iter1, 0.595 contre 0.609 pour iter2. Le réentraînement sur 100 % des données et la moyenne sur 5 seeds apportent probablement ce petit bonus. On se sert donc de la validation comme indicateur.

---

## Itération 1 : modèle de base, sans recherche d'hyperparamètres

**Code** : `neural_network.py` (configuration de départ)

**Architecture** : couches `Linear → BatchNorm → ReLU → Dropout`, puis une couche de sortie linéaire.

- Loss : entropie croisée avec label smoothing 0.1.
- Optimiseur : AdamW (weight decay 0.01), taux d'apprentissage en cosinus.
- Batch de 128, 60 epochs au maximum, patience de 10.

| | Image | Sentiment |
|---|---|---|
| Prétraitement | racine carrée des features (Hellinger), puis standardisation | standardisation |
| Couches cachées | [1024, 512] | [512, 128] |
| Dropout | 0.5 | 0.5 |
| Taux d'apprentissage | 1e-3 | 5e-4 |
| Val. accuracy | 0.397 (epoch 49) | 0.673 (epoch 11) |

**Résultat Kaggle : 0.539.**

**Observation** : l'image surapprend énormément. L'accuracy d'entraînement atteint 1.0 alors que la validation reste à 0.40.

### Entre iter1 et iter2 : première recherche d'hyperparamètres (`hp_search.py`)

On a fait varier un facteur à la fois autour de la configuration de base. Résultats dans `hp_results_*.csv`.

- Image : les couches [2048, 1024] donnent 0.429 (contre 0.397). Enlever la racine carrée donne 0.401 contre 0.397 : différence négligeable, donc on l'a gardée.
- Sentiment : un dropout de 0.7 donne 0.681 (contre 0.673).

Aucun CSV n'a été produit avec ces réglages.

---

## Itération 2 : régularisation de l'entrée et réseau plus large

**Code** : `nn_v2.py` (pipeline identique, avec de nouvelles options) et `hp_search2.py` (recherche). Résultats dans `hp2_results_image.csv` et `hp2_results_sentiment.csv`.

**Nouvelles options testées** :

- `in_dropout` : dropout appliqué directement sur les features d'entrée ;
- `row_norm` : normalisation L1 de chaque ligne avant la racine (vrai Hellinger) ;
- `l2_norm` : normalisation L2 de chaque ligne ;
- `mixup` : interpolation entre paires d'exemples ;
- `smoothing` : réglage du label smoothing.

### Image : de 0.429 à 0.509

Variation d'un facteur à la fois :

| Variante | Val. acc |
|---|---|
| base [2048] (référence) | 0.4218 |
| + row_norm | 0.4238 |
| + in_dropout 0.2 | 0.466 |
| + in_dropout 0.4 | 0.479 |
| **+ in_dropout 0.5** | **0.4808** |
| + in_dropout 0.6 | 0.4557 |
| + in_dropout 0.7 | 0.4315 |
| in_dropout 0.5, deux couches [2048, 1024] | 0.4492 |
| in_dropout 0.5, une couche [4096] | 0.496 |
| in_dropout 0.5, dropout 0.3 | 0.4875 |
| **[4096], in_dropout 0.5, dropout 0.3** | **0.5092** |
| [4096], in_dropout 0.5, 100 epochs | 0.508 |
| [4096], in_dropout 0.5, mixup 0.2, 100 epochs | 0.509 |
| [8192], in_dropout 0.5 | 0.514 |

**Conclusions** :

1. Le dropout sur l'entrée est le levier principal (+6 points). Les features ressemblent à des histogrammes creux (85 % de zéros). Masquer aléatoirement la moitié des features force le réseau à ne pas dépendre de quelques-unes, ce qui combat directement le surapprentissage.
2. Avec ce dropout, **une seule couche large** fait mieux que deux couches : [4096] donne 0.496 contre 0.449 pour [2048, 1024]. Plus de largeur aide encore : [8192] donne 0.514.
3. Un dropout plus faible après la couche cachée (0.3) aide une fois l'entrée régularisée.
4. row_norm et mixup n'apportent rien.

### Sentiment : plateau à 0.68

| Variante (base : [512, 128], dropout 0.7) | Val. acc |
|---|---|
| base | 0.6813 |
| in_dropout 0.1 / 0.2 | 0.680 / 0.679 |
| l2_norm | 0.675 |
| mixup 0.2 / 0.4 | 0.677 / 0.674 |
| label smoothing 0.2 / 0.0 | 0.675 / 0.678 |
| [256, 64] / [1024] | 0.675 / 0.675 |
| weight decay 0.05 | 0.674 |
| taux d'apprentissage 2e-4 | 0.678 |

Ensembles mesurés sur la validation (`ens_eval.py`), 3 seeds par configuration :

- un seul type de modèle : de 0.675 à 0.680 ;
- meilleure combinaison : base + [1024] + mixup, 0.684.

**Conclusion** : tous les écarts restent sous 0.8 point. C'est du bruit : sur 2 400 exemples de validation, l'écart-type est d'environ ±0.9 point, et rien qu'en changeant de seed, la base varie de 0.672 à 0.681. On garde donc la configuration de base.

### Configuration finale d'iter2

| | Image | Sentiment |
|---|---|---|
| Prétraitement | racine carrée, puis standardisation | standardisation |
| Couches cachées | **[4096]** | [512, 128] |
| Dropout sur l'entrée | **0.5** | 0 |
| Dropout caché | **0.3** | 0.7 |
| Taux d'apprentissage | 1e-3 | 5e-4 |
| Val. accuracy | **0.5085** (epoch 51) | 0.6817 (epoch 47) |

**Moyenne de validation : 0.595. Résultat Kaggle : 0.609** (+7 points par rapport à iter1).

Comparé à iter1, 53 % des prédictions image restent identiques, contre 92 % pour le sentiment.

---

## Itération 3 : transformation des features image (piste 2)

**Piste 2, transformer les features image.** On garde la configuration image d'iter2 ([4096], in_dropout 0.5, dropout 0.3) et on change seulement ce que le réseau voit en entrée.

- `tfidf` : on pondère chaque « mot visuel » par son idf, log((1+N)/(1+df)) + 1, calculé sur la partie entraînement. Les features rares pèsent ainsi plus.
- `tfidf + row_norm` : idem, puis on normalise chaque ligne pour qu'elle somme à 1.
- `pca_whiten_0.99` : PCA avec blanchiment (whitening) sur les features standardisées, en gardant 99 % de la variance (488 composantes).
- `pca_whiten_0.99, in_dropout 0.2` : idem, avec un dropout d'entrée plus faible.

| Variante | Val. acc |
|---|---|
| référence iter2 | 0.5092 |
| tfidf | 0.510 |
| tfidf + row_norm | 0.5098 |
| pca_whiten_0.99 | 0.4335 |
| pca_whiten_0.99, in_dropout 0.2 | 0.408 |

**Conclusions de la piste 2** :

1. Le **TF-IDF** ne change presque rien (0.510 contre 0.509, avec ou sans normalisation des lignes). La racine carrée suivie de la standardisation compense déjà les différences d'échelle entre features, et la pondération idf n'y ajoute pas d'information.
2. La **PCA blanchie** fait nettement moins bien (0.43, puis 0.41 avec moins de dropout d'entrée). La PCA mélange toutes les features dans chaque composante. Le dropout d'entrée, qui consistait à masquer des « mots visuels » entiers, perd alors son sens. Or c'était justement lui qui apportait le gros du gain en iter2. Le blanchiment donne aussi le même poids aux composantes de faible variance, qui sont surtout du bruit.

La piste 2 n'apporte donc pas de gain : le levier reste la régularisation et la largeur du réseau, pas la représentation des features.

### Soumission iter3

`iter3.csv` reprend la configuration d'iter2 en ajoutant le TF-IDF, la meilleure variante de la piste 2. Comme le gain en validation est nul (+0.1 point), le score Kaggle devrait être pratiquement le même qu'iter2 (≈ 0.61).

| | Image | Sentiment |
|---|---|---|
| Configuration | iter2 + tfidf | identique à iter2 |
| Val. accuracy | 0.5075 (epoch 52) | 0.6817 (epoch 47) |
| Kaggle | **0.608** (iter2 : 0.609) | |

Comparé à iter2, 91.7 % des prédictions image sont identiques. Les prédictions sentiment sont identiques à 100 %, puisque c'est la même configuration et la même seed.

---

## Itération 4 : réseau image plus large (piste 1)

**Code** : `python nn_v2.py iter4.csv`. Chaque itération a maintenant sa configuration dans `ITERS` dans `nn_v2.py`, et le script la choisit d'après le nom du fichier de sortie.

**Idée** : en iter2, élargir la couche a toujours aidé ([2048] : 0.481, puis [4096] : 0.496, puis [8192] : 0.514, toutes avec in_dropout 0.5). Allonger l'entraînement a aussi aidé ([4096] à 100 epochs : 0.508 contre 0.496). On combine les trois meilleurs réglages :

| | Image | Sentiment |
|---|---|---|
| Couches cachées | **[8192]** | [512, 128] (inchangé) |
| Dropout sur l'entrée | 0.5 | 0 |
| Dropout caché | 0.3 | 0.7 |
| Epochs max / patience | **100 / 15** | 60 / 10 |
| Val. accuracy | **0.5155** (epoch 95) | 0.6817 (epoch 47) |
| Kaggle | **0.619** (+1.0 par rapport à iter2) | |

Le meilleur epoch est le 95 sur 100 : allonger l'entraînement était utile. On gagne +0.6 point en image par rapport à iter2. Comparé à iter2, 75 % des prédictions image sont identiques ; les prédictions sentiment le sont toutes. Le run complet a pris environ 2 h 20.

## Itération 5 : augmentation de données et SWA (pas de soumission)

**Découverte sur les données** : les features image sont des **comptages entiers de « mots visuels »** normalisés en L2. Dans chaque ligne, toutes les valeurs non nulles sont des multiples de la plus petite, et la norme L2 vaut 1. Chaque image contient entre ~25 et ~110 patchs. On peut donc retrouver les comptages exacts : comptage = valeur / (plus petite valeur non nulle de la ligne).

**Idée** : le dropout sur l'entrée était déjà une forme d'augmentation de données (+6 points). On ajoute des augmentations qui ont un sens physique pour ce type de données.

- `thin` (sous-échantillonnage binomial) : à chaque epoch, chaque patch d'une image est gardé avec la probabilité 1 − thin, puis l'histogramme est renormalisé. Ça simule la même image, décrite par un autre tirage de patchs. La transformation (norme L2, racine, standardisation) est refaite en PyTorch batch par batch. On a vérifié qu'elle est identique à la version NumPy (écart < 1e-6).
- `noise` : bruit gaussien ajouté aux features standardisées.
- `swa_start` (Stochastic Weight Averaging) : à partir de cette epoch, on moyenne les poids du réseau à chaque epoch, puis on recalcule les statistiques de BatchNorm. Le modèle moyen se situe en général dans une zone plus « plate » de la loss, et il généralise mieux.

Tests sur la configuration image d'iter2 ([4096], in_dropout 0.5, dropout 0.3), référence 0.5092 :

| Variante | Val. acc |
|---|---|
| thin 0.3 | 0.4898 (epoch 59) |
| thin 0.5 | 0.442 |
| thin 0.5, in_dropout 0.3 | 0.4718 |
| noise 0.3 | 0.5045 |
| SWA à partir de l'epoch 40 | 0.5108 |

**Conclusions** :

1. **Retirer des patchs fait baisser le score**, et d'autant plus qu'on en retire : 0.490 avec thin 0.3, 0.442 avec thin 0.5. Avec 25 à 110 patchs par image, en retirer détruit une information rare. Le dropout d'entrée masque des mots visuels sur toutes les images à la fois. Le sous-échantillonnage, lui, rend chaque histogramme plus bruité, et ça ne marche pas aussi bien. Réduire le dropout d'entrée pour compenser ne suffit pas (0.472).
2. **Le bruit gaussien** ne change rien (0.5045).
3. **La SWA** donne 0.5108 contre 0.5092, soit +0.16 point, ce qui reste dans le bruit.

Aucune variante n'améliore nettement iter2. Le dropout d'entrée semble déjà apporter l'essentiel de ce qu'une augmentation de données peut faire ici. Comme convenu, on n'a pas généré `iter5.csv` : il aurait été presque identique à iter2, comme iter3.

## Analyse : la structure de voisinage des images (2026-11-12)

**Objectif fixé** : au moins **0.651** sur Kaggle. Le meilleur score du classement est 0.671, obtenu avec kNN, réseau de neurones, arbre de décision et random forest.

Dans le projet du groupe (`Comp472_Group16`), le kNN image donnait 0.46 avec **k = 1**. Qu'un seul voisin soit le meilleur réglage suggère une forte structure locale. On a vérifié avec un 1-NN et la **distance de Hellinger** (normalisation L1 de chaque histogramme, racine carrée, distance euclidienne, **sans standardisation par feature**), sur le même split 80/20 :

- 1-NN euclidien sur les features brutes : 0.531 ; **1-NN Hellinger : 0.572**, contre **0.5155 pour le meilleur MLP** (iter4).
- L'accuracy dépend fortement de la distance au plus proche voisin :

| Distance au plus proche voisin (val) | Exemples | Accuracy 1-NN |
|---|---|---|
| 5 % les plus proches | 200 | 0.985 |
| 5 à 10 % | 200 | 0.905 |
| 10 à 20 % | 400 | 0.807 |
| 20 à 30 % | 400 | 0.725 |
| 30 à 50 % | 800 | 0.657 |
| 50 % les plus éloignés | 2000 | 0.386 |

- Il n'y a pas de doublons exacts (0 entre validation et entraînement, 1 entre test et entraînement), mais beaucoup d'images ont un très proche voisin de la même classe, probablement des vues d'une même scène. Les distances test → entraînement ont la même distribution, donc l'effet devrait se retrouver sur Kaggle.

### Comparaison avec les autres modèles du groupe (Kaggle)

Même split 80/20 (stratifié, seed 42) dans le repo du groupe, donc les validations sont comparables.

| Soumission | Modèle | Val. image | Val. sentiment | Val. moyenne | Kaggle | Kaggle − val |
|---|---|---|---|---|---|---|
| rf_submission | Random forest | 0.311 | 0.606 | 0.459 | 0.478 | +1.9 |
| knn_submission | kNN (standardisation + PCA, k = 1 / cosinus, k = 41) | 0.460 | 0.584 | 0.522 | 0.544 | +2.2 |
| hybrid_submission | kNN image + RF sentiment | 0.460 | 0.606 | 0.533 | 0.562 | +2.9 |
| best_hybrid_submission | (Layla, combinaison non documentée) | | | | 0.575 | |
| nn_submission | (Layla, réseau) | | | | 0.558 | |
| iter1 | notre réseau | 0.397 | 0.673 | 0.535 | 0.539 | +0.4 |
| iter2 | notre réseau | 0.5085 | 0.6817 | 0.595 | **0.609** | +1.4 |
| iter3 | notre réseau + tfidf | 0.5075 | 0.6817 | 0.595 | 0.608 | +1.3 |

**À retenir** : Kaggle est toujours **1.5 à 3 points au-dessus** de la validation, parce que les modèles finaux voient 100 % des données. Le kNN en profite le plus, avec 25 % de voisins candidats en plus. Le kNN Hellinger (0.572 en validation image) devrait donc encore gagner sur Kaggle.

**Hypothèses** :

1. La standardisation par feature utilisée par le MLP donne trop de poids aux mots visuels rares et bruités, et casse la géométrie des histogrammes.
2. Un ensemble réseau + kNN Hellinger pourrait monter l'image vers ~0.60–0.63, donc Kaggle vers ~0.645–0.65.

**Tests prévus, non lancés** : (A) kNN Hellinger avec k et pondération réglés ; (B) MLP sur features Hellinger sans standardisation ; (C) ensemble réseau + kNN, si l'énoncé l'autorise.

## Itération 6 : kNN Hellinger pour l'image + réseau pour le sentiment

**Code** : `knn_hellinger.py` (test A), puis `make_iter6.py`. Aucun entraînement de réseau n'est nécessaire.

**Test A, kNN avec la distance de Hellinger** : on normalise chaque histogramme en L1, on prend la racine carrée, puis la distance euclidienne, sans standardisation. Les probabilités sont les votes pondérés des k plus proches voisins. Grille sur k (1 à 50) et sur la pondération (uniforme, 1/d, exp(−(d − d_min)/τ)). Résultats complets dans `knn_results_image.csv`.

| k | Pondération | Val. image |
|---|---|---|
| 1 | — | 0.572 |
| 3 / 5 / 10 | uniforme | 0.447 / 0.449 / 0.431 |
| 10 | exp, τ = 0.05 | 0.5855 |
| 20 | exp, τ = 0.02 | 0.5873 |
| **50** | **exp, τ = 0.02** | **0.5893** |

Le vote uniforme est mauvais : avec k > 1, les voisins éloignés « noient » le très proche voisin. La pondération exponentielle donne presque tout le poids aux voisins vraiment proches, tout en départageant les cas ambigus. Le kNN bat ainsi notre meilleur réseau sur l'image de **+7.4 points** (0.589 contre 0.5155).

**Soumission iter6** : image = kNN Hellinger réentraîné sur 100 % des données ; sentiment = prédictions d'iter4 (réseau, 0.682). Validation moyenne : **0.636**, contre 0.599 pour iter4. Comme Kaggle a toujours été 1.4 à 3 points au-dessus de la validation (et que le kNN profite le plus des 25 % de données en plus), on attend **≈ 0.645–0.66** sur Kaggle. Comparé à iter4, 59 % des prédictions image sont identiques.

Résultat Kaggle : **0.637**. Contrairement aux autres modèles, le kNN Hellinger n'a **pas** profité du réentraînement sur 100 % des données : 0.636 en validation pour 0.637 sur Kaggle.

### Test B : réseau sur features Hellinger sans standardisation

| Variante (base : config iter2, val 0.509) | Val. image |
|---|---|
| Hellinger (L1 + racine), sans standardisation | 0.5012 |
| idem, in_dropout 0.3 | 0.5015 |
| racine des features normalisées en L2, sans standardisation | 0.4995 |

Pour le réseau, la standardisation par feature n'est **pas** le problème : sans elle, on perd environ 1 point. L'hypothèse 1 est donc rejetée pour le réseau. Le kNN, lui, a besoin de la géométrie de Hellinger.

## Itération 7 : ensemble réseau + kNN pour l'image

**Idée** : le kNN est excellent quand il existe un voisin très proche, alors que le réseau généralise mieux sur les images « isolées ». Les deux modèles ne se trompent pas sur les mêmes exemples. On moyenne leurs probabilités : P = w · P_réseau + (1 − w) · P_kNN.

**Mesure sur la validation**, avec les probabilités sauvegardées par le test B :

| w (poids du réseau) | 0 (kNN seul) | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 1 (réseau seul) |
|---|---|---|---|---|---|---|---|
| réseau Hellinger sans std. | 0.589 | 0.606 | 0.624 | **0.627** | 0.626 | 0.612 | 0.501 |
| moyenne des 3 réseaux du test B | 0.589 | 0.607 | 0.624 | 0.632 | **0.632** | 0.620 | 0.518 |

Gain de **+4 points en image** par rapport au kNN seul.

**Code** : `python nn_v2.py iter7_nn.csv`, qui réentraîne la config iter2 avec sauvegarde des probabilités dans `ens/`. Puis `make_iter7.py` choisit w sur la validation et combine les probabilités de test. Sentiment : réseau (config iter2).

**Résultat** : le réseau réentraîné donne exactement le même score qu'iter2 (val 0.5085, epoch 51). Le poids choisi sur la validation est **w = 0.55** :

| w | 0 | 0.3 | 0.45 | 0.5 | **0.55** | 0.6 | 0.7 | 0.8 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| Val. image | 0.589 | 0.612 | 0.622 | 0.625 | **0.627** | 0.6265 | 0.623 | 0.610 | 0.5085 |

Le plateau est large (≥ 0.62 entre w = 0.45 et 0.7), donc le choix de w est robuste. **Image : 0.627** (+3.8 points par rapport au kNN seul, +11.2 par rapport au réseau seul). **Validation moyenne : 0.654**, avec le sentiment à 0.6817. Comparé à iter6, 86.6 % des prédictions image sont identiques.

Kaggle attendu : ≈ 0.65–0.655, car iter6 a montré que la partie kNN ne gagne rien entre validation et Kaggle. **Résultat Kaggle : 0.657**, au-dessus de l'objectif de 0.651.

## Diagnostic : ensemble sur le sentiment (pas de soumission)

On a utilisé les prédictions de validation des modèles du groupe (`Comp472_Group16/results/sentiment_*_predictions.npy`, même split 80/20, accuracies vérifiées).

| Modèle | Val. sentiment |
|---|---|
| **Réseau (iter2/iter7)** | **0.6817** |
| Random forest | 0.6058 |
| kNN cosinus | 0.5842 |
| Arbre de décision | 0.5033 |
| Vote majoritaire réseau + RF + kNN (égalité → réseau) | 0.6429 |
| Réseau + 0.1 × RF (one-hot) | 0.6779 |
| Réseau + 0.2 × RF + 0.1 × kNN | 0.6692 |
| Réseau + 0.5 × RF + 0.3 × kNN | 0.6458 |

Le réseau se trompe sur 31.8 % des exemples. Le RF n'en corrige que 8.5 points et le kNN 8.3, alors qu'ils introduisent davantage d'erreurs ailleurs. Les trois modèles se trompent en grande partie sur les **mêmes** exemples difficiles (RF et kNN sont d'accord à 74 %). **Toutes les combinaisons font moins bien que le réseau seul**, donc on n'est pas allé jusqu'à l'étape 2 (recalcul des probabilités du RF et du kNN). Le sentiment reste à 0.68 avec le réseau seul.

## Itération 8 : ensemble avec le réseau d'iter4

**Idée** : remplacer, dans l'ensemble d'iter7, le réseau [4096] (val 0.5085) par celui d'iter4 ([8192], 100 epochs, val 0.5155), ou utiliser la moyenne des deux réseaux. On choisit l'option et le poids du kNN sur la validation.

**Code** : `python nn_v2.py iter8_nn.csv` (config iter4, avec sauvegarde des probabilités), puis `make_iter8.py`. Le réentraînement du réseau a reproduit exactement la validation d'iter4 (0.5155, best epoch 95).

**Résultats (validation image)** :

| Option | Réseau seul | Meilleur w_nn | Ensemble |
|---|---|---|---|
| iter7 : [4096] + kNN | 0.5085 | 0.55 | 0.627 |
| [8192] + kNN | 0.5155 | 0.65 | 0.621 |
| **moyenne([4096],[8192]) + kNN** | **0.5373** | **0.70** | **0.630** |

**Analyse** :
- Remplacer seulement le réseau **fait moins bien** (0.621 < 0.627). Un réseau meilleur seul n'est pas forcément plus complémentaire du kNN.
- En revanche, **moyenner les deux réseaux** donne un réseau combiné nettement meilleur (0.537, soit +2.2 points) et l'ensemble final gagne +0.3 point en image (0.630), soit environ +0.15 point sur la moyenne (0.656 contre 0.654).
- iter8.csv est d'accord avec iter7.csv sur 94.3 % des images et sur 100 % du sentiment.
- Le gain reste faible, du même ordre que le bruit de la validation. Kaggle attendu : environ 0.657–0.660. **Kaggle obtenu : 0.659** (+0.2 point par rapport à iter7).

## Itération 9 : kNN affiné et pseudo-étiquetage (pistes 1 + 2)

**Code** : `python knn_v2.py` (environ 2 min), résultats complets dans `knn2_results_image.csv`.

Le réseau reste celui d'iter8 (moyenne des réseaux [4096] et [8192]). Seul le kNN change, et chaque réglage est jugé directement sur le score de l'ensemble en validation, pas sur le kNN seul.

**Étape 1, réglage du kNN** : on teste la puissance p appliquée aux comptages normalisés (p = 0.5 correspond à Hellinger), avec ou sans pondération idf, k jusqu'à 200 et la température τ.

| Transformation | Meilleurs k, τ | kNN seul | Ensemble |
|---|---|---|---|
| p=0.5 (Hellinger, comme avant) | 30, 0.02 | 0.589 | 0.630 |
| p=0.5 + idf | 20, 0.03 | **0.598** | 0.632 |
| p=0.3 + idf | 75, 0.02 | 0.588 | 0.633 |
| **p=0.4** | **50, 0.02** | 0.585 | **0.634** |
| p=0.7 | 200, 0.02 | 0.583 | 0.628 |

Le meilleur kNN seul (p=0.5 + idf) n'est pas le meilleur dans l'ensemble. Le gain reste faible (+0.4 point).

**Étape 2, pseudo-étiquetage** : les images non étiquetées dont la prédiction de l'ensemble est assez confiante (probabilité au-dessus d'un seuil) sont ajoutées comme voisins avec l'étiquette prédite et un poids pw. En validation, ce « test non étiqueté » est simulé par les images de validation elles-mêmes, en leave-one-out et sans jamais utiliser leurs vraies étiquettes. Le rapport entre taille du pool et taille du train est le même que pour le test (4000/16000 = 5000/20000). Meilleur réglage : seuil 0.3, pw 0.5, k=100, τ=0.02, ce qui donne un kNN seul à 0.594 et un **ensemble à 0.637** (w_nn = 0.7), soit +0.3 point.

**Bilan** : 0.637 contre 0.630 pour iter8 (+0.7 point en image, environ +0.35 sur la moyenne). Une partie de ce gain vient sans doute de la sélection sur la validation (plusieurs centaines de réglages testés). iter9.csv est d'accord avec iter8.csv sur 97.3 % des lignes.

**Diversité des réseaux** : ajouter au mélange un réseau différent, entraîné sur les features Hellinger sans standardisation (in_dropout 0.3), fait passer l'ensemble avec l'ancien kNN de 0.630 à **0.635**, alors que ce réseau est plus faible seul (0.50). Des réseaux différents apportent plus que des réseaux plus forts.

## Itération 10 : réseaux variés dans l'ensemble (piste 3)

**Idée** : ajouter au mélange des réseaux *différents* plutôt que plus forts. Trois nouveaux réseaux sont entraînés en version complète (validation + 5 seeds sur 100 %) :
- `iter10_a` : features Hellinger sans standardisation, in_dropout 0.3 ;
- `iter10_b` : deux couches [2048, 1024] ;
- `iter10_c` : [4096] avec in_dropout 0.7.

`make_iter10.py` évalue en validation tous les sous-ensembles des 5 réseaux (moyenne à poids égaux), avec la recette d'iter9 (kNN p=0.4 + pseudo-étiquetage), puis écrit iter10.csv avec le meilleur sous-ensemble. Avec les seuls réseaux iter7 et iter8, le script retrouve exactement iter9 (0.6370).

**Résultat intermédiaire (réseau a seul ajouté, `OUT=iter10a python make_iter10.py`)** :

| Réseaux moyennés | Réseaux seuls | Ensemble (kNN + pseudo-étiquetage) |
|---|---|---|
| iter7 + iter8 (= iter9) | 0.537 | 0.6370 |
| **iter7 + iter8 + a** | 0.539 | **0.6372** |
| iter8 + a | 0.537 | 0.6355 |
| a seul | 0.4945 | 0.6320 |

Le gain disparaît (+1 image sur 4000, c'est du bruit). Le +0.5 point observé avec l'ancien kNN est absorbé par le kNN affiné et le pseudo-étiquetage. Fait notable : le réseau a, le plus faible seul, est le meilleur partenaire individuel du kNN (0.632). iter10a.csv est d'accord avec iter9.csv sur 98.2 % des lignes, Soumis quand même : **Kaggle 0.660** (contre 0.661 pour iter9), ce qui confirme l'absence de gain.

**Résultat final (5 réseaux)** : validation des réseaux seuls : a 0.4945, b 0.4547, c 0.4645. Meilleur sous-ensemble : les 5 réseaux, avec un **ensemble à 0.6385** (contre 0.6370 pour iter9, soit +6 images sur 4000). Ce gain est au niveau du bruit, d'autant qu'il est choisi parmi 31 sous-ensembles. iter10.csv est d'accord avec iter9.csv sur 96.5 % des lignes. Kaggle attendu : environ 0.660–0.662, comme iter9.

**Conclusion** : l'ensemble réseaux + kNN plafonne à environ 0.637–0.638 en image (environ 0.661 sur Kaggle). Ajouter des réseaux ne fait plus progresser le score.

## Pistes restantes

La piste 2 (features) et l'augmentation de données (iter5) sont écartées. La piste 1 (image plus large) a été faite en iter4 (+0.6 point en image).

1. **Ensemble de réseaux de largeurs différentes**, par exemple [4096] + [8192], au lieu de 5 seeds du même réseau.
2. **Réentraînement final** : environ 15 % d'epochs en plus (il y a 25 % de données en plus) et 10 seeds.
3. **Validation croisée en 5 parties**, pour des décisions plus fiables, puisque les écarts sous ~0.8 point sont du bruit.
4. **Sentiment** : tester un réseau sans couche cachée (softmax linéaire), pour savoir si 0.68 est la limite des embeddings.
