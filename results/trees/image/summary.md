# Trees error analysis - image

DT params: `{'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 20, 'criterion': 'gini'}`  
RF params: `{'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_estimators': 182}`

## DT
- train accuracy: **0.2601**, validation accuracy: **0.1197** (gap +0.1404)
- macro F1 (validation): 0.1191
- fit time 26.0s, predict time 0.01s for 4000 samples; size: 1245 nodes, 623 leaves, depth 53
- worst classes by recall: 31 (0.00), 29 (0.01), 9 (0.03), 17 (0.03), 24 (0.03)
- best classes by recall: 46 (0.53), 5 (0.45), 2 (0.42), 45 (0.30), 48 (0.28)
- most confused pairs (true -> predicted, count): 3.0->4.0 (12.0), 37.0->39.0 (11.0), 4.0->2.0 (10.0), 2.0->4.0 (9.0), 38.0->25.0 (9.0)

## RF
- train accuracy: **1.0000**, validation accuracy: **0.3110** (gap +0.6890)
- macro F1 (validation): 0.2957
- fit time 747.8s, predict time 2.18s for 4000 samples; size: 182 trees, 2700920 nodes in total, OOB accuracy 0.2653
- worst classes by recall: 29 (0.05), 9 (0.07), 24 (0.10), 22 (0.11), 21 (0.12)
- best classes by recall: 46 (0.78), 2 (0.68), 5 (0.62), 13 (0.60), 48 (0.57)
- most confused pairs (true -> predicted, count): 4.0->2.0 (17.0), 37.0->39.0 (13.0), 38.0->39.0 (10.0), 17.0->47.0 (9.0), 6.0->7.0 (9.0)

## DT vs RF
| case | count | share |
|---|---|---|
| both correct | 369 | 9.22% |
| only RF correct | 875 | 21.88% |
| only DT correct | 110 | 2.75% |
| both wrong | 2646 | 66.15% |
| both wrong with the SAME wrong label | 323 | 8.08% |

Of the samples the DT gets wrong, the RF fixes 24.9%.

## Comparison with k-NN
k-NN validation accuracy: 0.4600  
- DT: both correct 330, only k-NN correct 1510, only DT correct 149, both wrong 2011 (same wrong label: 86)
- RF: both correct 915, only k-NN correct 925, only RF correct 329, both wrong 1831 (same wrong label: 156)

## Confidence (RF)
- mean confidence when correct: 0.166, when wrong: 0.073

## Feature importance
- DT: 279/500 features used at least once; 39 features give 50% of the importance, 182 give 90%. Top 5 features: feature_405, feature_230, feature_120, feature_454, feature_91
- RF: 500/500 features used at least once; 190 features give 50% of the importance, 414 give 90%. Top 5 features: feature_425, feature_319, feature_348, feature_269, feature_230
