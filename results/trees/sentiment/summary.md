# Trees error analysis - sentiment

DT params: `{'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'}`  
RF params: `{'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.2, 'n_estimators': 165}`

## DT
- train accuracy: **0.5127**, validation accuracy: **0.5033** (gap +0.0094)
- macro F1 (validation): 0.4913
- fit time 2.0s, predict time 0.00s for 2400 samples; size: 31 nodes, 16 leaves, depth 4
- worst classes by recall: Neutral (0.29), Positive (0.56), Negative (0.65)
- best classes by recall: Negative (0.65), Positive (0.56), Neutral (0.29)
- most confused pairs (true -> predicted, count): Neutral->Negative (310), Neutral->Positive (254), Positive->Negative (181), Positive->Neutral (168), Negative->Positive (156)
- per-class recall: Negative 0.651, Neutral 0.295, Positive 0.564

## RF
- train accuracy: **1.0000**, validation accuracy: **0.6058** (gap +0.3942)
- macro F1 (validation): 0.6044
- fit time 282.6s, predict time 0.46s for 2400 samples; size: 165 trees, 385351 nodes in total, OOB accuracy 0.5944
- worst classes by recall: Neutral (0.51), Positive (0.60), Negative (0.70)
- best classes by recall: Negative (0.70), Positive (0.60), Neutral (0.51)
- most confused pairs (true -> predicted, count): Neutral->Negative (226), Positive->Neutral (205), Neutral->Positive (166), Negative->Neutral (151), Positive->Negative (111)
- per-class recall: Negative 0.703, Neutral 0.510, Positive 0.605

## DT vs RF
| case | count | share |
|---|---|---|
| both correct | 958 | 39.92% |
| only RF correct | 496 | 20.67% |
| only DT correct | 250 | 10.42% |
| both wrong | 696 | 29.00% |
| both wrong with the SAME wrong label | 516 | 21.50% |

Of the samples the DT gets wrong, the RF fixes 41.6%.

## Comparison with k-NN
k-NN validation accuracy: 0.5842  
- DT: both correct 887, only k-NN correct 515, only DT correct 321, both wrong 677 (same wrong label: 478)
- RF: both correct 1183, only k-NN correct 219, only RF correct 271, both wrong 727 (same wrong label: 596)

## Confidence (RF)
- mean confidence when correct: 0.506, when wrong: 0.451

## Feature importance
- DT: 12/768 features used at least once; 3 features give 50% of the importance, 9 give 90%. Top 5 features: feature_42, feature_594, feature_657, feature_532, feature_164
- RF: 768/768 features used at least once; 268 features give 50% of the importance, 652 give 90%. Top 5 features: feature_42, feature_594, feature_164, feature_12, feature_607
