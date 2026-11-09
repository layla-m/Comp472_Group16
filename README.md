# Comp472

## Decision Tree and Random Forest 

Run from the project root, after `python prepare_data.py`:

    python -m experiments.tree_unit_tests
    python -m experiments.tune_decision_tree --task both
    python -m experiments.tune_random_forest --task both
    python -m experiments.error_analysis_trees --task both
    python submit_trees.py

Results are written to results/trees/<task>/ (see summary.md). Seed: 42.
