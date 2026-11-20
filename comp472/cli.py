"""
Command line: python -m comp472 <command> ...

  list                       experiments, searches, recipes and figures available
  train <experiment>         train an MLP experiment (iter1 ... iter10_c), save probabilities
                             and results/submissions/<experiment>.csv
  search <name>              hyperparameter search (image-v1, image-1 ... sentiment-1)
  seed-study                 seed / architecture ensembles on sentiment
  knn-grid                   first Hellinger k-NN (probabilities "knn", iter6-8)
  knn-search                 tuned k-NN + pseudo-labelling search (settings of iter9)
  submit <recipe>            build an ensemble submission (iter6 ... iter11)
  plots [name ...]           draw the figures in results/figures
  verify [--fast]            check that the saved results are reproduced exactly
"""
import argparse
import os
import sys

import torch


def _threads(n):
    torch.set_num_threads(n)


def cmd_list(args):
    from .config import EXPERIMENTS
    from .plots import FIGURES
    from .recipes import RECIPES
    from .search import SEARCHES
    print("Experiments (train):")
    for e in EXPERIMENTS.values():
        print(f"  {e.name:10s} {e.description}")
    print("Searches (search):")
    for name, s in SEARCHES.items():
        print(f"  {name:13s} {len(s.variants)} variants -> results/search/{s.table}.csv")
    print("Recipes (submit):")
    for name, (_, d) in RECIPES.items():
        print(f"  {name:8s} {d}")
    print("Figures (plots):")
    for name, (_, t) in FIGURES.items():
        print(f"  {name}  {t}")


def cmd_train(args):
    from .config import EXPERIMENTS
    from .experiment import NNExperimentRunner
    NNExperimentRunner(record_history=not args.no_history).run(EXPERIMENTS[args.experiment])


def cmd_search(args):
    from .search import run_search
    run_search(args.name)


def cmd_seed_study(args):
    from .search import seed_study
    seed_study()


def cmd_knn_grid(args):
    from .search import knn_grid
    knn_grid()


def cmd_knn_search(args):
    from .search import knn_search
    knn_search()


def cmd_submit(args):
    import pandas as pd
    from .paths import SEARCH_DIR
    from .recipes import make_submission
    _, res = make_submission(args.recipe)
    if "subsets" in res.details:
        pd.DataFrame(res.details["subsets"]).round(4).to_csv(SEARCH_DIR / f"{args.recipe}_subsets.csv", index=False)


def cmd_plots(args):
    from .plots import make_all
    make_all(args.names or None)


def cmd_verify(args):
    from .verify import verify_all
    sys.exit(0 if verify_all(fast=args.fast) else 1)


def main(argv=None):
    from .config import EXPERIMENTS
    from .paths import ensure_dirs
    parser = argparse.ArgumentParser(prog="python -m comp472", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NTHREADS", 8)),
                        help="torch CPU threads (default 8, or $NTHREADS)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("train")
    p.add_argument("experiment", choices=sorted(EXPERIMENTS))
    p.add_argument("--no-history", action="store_true", help="do not record learning curves")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("search")
    p.add_argument("name")
    p.set_defaults(fn=cmd_search)
    sub.add_parser("seed-study").set_defaults(fn=cmd_seed_study)
    sub.add_parser("knn-grid").set_defaults(fn=cmd_knn_grid)
    sub.add_parser("knn-search").set_defaults(fn=cmd_knn_search)
    p = sub.add_parser("submit")
    p.add_argument("recipe")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("plots")
    p.add_argument("names", nargs="*")
    p.set_defaults(fn=cmd_plots)
    p = sub.add_parser("verify")
    p.add_argument("--fast", action="store_true", help="only rebuild the submissions")
    p.set_defaults(fn=cmd_verify)
    args = parser.parse_args(argv)
    _threads(args.threads)
    ensure_dirs()
    args.fn(args)
