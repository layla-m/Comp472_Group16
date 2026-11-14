"""Live progress counter for hp_search.py.  Usage: python hp_progress.py"""
import ast
import os
import re
import time

from hp_search import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))


def status(task):
    path = os.path.join(HERE, f"hp_{task}.log")
    if not os.path.exists(path):
        return None
    done = []
    for line in open(path):
        if line.startswith("{"):
            done.append(ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", line)))
    total = len(VARIANTS[task])
    n = len(done)
    elapsed_current = time.time() - os.path.getmtime(path)  # time since last finished config
    avg = sum(d["seconds"] for d in done) / n if n else 180
    remaining = max(0, (total - n) * avg - (elapsed_current if n < total else 0))
    best = max(done, key=lambda d: d["val_acc"]) if done else None
    return n, total, remaining, best


def bar(n, total, width=30):
    k = int(width * n / total)
    return "#" * k + "-" * (width - k)


while True:
    lines, all_done = [], True
    for task in ("image", "sentiment"):
        st = status(task)
        if st is None:
            continue
        n, total, rem, best = st
        all_done &= n == total
        best_txt = f"best={best['variant']} ({best['val_acc']:.4f})" if best else ""
        eta = "terminé" if n == total else f"~{int(rem // 60)} min {int(rem % 60):02d} s restantes"
        lines.append(f"{task:10s} [{bar(n, total)}] {n}/{total}  {eta}  {best_txt}")
    print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "\n" + "\n".join(lines), flush=True)
    if all_done:
        break
    time.sleep(5)
