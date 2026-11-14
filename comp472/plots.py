"""
Figures of the report (matplotlib, PNG in results/figures).

Each figure is a function registered with @figure; `make_all()` draws them all.
Only saved results are read (search tables, histories, saved probabilities): nothing is
retrained. Figures that need the tuned k-NN recompute its neighbours (about a minute).
"""
import itertools
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .data import SENTIMENT_NAMES, load_task  # noqa: E402
from .config import KNN_ITER9  # noqa: E402
from .ensemble import WEIGHT_GRID, PseudoLabelEnsemble, accuracy, blend, weight_curve  # noqa: E402
from .features import Hellinger  # noqa: E402
from .history import GROUP_KAGGLE, ITERATIONS, KAGGLE_TARGET, KAGGLE_TOP  # noqa: E402
from .models.knn import KNNClassifier  # noqa: E402
from .paths import FIGURES_DIR, LOGS_DIR, SEARCH_DIR, SEED_STUDY_DIR  # noqa: E402
from .recipes import RecipeContext  # noqa: E402
from .search import SEED_OFFSETS, SEED_STUDY  # noqa: E402

# ------------------------------------------------------------------ style (reference palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]   # secondary encoding (not colour alone)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, EMPHASIS_OFF = "#e6e5e0", "#fcfcfb", "#c9c8c2"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
    "font.size": 10, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})

FIGURES = {}


def figure(name, title):
    def register(fn):
        FIGURES[name] = (fn, title)
        return fn
    return register


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _hbars(ax, labels, values, highlight=None, ref=None, fmt="{:.3f}"):
    """Horizontal bars, one colour; `highlight` (indices) in slot 1, the others recessive."""
    y = np.arange(len(values))
    colors = [SERIES[0] if highlight is None or i in highlight else EMPHASIS_OFF for i in y]
    ax.barh(y, values, color=colors, height=0.7, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo = min(values)
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1)
    ax.set_xlim(lo - (max(values) - lo) * 0.25 - 1e-3, max(values) + (max(values) - lo) * 0.18 + 1e-3)


@lru_cache(maxsize=None)
def _context():
    return RecipeContext(log=lambda *a: None)


@lru_cache(maxsize=None)
def _iter9_val():
    """Validation probabilities of the iter9 components and of the final ensemble."""
    ctx = _context()
    N = ctx.nets(("iter7_nn", "iter8_nn"), "val")
    nb_va, _ = ctx.neighbourhoods
    ens = PseudoLabelEnsemble(KNN_ITER9, ctx.image.n_classes)
    val = ens.run(N, nb_va, y=ctx.y_va)
    Ka = ens.first_knn(nb_va)
    Kb = ens.second_knn(nb_va, blend(N, Ka, val.w1))
    return N, Ka, Kb, val


# ------------------------------------------------------------------ 1. overview
@figure("01_scores_iterations", "Scores par itération")
def scores_iterations():
    its = list(ITERATIONS)
    x = np.arange(len(its))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    kag = [i.kaggle if i.kaggle else np.nan for i in its]
    val = [i.val_mean if i.val_mean else np.nan for i in its]
    a1.plot(x, kag, color=SERIES[0], marker="o", label="Kaggle")
    a1.plot(x, val, color=SERIES[1], marker="s", label="Validation (moyenne des 2 tâches)")
    a1.axhline(KAGGLE_TARGET, color=INK2, linewidth=1)
    a1.axhline(KAGGLE_TOP, color=INK, linewidth=1)
    a1.text(-0.3, KAGGLE_TARGET, "objectif 0.651", va="bottom", ha="left", fontsize=8, color=INK2)
    a1.text(-0.3, KAGGLE_TOP, "meilleur score Kaggle 0.671", va="bottom", ha="left", fontsize=8, color=INK)
    best = int(np.nanargmax(kag))
    a1.annotate(f"{kag[best]:.3f}", (best, kag[best]), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color=INK)
    a1.set_xticks(x, [i.name for i in its], rotation=45, ha="right")
    a1.set_ylabel("accuracy")
    a1.set_title("Kaggle et validation")
    a1.legend(loc="lower right")

    a2.plot(x, [i.val_image for i in its], color=SERIES[0], marker="o", label="image (50 classes)")
    a2.plot(x, [i.val_sentiment if i.val_sentiment else np.nan for i in its], color=SERIES[1],
            marker="s", label="sentiment (3 classes)")
    a2.set_xticks(x, [i.name for i in its], rotation=45, ha="right")
    a2.set_ylabel("accuracy de validation (80/20)")
    a2.set_title("Validation par tâche : l'image progresse, le sentiment plafonne")
    a2.legend(loc="center right")
    fig.tight_layout()
    return fig


@figure("02_comparaison_groupe", "Comparaison avec les modèles du groupe (Kaggle)")
def group_comparison():
    ours = {f"{i.name} ({i.method})": i.kaggle for i in ITERATIONS if i.kaggle}
    items = sorted({**GROUP_KAGGLE, **ours}.items(), key=lambda kv: -kv[1])
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(8, 6))
    _hbars(ax, labels, values, highlight={i for i, l in enumerate(labels) if l in ours})
    ax.axvline(KAGGLE_TARGET, color=INK2, linewidth=1)
    ax.text(KAGGLE_TARGET, len(labels) - 0.4, " objectif", fontsize=8, color=INK2)
    ax.set_xlabel("score Kaggle")
    ax.set_title("Nos itérations (bleu) face aux autres modèles du groupe (gris)")
    return fig


# ------------------------------------------------------------------ 2. neural network
@figure("03_courbes_apprentissage_iter1", "Courbes d'apprentissage d'iter1")
def learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, task in zip(axes, ("image", "sentiment")):
        h = pd.read_csv(LOGS_DIR / f"iter1_{task}_history.csv")
        ax.plot(h.epoch, h.train_acc, color=SERIES[0], label="entraînement")
        ax.plot(h.epoch, h.val_acc, color=SERIES[1], label="validation")
        b = h.val_acc.idxmax()
        ax.scatter([h.epoch[b]], [h.val_acc[b]], color=SERIES[1], s=40, zorder=3, edgecolor=SURFACE)
        ax.annotate(f"meilleure epoch {int(h.epoch[b])} : {h.val_acc[b]:.3f}", (h.epoch[b], h.val_acc[b]),
                    textcoords="offset points", xytext=(-10, -16), ha="right", fontsize=8, color=INK2)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("accuracy")
        _pct(ax)
        ax.set_title(f"{task} : écart entraînement / validation")
        ax.legend(loc="center right")
    fig.suptitle("iter1 : l'image surapprend (100 % en entraînement, 40 % en validation)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def _search_table(name):
    return pd.read_csv(SEARCH_DIR / f"{name}.csv")


@figure("04_recherche_hp_image", "Recherches d'hyperparamètres (image)")
def hp_image():
    v1 = _search_table("hp_results_image")
    v2 = _search_table("hp2_results_image")
    v2["round"] = v2["round"].astype(str)
    panels = [("Recherche 1 (base iter1 [1024, 512])", v1, v1.loc[v1.variant == "base", "val_acc"].iloc[0]),
              ("Rounds 1-3 (base [2048])", v2[v2["round"].isin(["1", "2", "3a", "3b"])],
               v2.loc[v2.variant == "base[2048]", "val_acc"].iloc[0]),
              ("Rounds 4-6 (base iter2 [4096], in_dropout 0.5)", v2[v2["round"].isin(["4", "5", "6"])], 0.5085)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    for ax, (title, df, ref) in zip(axes, panels):
        df = df.sort_values("val_acc", ascending=False)
        labels = [f"{r.variant}" + (f"  [r{r['round']}]" if "round" in df else "") for _, r in df.iterrows()]
        best = {0}
        _hbars(ax, labels, df.val_acc.tolist(), highlight=best, ref=ref)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("accuracy de validation")
    fig.suptitle("Image : le dropout sur l'entrée et une couche large sont les vrais leviers "
                 "(trait = référence de chaque recherche)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


@figure("05_recherche_hp_sentiment", "Recherches d'hyperparamètres et seeds (sentiment)")
def hp_sentiment():
    v1 = _search_table("hp_results_sentiment").sort_values("val_acc", ascending=False)
    v2 = _search_table("hp2_results_sentiment").sort_values("val_acc", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, df, title, ref in ((axes[0], v1, "Recherche 1 (base iter1)", 0.6729),
                               (axes[1], v2, "Recherche 2 (base dropout 0.7)", 0.6813)):
        _hbars(ax, df.variant.tolist(), df.val_acc.tolist(), highlight={0}, ref=ref)
        ax.axvspan(ref - 0.009, ref + 0.009, color=GRID, alpha=0.6, zorder=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("accuracy de validation")
    # seed study: single models vs ensembles
    ax = axes[2]
    y_va = load_task("sentiment").split.y_va
    P = {(n, k): np.load(SEED_STUDY_DIR / f"{n}_s{k}.npy") for n in SEED_STUDY for k in SEED_OFFSETS}
    names = list(SEED_STUDY)
    for i, n in enumerate(names):
        singles = [accuracy(P[(n, k)], y_va) for k in SEED_OFFSETS]
        ax.scatter(singles, [i] * len(singles), color=SERIES[0], marker="o", s=30, zorder=3,
                   label="un modèle (1 seed)" if i == 0 else None, edgecolor=SURFACE)
        ens = accuracy(sum(P[(n, k)] for k in SEED_OFFSETS), y_va)
        ax.scatter([ens], [i], color=SERIES[1], marker="D", s=40, zorder=3,
                   label="ensemble des 3 seeds" if i == 0 else None, edgecolor=SURFACE)
    best = max((c for r in range(2, 5) for c in itertools.combinations(names, r)),
               key=lambda c: accuracy(sum(P[(n, k)] for n in c for k in SEED_OFFSETS), y_va))
    acc_best = accuracy(sum(P[(n, k)] for n in best for k in SEED_OFFSETS), y_va)
    ax.axvline(acc_best, color=INK2, linewidth=1)
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -0.9)
    ax.text(acc_best, -0.7, f"meilleure combinaison {acc_best:.3f} ", fontsize=8, color=INK2, ha="right")
    ax.set_xlabel("accuracy de validation")
    ax.set_title("Seeds et architectures", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("Sentiment : tous les écarts restent dans le bruit (bande grise = ±0.9 point)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 3. k-NN
@figure("06_voisinage_hellinger", "Structure de voisinage (1-NN Hellinger)")
def neighbourhood():
    img = load_task("image")
    s = img.split
    nb = KNNClassifier(Hellinger(), 1).fit(s.X_tr, s.y_tr).neighbourhood(s.X_va)
    d, correct = nb.dist[:, 0], nb.labels[:, 0] == s.y_va
    order = np.argsort(d)
    bins = np.array_split(order, 20)
    acc = [correct[b].mean() for b in bins]
    centers = (np.arange(20) + 0.5) * 5
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(centers, acc, width=4.4, color=SERIES[0], edgecolor=SURFACE)
    ax.axhline(correct.mean(), color=INK2, linewidth=1)
    ax.text(100, correct.mean(), f"moyenne {correct.mean():.3f}", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.annotate(f"{acc[0]:.1%}", (centers[0], acc[0]), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color=INK)
    ax.set_xlabel("distance au plus proche voisin (percentile, du plus proche au plus lointain)")
    ax.set_ylabel("accuracy du 1-NN")
    _pct(ax)
    ax.set_title("Quand le voisin le plus proche est très proche, il a presque toujours la bonne classe")
    return fig


@figure("07_knn_k_ponderation", "kNN Hellinger : k et pondération")
def knn_grid_plot():
    df = _search_table("knn_results_image")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, w in enumerate(["uniform", "inv", "exp0.02", "exp0.05", "exp0.1"]):
        sub = df[df.weight == w].sort_values("k")
        ax.plot(sub.k, sub.val_acc, color=SERIES[i], marker=MARKERS[i], label=w)
    b = df.iloc[df.val_acc.idxmax()]
    ax.annotate(f"k={int(b.k)}, {b.weight} : {b.val_acc:.3f}", (b.k, b.val_acc), textcoords="offset points",
                xytext=(0, -30), ha="right", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20, 50], ["1", "3", "5", "10", "20", "50"])
    ax.set_xlabel("k (nombre de voisins, échelle log)")
    ax.set_ylabel("accuracy de validation")
    ax.set_title("Pondérer les voisins de façon exponentielle permet d'utiliser beaucoup de voisins")
    ax.legend(title="pondération", loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


@figure("08_knn_geometrie", "kNN affiné : géométrie et k")
def knn_geometry():
    df = _search_table("knn2_results_image")
    s1 = df[df.step == 1]
    rows = [(p, idf) for p in sorted(s1.p.unique()) for idf in (False, True)]
    ks = sorted(s1.k.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, col, title in ((axes[0], "knn_acc", "kNN seul"), (axes[1], "ens_acc", "dans l'ensemble avec les réseaux")):
        M = np.array([[s1[(s1.p == p) & (s1.idf == idf) & (s1.k == k)][col].max() for k in ks] for p, idf in rows])
        im = ax.imshow(M, cmap=BLUES, aspect="auto")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(rows)), [f"p={p}" + (" + idf" if idf else "") for p, idf in rows])
        ax.grid(False)
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if v > np.nanpercentile(M, 70) else INK,
                    fontweight="bold" if (r, c) == (i, j) else "normal")
        ax.set_xlabel("k")
        ax.set_title(f"Accuracy {title} (meilleur τ)", fontsize=10.5)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Le meilleur kNN seul (p=0.5 + idf) n'est pas le meilleur partenaire des réseaux (p=0.4)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


@figure("09_pseudo_etiquetage", "Pseudo-étiquetage")
def pseudo_labelling():
    df = _search_table("knn2_results_image")
    s1_best = df[df.step == 1].ens_acc.max()
    s2 = df[df.step == 2]
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for i, pw in enumerate(sorted(s2.pw.unique())):
        g = s2[s2.pw == pw].groupby("thr").ens_acc.max()
        ax.plot(g.index, g.values, color=SERIES[i], marker=MARKERS[i], label=f"poids des pseudo-étiquettes {pw}")
    ax.axhline(s1_best, color=INK2, linewidth=1)
    ax.text(0.42, s1_best, f"sans pseudo-étiquetage {s1_best:.4f} ", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xlabel("seuil de confiance pour ajouter un point non étiqueté")
    ax.set_ylabel("accuracy de l'ensemble (validation)")
    ax.set_title("Ajouter les images de test confiantes comme voisins : +0.3 point")
    ax.legend(loc="lower left")
    return fig


# ------------------------------------------------------------------ 4. ensembles
@figure("10_poids_ensemble", "Poids réseau / kNN")
def ensemble_weights():
    ctx = _context()
    y = ctx.y_va
    K1 = ctx.store.load("knn")
    N, Ka, Kb, val = _iter9_val()
    curves = [("iter7 : [4096] + kNN Hellinger", ctx.store.load("iter7_nn"), K1),
              ("[8192] + kNN Hellinger", ctx.store.load("iter8_nn"), K1),
              ("iter8 : moyenne des 2 réseaux + kNN Hellinger", N, K1),
              ("iter9 : moyenne des 2 réseaux + kNN affiné + pseudo-étiq.", N, Kb)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (label, Nn, K) in enumerate(curves):
        accs = weight_curve(Nn, K, y)
        ax.plot(WEIGHT_GRID, accs, color=SERIES[i], marker=MARKERS[i], markersize=4, label=label)
        j = int(np.argmax(accs))
        ax.scatter([WEIGHT_GRID[j]], [accs[j]], color=SERIES[i], s=70, zorder=3, edgecolor=INK, linewidth=0.8)
    ax.set_xlabel("poids du réseau w  (0 = kNN seul, 1 = réseau seul)")
    ax.set_ylabel("accuracy image (validation)")
    ax.set_title("Le mélange bat largement chaque modèle seul (point cerclé = poids retenu)")
    ax.legend(loc="lower center", fontsize=8)
    return fig


@figure("11_complementarite", "Complémentarité réseau / kNN")
def complementarity():
    ctx = _context()
    y = ctx.y_va
    N, Ka, Kb, val = _iter9_val()
    n_ok, k_ok = N.argmax(1) == y, Kb.argmax(1) == y
    e_ok = val.proba.argmax(1) == y
    cats = [("les deux justes", n_ok & k_ok), ("réseau seul juste", n_ok & ~k_ok),
            ("kNN seul juste", ~n_ok & k_ok), ("les deux faux", ~n_ok & ~k_ok)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    labels = [c for c, _ in cats]
    _hbars(a1, labels, [m.mean() for _, m in cats], fmt="{:.1%}")
    _pct(a1, "x")
    a1.set_title("Réseaux (moyenne) et kNN affiné, images de validation", fontsize=10.5)
    rescued = [e_ok[m].mean() if m.any() else 0 for _, m in cats[1:3]]
    _hbars(a2, ["réseau seul juste", "kNN seul juste"], rescued, fmt="{:.1%}")
    _pct(a2, "x")
    a2.set_xlim(0, 1)
    a2.set_title("Part de ces désaccords que l'ensemble tranche correctement", fontsize=10.5)
    fig.suptitle(f"Les deux modèles se trompent sur des images différentes : réseaux {n_ok.mean():.3f}, "
                 f"kNN {k_ok.mean():.3f}, ensemble {e_ok.mean():.3f}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


@figure("12_confiance", "Confiance de l'ensemble")
def confidence():
    ctx = _context()
    y = ctx.y_va
    *_, val = _iter9_val()
    conf, ok = val.proba.max(1), val.proba.argmax(1) == y
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
    acc = [ok[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    centers = edges[:-1] + 0.05
    a1.bar(centers, acc, width=0.088, color=SERIES[0], edgecolor=SURFACE, label="accuracy observée")
    a1.plot([0, 1], [0, 1], color=INK2, linewidth=1, label="calibration parfaite")
    a1.set_xlabel("confiance de l'ensemble (probabilité max.)")
    a1.set_ylabel("accuracy")
    _pct(a1)
    a1.legend(loc="upper left")
    a1.set_title("Accuracy par tranche de confiance", fontsize=10.5)
    thr = np.linspace(0, 0.95, 20)
    a2.plot(thr, [(conf >= t).mean() for t in thr], color=SERIES[0], marker="o", markersize=4,
            label="part des images au-dessus du seuil")
    a2.plot(thr, [ok[conf >= t].mean() for t in thr], color=SERIES[1], marker="s", markersize=4,
            label="accuracy de ces images")
    a2.axvline(0.3, color=INK2, linewidth=1)
    a2.text(0.3, 0.05, " seuil retenu 0.3", fontsize=8, color=INK2)
    a2.set_xlabel("seuil de confiance")
    _pct(a2)
    a2.legend(loc="center right")
    a2.set_title("Choix du seuil de pseudo-étiquetage", fontsize=10.5)
    fig.tight_layout()
    return fig


@figure("13_accuracy_par_classe", "Accuracy par classe (image)")
def per_class():
    ctx = _context()
    y = ctx.y_va
    N, Ka, Kb, val = _iter9_val()
    nc = ctx.image.n_classes
    series = [("réseaux (moyenne)", N), ("kNN affiné + pseudo-étiq.", Kb), ("ensemble iter9", val.proba)]
    accs = {n: np.array([(P.argmax(1)[y == c] == c).mean() for c in range(nc)]) for n, P in series}
    order = np.argsort(-accs["ensemble iter9"])
    fig, ax = plt.subplots(figsize=(14, 4.6))
    x = np.arange(nc)
    for i, (n, _) in enumerate(series):
        ax.scatter(x, accs[n][order], color=SERIES[i], marker=MARKERS[i], s=26, label=n, zorder=3,
                   edgecolor=SURFACE, linewidth=0.5)
    ax.set_xticks(x, order, fontsize=7)
    ax.set_xlabel("classe (triée par accuracy de l'ensemble)")
    ax.set_ylabel("accuracy de validation")
    _pct(ax)
    ax.set_title("Certaines classes restent difficiles pour tous les modèles")
    ax.legend(loc="upper right")
    return fig


@figure("14_sous_ensembles_iter10", "Sous-ensembles de réseaux (iter10)")
def subsets():
    df = _search_table("iter10_subsets").sort_values("val_acc", ascending=False)
    short = {"iter7_nn": "[4096]", "iter8_nn": "[8192]", "iter10_a": "a", "iter10_b": "b", "iter10_c": "c"}
    labels = [" + ".join(short[n] for n in s.split("+")) for s in df.nets]
    fig, ax = plt.subplots(figsize=(9, 8))
    iter9 = labels.index("[4096] + [8192]")
    _hbars(ax, labels, df.val_acc.tolist(), highlight={0, iter9}, ref=0.637, fmt="{:.4f}")
    ax.set_xlabel("accuracy image (validation), kNN affiné + pseudo-étiquetage")
    ax.set_title("Ajouter des réseaux ne fait plus progresser : les meilleurs sous-ensembles sont à moins de 0.2 point d'iter9\n"
                 "(a = Hellinger sans standardisation, b = [2048, 1024], c = in_dropout 0.7 ; trait = iter9)",
                 fontsize=10.5)
    return fig


@figure("15_confusion_sentiment", "Matrice de confusion (sentiment)")
def sentiment_confusion():
    data = load_task("sentiment")
    y = data.split.y_va
    P = _context().store.load("iter8_nn", "sentiment", "val")
    cm = confusion_matrix(y, P.argmax(1))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(cmn):
        ax.text(c, r, f"{v:.0%}\n({cm[r, c]})", ha="center", va="center", fontsize=9,
                color=SURFACE if v > 0.5 else INK)
    ax.set_xticks(range(3), SENTIMENT_NAMES)
    ax.set_yticks(range(3), SENTIMENT_NAMES)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("vraie classe")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Sentiment (réseau, val {accuracy(P, y):.3f}) : la classe neutre est la plus confondue",
                 fontsize=10)
    return fig


def make_all(names=None):
    paths = []
    for name, (fn, title) in FIGURES.items():
        if names and name not in names:
            continue
        paths.append(_save(fn(), name))
        print(f"  {paths[-1].name}  ({title})", flush=True)
    return paths
