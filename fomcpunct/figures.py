"""Figures for the README and the slides (static PNG, light background).

Usage:
    python -m fomcpunct.figures [--split validation] [--focus bilstm_crf_s0] [--reference crf]

Reads results/<split>_metrics.json (run fomcpunct.evaluate first), the saved
predictions, results/training/*.json and data/processed/stats.json, and writes
PNG files to results/figures/. Every model in the metrics file is drawn;
models trained with several seeds (names ending in _s<seed>) form one family.

Colors follow the model family in a fixed order (never by rank), from a
color-vision-deficiency-checked categorical palette; every mark also has a text
label, so color is never the only cue.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import LABELS, PUNCT_LABELS  # noqa: E402
from .error_analysis import QUESTION_START_WINDOW, comma_kinds, questions, turn_length  # noqa: E402
from .evaluate import FEW_CLUSTERS, SEEDED, load_predictions  # noqa: E402
from .run_baselines import load_split  # noqa: E402

RESULTS, DATA = Path("results"), Path("data/processed")
OUT = RESULTS / "figures"

# Categorical palette (light mode), fixed slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_ORDER = ["heuristic", "hmm", "crf", "bilstm_softmax", "bilstm_crf", "bert", "deberta"]
FAMILY_NAME = {"heuristic": "Heuristic", "hmm": "HMM + Viterbi", "crf": "CRF", "bilstm_softmax": "BiLSTM (softmax)",
               "bilstm_crf": "BiLSTM-CRF", "bert": "BERT-base", "deberta": "DeBERTa-v3-base"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
CLASS_COLOR = dict(zip(PUNCT_LABELS, SLOTS[:3]))  # COMMA, PERIOD, QUESTION: a separate 3-series context

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def family(model: str) -> str:
    match = SEEDED.match(model)
    return match.group(1) if match else model


def color(fam: str) -> str:
    return SLOTS[FAMILY_ORDER.index(fam)] if fam in FAMILY_ORDER else MUTED


def name(model_or_family: str) -> str:
    fam = family(model_or_family)
    match = SEEDED.match(model_or_family)
    base = FAMILY_NAME.get(fam, fam)
    return f"{base} seed {match.group(2)}" if match else base


def ordered_families(models: list[str]) -> list[str]:
    fams = list(dict.fromkeys(family(m) for m in models))
    return sorted(fams, key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else len(FAMILY_ORDER))


def members(models: list[str], fam: str) -> list[str]:
    return [m for m in models if family(m) == fam]


def save(fig, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f"  wrote {OUT / filename}")


def forest(report: dict) -> None:
    """Macro-F1 with 95% CI for every model (rows), all words and within-turn (panels)."""
    models = list(report["models"])
    rows = [m for fam in ordered_families(models) for m in members(models, fam)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, view, title in zip(axes, ("all_words", "within_turn"), ("All words", "Within-turn words")):
        for yi, m in zip(y, rows):
            ci = report["models"][m][view]["macro_f1_ci"]
            c = color(family(m))
            ax.errorbar(ci["point"], yi, xerr=[[ci["point"] - ci["low"]], [ci["high"] - ci["point"]]], fmt="none",
                        ecolor=c, elinewidth=1.5, capsize=5, capthick=1.5)
            ax.plot(ci["point"], yi, "o", markersize=5, color=c)
            ax.annotate(f"{ci['point']:.3f}", (ci["high"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Macro-F1 (COMMA, PERIOD, QUESTION), 95% CI")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [name(m) for m in rows])
    fig.suptitle(f"Punctuation restoration, {report['split']} split ({report['documents']} press conferences, "
                 f"{report['words']:,} words)", x=0.01, y=1.04, ha="left", fontsize=12, fontweight="bold")
    note = f"CIs: percentile bootstrap over the {report['documents']} press conferences."
    if report["documents"] < FEW_CLUSTERS:
        note += " With this few conferences they are too narrow; read them as indicative."
    fig.text(0.01, -0.02, note, fontsize=8, color=MUTED)
    save(fig, f"{report['split']}_macro_f1.png")


def family_mean(report: dict, models: list[str], getter) -> float:
    return statistics.mean(getter(report["models"][m]) for m in models)


def per_class(report: dict) -> None:
    """Per-class F1 by model family (mean over seeds), all words."""
    models = list(report["models"])
    fams = ordered_families(models)
    x = np.arange(len(fams))
    width = 0.26
    fig, ax = plt.subplots(figsize=(1.35 * len(fams) + 2, 4))
    for k, label in enumerate(PUNCT_LABELS):
        values = [family_mean(report, members(models, f), lambda r, l=label: r["all_words"]["per_class"][l]["f1"])
                  for f in fams]
        ax.bar(x + (k - 1) * (width + 0.02), values, width, color=CLASS_COLOR[label], label=label,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [FAMILY_NAME.get(f, f) for f in fams], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("F1 by punctuation mark (mean over seeds where there are several)")
    ax.legend(ncols=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, f"{report['split']}_per_class_f1.png")


def speed(report: dict, skip: tuple[str, ...] = ()) -> None:
    """Macro-F1 against prediction time per 1,000 words (log scale), one point per family.

    `skip` leaves out families whose timing is known to be unusable (e.g. measured on battery power).
    """
    models = list(report["models"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    gpu_seen = False
    placed: list[tuple[float, float]] = []
    for fam in ordered_families(models):
        ms = [report["models"][m].get("latency", {}) for m in members(models, fam)]
        if fam in skip or not all("ms_per_1000_words" in x for x in ms):
            continue
        cpu = all(str(x.get("device", "cpu")).startswith("cpu") for x in ms)
        gpu_seen |= not cpu
        xv = statistics.median(x["ms_per_1000_words"] for x in ms)
        yv = family_mean(report, members(models, fam), lambda r: r["all_words"]["macro_f1"])
        ax.plot(xv, yv, "o", markersize=9, color=SLOTS[0] if cpu else SURFACE,
                markeredgecolor=SLOTS[0], markeredgewidth=2)
        near = [py for px, py in placed if abs(np.log10(xv / px)) < 0.6 and abs(yv - py) < 0.06]
        offset = (8, -3) if not near else (8, 8) if yv > max(near) else (8, -16)  # above a lower neighbour, else below
        ax.annotate(FAMILY_NAME.get(fam, fam) + ("" if cpu else " (GPU)"), (xv, yv),
                    xytext=offset, textcoords="offset points", fontsize=9, color=INK)
        placed.append((xv, yv))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Milliseconds per 1,000 words (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Accuracy against prediction time")
    note = "CPU: one process, one thread, median of 5 runs; compare orders of magnitude only."
    if gpu_seen:
        note += " Hollow points: GPU, batched (not comparable with CPU)."
    if skip:
        note += f"\nNot shown (timing to be repeated): {', '.join(FAMILY_NAME.get(f, f) for f in skip)}."
    fig.text(0.01, -0.01, note, fontsize=8, color=MUTED, va="top")
    save(fig, f"{report['split']}_speed.png")


def labels_chart() -> None:
    """Label shares per split (punctuation labels; O is given in the subtitle)."""
    stats = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))["splits"]
    splits = ["train", "validation", "test"]
    x = np.arange(len(PUNCT_LABELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 3.8))
    shades = ["#184f95", "#3987e5", "#86b6ef"]  # one hue, three steps: the splits are ordered in time
    for k, split in enumerate(splits):
        shares = [100 * stats[split]["label_share"][label] for label in PUNCT_LABELS]
        bars = ax.bar(x + (k - 1) * (width + 0.02), shares, width, color=shades[k], edgecolor=SURFACE,
                      label=f"{split} ({stats[split]['words']:,} words)")
        for bar, share in zip(bars, shares):
            ax.annotate(f"{share:.1f}", (bar.get_x() + bar.get_width() / 2, share), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(x, PUNCT_LABELS)
    ax.set_ylabel("Share of words (%)")
    o_share = ", ".join(f"{s} {100 * stats[s]['label_share']['O']:.1f}%" for s in splits)
    ax.set_title("Punctuation labels per split")
    fig.text(0.01, -0.03, f"The rest of the words carry no mark (O): {o_share}.", fontsize=8, color=MUTED)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "label_distribution.png")


def confusions(report: dict, focus: str, reference: str) -> None:
    """Row-normalized confusion matrices for two models (within-turn words are in the JSON)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#3987e5", "#104281"])
    for ax, model in zip(axes, (reference, focus)):
        cm = np.array(report["models"][model]["all_words"]["confusion"], dtype=float)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for g in range(4):
            for p in range(4):
                ax.text(p, g, f"{100 * share[g, p]:.1f}%\n{int(cm[g, p]):,}", ha="center", va="center",
                        fontsize=8, color=SURFACE if share[g, p] > 0.55 else INK)
        ax.set_xticks(range(4), LABELS)
        ax.set_yticks(range(4), LABELS)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Gold")
        ax.set_title(name(model))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.01, -0.02, "Each row sums to 100%: where the gold label's words went.", fontsize=8, color=MUTED)
    save(fig, f"{report['split']}_confusion.png")


def learning_curves() -> None:
    """Validation macro-F1 per epoch for every model with a per-epoch history."""
    runs = {}
    for path in sorted((RESULTS / "training").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("history")
        if isinstance(history, list) and history and "validation_macro_f1" in history[0]:
            best = data.get("summary", data).get("best_epoch")
            runs[path.stem] = (history, best)
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = ["-", "--", ":"]
    for fam in ordered_families(list(runs)):
        for k, run in enumerate(members(list(runs), fam)):
            history, best = runs[run]
            epochs = [h["epoch"] for h in history]
            scores = [h["validation_macro_f1"] for h in history]
            ax.plot(epochs, scores, styles[k % 3], color=color(fam), linewidth=1.6,
                    label=FAMILY_NAME.get(fam, fam) if k == 0 else None)
            if best:
                ax.plot(best, scores[epochs.index(best)], "o", markersize=7, color=color(fam),
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Learning curves (one line per seed; dot = epoch kept, best validation macro-F1)")
    ax.legend(loc="lower right")
    save(fig, "learning_curves.png")


def error_views(report: dict) -> None:
    """Three error views per family (seed 0 of each family, or the single model; missing values are not drawn)."""
    split = report["split"]
    examples = load_split(split)
    models = list(report["models"])
    def seed0(fam: str) -> str:
        group = members(models, fam)
        return next((m for m in group if SEEDED.match(m) and SEEDED.match(m).group(2) == "0"), group[0])

    picks = {fam: seed0(fam) for fam in ordered_families(models)}
    preds = {m: load_predictions(m, split, examples) for m in picks.values()}
    lengths = {m: turn_length(examples, p) for m, p in preds.items()}
    commas = comma_kinds(examples, preds)
    qs = questions(examples, preds, next(iter(preds)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    rows0 = next(iter(lengths.values()))
    keep = [i for i in range(len(rows0)) if any(lengths[m][i]["macro_f1_comma_period"] is not None for m in lengths)]
    for fam, m in picks.items():
        values = [lengths[m][i]["macro_f1_comma_period"] for i in keep]
        xs = [k for k, v in enumerate(values) if v is not None]
        ax.plot(xs, [values[k] for k in xs], "-o", color=color(fam), markersize=6, markeredgecolor=SURFACE,
                label=FAMILY_NAME.get(fam, fam))
    ax.set_xticks(range(len(keep)), [f"{rows0[i]['bucket']}\n{rows0[i]['turns']} turns" for i in keep], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Turn length (words)")
    ax.set_ylabel("Within-turn macro-F1 (COMMA, PERIOD)")
    ax.set_title("By turn length")

    short = {"written ','": "written ','", "repeated word (any mark)": "repeated\nword",
             "wh-word first": "wh-word\nfirst", "auxiliary verb first": "auxiliary\nfirst",
             f"wh-word or auxiliary within the first {QUESTION_START_WINDOW} words":
                 f"wh/auxiliary in\nfirst {QUESTION_START_WINDOW} words",
             f"no wh-word or auxiliary in the first {QUESTION_START_WINDOW} words":
                 f"none in first\n{QUESTION_START_WINDOW} words"}
    for ax, rows, title in ((axes[1], commas, "Comma recall by transcript mark"),
                            (axes[2], [{"kind": k, **v} for k, v in qs["by_start"].items()],
                             "Question recall by how the question starts")):
        x = np.arange(len(rows))
        width = 0.8 / len(picks)
        for k, (fam, m) in enumerate(picks.items()):
            pos = [(xi - 0.4 + width * (k + 0.5), r["recall"][m]) for xi, r in zip(x, rows) if r["recall"][m] is not None]
            ax.bar([p for p, _ in pos], [v for _, v in pos], width * 0.92, color=color(fam), edgecolor=SURFACE)
        ax.set_xticks(x, [f"{short.get(r['kind'], r['kind'])}\nn={r.get('gold_commas', r.get('gold')):,}"
                          for r in rows], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles = [matplotlib.patches.Patch(color=color(fam), label=FAMILY_NAME.get(fam, fam)) for fam in picks]
    fig.legend(handles=handles, ncols=len(handles), loc="upper left", bbox_to_anchor=(0.01, 1.08))
    fig.text(0.01, -0.08, "Seed 0 of each family. Turn length: turns under 31 words have too few within-turn "
             "PERIOD labels. Comma marks exclude repeated words (own group). Questions: after lead-in words "
             "such as \"and\", \"so\", \"well\".", fontsize=8, color=MUTED)
    save(fig, f"{split}_error_views.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--focus", default="bilstm_crf_s0")
    parser.add_argument("--reference", default="crf")
    parser.add_argument("--skip-latency", nargs="*", default=[],
                        help="model families to leave out of the speed chart (timing to be repeated)")
    args = parser.parse_args()
    report = json.loads((RESULTS / f"{args.split}_metrics.json").read_text(encoding="utf-8"))
    forest(report)
    per_class(report)
    if args.split == "validation":  # latency is reported from the validation timing passes only
        speed(report, tuple(args.skip_latency))
    labels_chart()
    if args.focus in report["models"] and args.reference in report["models"]:
        confusions(report, args.focus, args.reference)
    learning_curves()
    error_views(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
