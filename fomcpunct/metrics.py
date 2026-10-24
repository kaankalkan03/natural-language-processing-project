"""Scores, confidence intervals and paired comparisons.

Main metric: macro-F1 over COMMA, PERIOD and QUESTION. About 85% of words
carry no mark (O), so accuracy would reward predicting no punctuation at all.

Confidence intervals use a cluster bootstrap over press conferences: words in
one conference share a speaker and topic, so they are not independent samples.
Each resample draws conferences with replacement and sums their confusion
matrices; the 95% interval is the 2.5th-97.5th percentile of the metric.
"""

from __future__ import annotations

import numpy as np

from . import LABEL2ID, LABELS, PUNCT_LABELS

N = len(LABELS)
PUNCT_IDS = [LABEL2ID[label] for label in PUNCT_LABELS]


def confusion(gold: list[list[str]], pred: list[list[str]]) -> np.ndarray:
    """4x4 matrix: rows are gold labels, columns are predicted labels."""
    cm = np.zeros((N, N), dtype=np.int64)
    for g_seq, p_seq in zip(gold, pred, strict=True):
        if len(g_seq) != len(p_seq):
            raise ValueError(f"length mismatch: {len(g_seq)} gold vs {len(p_seq)} predicted")
        g = np.fromiter((LABEL2ID[x] for x in g_seq), dtype=np.int64, count=len(g_seq))
        p = np.fromiter((LABEL2ID[x] for x in p_seq), dtype=np.int64, count=len(p_seq))
        np.add.at(cm, (g, p), 1)
    return cm


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)


def class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 per label; works on one matrix (4x4) or a stack (Bx4x4). F1 = 2TP / (pred + gold)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return _safe_div(2 * tp, cm.sum(axis=-2) + cm.sum(axis=-1))


def macro_f1(cm: np.ndarray) -> np.ndarray:
    return class_f1(cm)[..., PUNCT_IDS].mean(axis=-1)


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm)
    pred, gold = cm.sum(axis=0), cm.sum(axis=1)
    per_class = {
        label: {
            "precision": float(_safe_div(tp[i], pred[i])),
            "recall": float(_safe_div(tp[i], gold[i])),
            "f1": float(_safe_div(2 * tp[i], pred[i] + gold[i])),
            "support": int(gold[i]),
        }
        for i, label in enumerate(LABELS)
    }
    p_tp, p_pred, p_gold = tp[PUNCT_IDS].sum(), pred[PUNCT_IDS].sum(), gold[PUNCT_IDS].sum()
    # Mark detection: was any punctuation predicted where there is any punctuation?
    detected = cm[np.ix_(PUNCT_IDS, PUNCT_IDS)].sum()
    return {
        "macro_f1": float(macro_f1(cm)),
        "micro_f1": float(_safe_div(2 * p_tp, p_pred + p_gold)),
        "micro_precision": float(_safe_div(p_tp, p_pred)),
        "micro_recall": float(_safe_div(p_tp, p_gold)),
        "mark_detection_f1": float(_safe_div(2 * detected, p_pred + p_gold)),
        "accuracy": float(_safe_div(tp.sum(), cm.sum())),
        "per_class": per_class,
        "confusion": cm.tolist(),
    }


def bootstrap_ci(doc_cms: np.ndarray, metric=macro_f1, n_resamples: int = 1000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """Percentile CI of `metric` from resampling documents. doc_cms has shape (D, 4, 4)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(doc_cms), size=(n_resamples, len(doc_cms)))
    values = metric(doc_cms[idx].sum(axis=1))
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return {"point": float(metric(doc_cms.sum(axis=0))), "low": float(low), "high": float(high),
            "std": float(values.std(ddof=1)), "n_resamples": n_resamples, "n_documents": len(doc_cms)}


def paired_bootstrap(doc_cms_a: np.ndarray, doc_cms_b: np.ndarray, metric=macro_f1,
                     n_resamples: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """CI of metric(A) - metric(B), resampling the same documents for both models."""
    if doc_cms_a.shape != doc_cms_b.shape:
        raise ValueError("models must be scored on the same documents")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(doc_cms_a), size=(n_resamples, len(doc_cms_a)))
    diffs = metric(doc_cms_a[idx].sum(axis=1)) - metric(doc_cms_b[idx].sum(axis=1))
    alpha = (1 - level) / 2
    low, high = np.quantile(diffs, [alpha, 1 - alpha])
    point = float(metric(doc_cms_a.sum(axis=0)) - metric(doc_cms_b.sum(axis=0)))
    return {"difference": point, "low": float(low), "high": float(high),
            "share_not_better": float((diffs <= 0).mean()), "n_resamples": n_resamples}
