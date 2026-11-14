"""Deterministic multi-omics analysis tools (the agent's *Action* module).

Every tool is a plain Python function returning a small JSON-serialisable dict, so it can be
called by hand in a notebook or by an LLM agent through tool use. The LLM never touches raw
data - it only sees these compact summaries (this is how agents overcome the "LLMs can't
process complex biomedical data" limitation discussed in the paper's introduction).
"""
from __future__ import annotations

import json
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from .config import OUTPUT_DIR

MODALITIES = list(data.MODALITIES)
SUBTYPES = ["Basal", "Her2", "LumA"]


class ToolError(ValueError):
    """Raised for bad arguments; the agent loop returns it to the model as is_error."""


# ----------------------------------------------------------------------------- helpers
def _check_modality(modality: str, split: str = "train") -> pd.DataFrame:
    d = data.load()
    if modality not in MODALITIES:
        raise ToolError(f"Unknown modality '{modality}'. Choose from {MODALITIES}.")
    if modality not in d[split]:
        raise ToolError(f"Modality '{modality}' is not available in the {split} split.")
    return d[split][modality]


def _check_group(group: str) -> str:
    if group not in SUBTYPES:
        raise ToolError(f"Unknown subtype '{group}'. Choose from {SUBTYPES}.")
    return group


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def _confidence(proba: np.ndarray) -> np.ndarray:
    """1 - normalised entropy: 1 = certain, 0 = uniform guess (per sample)."""
    p = np.clip(proba, 1e-12, 1)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
    return 1 - ent


def _late_fusion(probas: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    """Uncertainty-weighted voting (Agentomics-ML-style per-layer uncertainty weighting)."""
    weights = {m: _confidence(p) for m, p in probas.items()}
    total = sum(weights[m][:, None] * probas[m] for m in probas)
    total /= sum(weights.values())[:, None]
    mean_w = {m: round(float(w.mean()), 3) for m, w in weights.items()}
    return total, mean_w


# ----------------------------------------------------------------------------- tools
def list_modalities() -> dict:
    """Overview of the dataset: modalities, sample counts and subtype balance per split."""
    d = data.load()
    out = {}
    for split in data.SPLITS:
        out[split] = {
            "n_samples": int(len(d[split]["subtype"])),
            "subtype_counts": d[split]["subtype"].value_counts().to_dict(),
            "modalities": {
                m: list(d[split][m].shape) for m in MODALITIES if m in d[split]
            },
            "missing_modalities": [m for m in MODALITIES if m not in d[split]],
        }
    return out


def describe_modality(modality: str, split: str = "train") -> dict:
    """Shape, missing values, value range and a few example features of one omics layer."""
    if split not in data.SPLITS:
        raise ToolError(f"split must be one of {data.SPLITS}")
    df = _check_modality(modality, split)
    return {
        "modality": modality,
        "split": split,
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "value_range": [round(float(df.min().min()), 2), round(float(df.max().max()), 2)],
        "mean_feature_sd": round(float(df.std().mean()), 3),
        "example_features": list(df.columns[:8]),
    }


def differential_features(modality: str, group_a: str, group_b: str, top_n: int = 10) -> dict:
    """Welch t-test per feature between two subtypes, Benjamini-Hochberg FDR, top hits."""
    df = _check_modality(modality)
    a, b = _check_group(group_a), _check_group(group_b)
    if a == b:
        raise ToolError("group_a and group_b must differ.")
    top_n = int(max(1, min(top_n, 30)))
    y = data.load()["train"]["subtype"]
    xa, xb = df[y == a], df[y == b]
    t, p = stats.ttest_ind(xa, xb, equal_var=False)
    res = pd.DataFrame(
        {"feature": df.columns, "log_fc_like": xa.mean() - xb.mean(), "t": t, "p": p}
    )
    res["fdr"] = stats.false_discovery_control(res["p"].fillna(1.0))
    res = res.reindex(res["t"].abs().sort_values(ascending=False).index)
    top = res.head(top_n)
    return {
        "modality": modality,
        "comparison": f"{a} vs {b} (positive = higher in {a})",
        "n_significant_fdr_0.05": int((res["fdr"] < 0.05).sum()),
        "n_features_tested": int(len(res)),
        "top_features": [
            {
                "feature": r.feature,
                "mean_diff": round(float(r.log_fc_like), 3),
                "t": round(float(r.t), 2),
                "fdr": float(f"{r.fdr:.2e}"),
            }
            for r in top.itertuples()
        ],
    }


def pca_summary(modality: str, n_components: int = 5) -> dict:
    """PCA of one layer: variance explained and how well PC1/PC2 separate the subtypes."""
    df = _check_modality(modality)
    n_components = int(max(2, min(n_components, 10)))
    y = data.load()["train"]["subtype"].values
    X = StandardScaler().fit_transform(df.values)
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    pcs = pca.transform(X)
    centroids = {
        s: [round(float(v), 2) for v in pcs[y == s, :2].mean(axis=0)] for s in SUBTYPES
    }
    sep = {}
    for k in range(2):
        f, p = stats.f_oneway(*[pcs[y == s, k] for s in SUBTYPES])
        sep[f"PC{k + 1}"] = {"anova_F": round(float(f), 1), "p": float(f"{p:.2e}")}
    loadings = pd.Series(pca.components_[0], index=df.columns)
    return {
        "modality": modality,
        "variance_explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "subtype_centroids_PC1_PC2": centroids,
        "subtype_separation": sep,
        "top_PC1_loadings": loadings.abs().sort_values(ascending=False).head(8).index.tolist(),
    }


def classify_subtype(modalities: list[str], fusion: str = "early") -> dict:
    """5-fold CV logistic regression predicting PAM50 subtype from one or more layers.

    fusion="early": concatenate features of all layers into one model.
    fusion="late":  one model per layer, combined by uncertainty-weighted voting.
    """
    if not modalities:
        raise ToolError("Provide at least one modality.")
    if fusion not in ("early", "late"):
        raise ToolError("fusion must be 'early' or 'late'.")
    frames = {m: _check_modality(m) for m in modalities}
    y = data.load()["train"]["subtype"].values
    per_layer, probas = {}, {}
    for m, df in frames.items():
        p = cross_val_predict(_model(), df.values, y, cv=_cv(), method="predict_proba")
        probas[m] = p
        per_layer[m] = round(float((np.array(sorted(SUBTYPES))[p.argmax(1)] == y).mean()), 3)
    classes = np.array(sorted(SUBTYPES))
    out = {"fusion": fusion, "per_layer_cv_accuracy": per_layer}
    if len(frames) > 1:
        if fusion == "early":
            X = pd.concat(frames.values(), axis=1).values
            p = cross_val_predict(_model(), X, y, cv=_cv(), method="predict_proba")
        else:
            p, mean_w = _late_fusion(probas)
            out["mean_layer_confidence_weights"] = mean_w
        out["fused_cv_accuracy"] = round(float((classes[p.argmax(1)] == y).mean()), 3)
    return out


def predict_on_test(modalities: list[str]) -> dict:
    """Train on the train split, predict the held-out test split.

    Layers missing from the test split are skipped automatically (late fusion over the
    layers that ARE available) - a concrete example of omics data-sparsity heterogeneity.
    """
    if not modalities:
        raise ToolError("Provide at least one modality.")
    d = data.load()
    for m in modalities:
        _check_modality(m)
    available = [m for m in modalities if m in d["test"]]
    skipped = [m for m in modalities if m not in d["test"]]
    if not available:
        raise ToolError(f"None of {modalities} exist in the test split.")
    ytr, yte = d["train"]["subtype"].values, d["test"]["subtype"].values
    probas, per_layer = {}, {}
    for m in available:
        clf = _model().fit(d["train"][m].values, ytr)
        probas[m] = clf.predict_proba(d["test"][m].values)
        per_layer[m] = round(float((clf.classes_[probas[m].argmax(1)] == yte).mean()), 3)
    p, mean_w = _late_fusion(probas) if len(probas) > 1 else (probas[available[0]], None)
    pred = clf.classes_[p.argmax(1)]
    cm = pd.crosstab(pd.Series(yte, name="true"), pd.Series(pred, name="pred"))
    return {
        "used_modalities": available,
        "skipped_missing_in_test": skipped,
        "per_layer_test_accuracy": per_layer,
        "fused_test_accuracy": round(float((pred == yte).mean()), 3),
        "mean_layer_confidence_weights": mean_w,
        "confusion_matrix": {k: v for k, v in cm.to_dict(orient="index").items()},
    }


def make_plot(kind: str, modality: str, feature: str | None = None) -> dict:
    """Save a figure to outputs/. kind='pca' (PC1 vs PC2) or 'boxplot' (one feature by subtype)."""
    df = _check_modality(modality)
    y = data.load()["train"]["subtype"]
    fig, ax = plt.subplots(figsize=(5, 4))
    if kind == "pca":
        pcs = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(df.values))
        for s in SUBTYPES:
            ax.scatter(*pcs[(y == s).values].T, label=s, s=18, alpha=0.8)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"PCA of {modality}")
        ax.legend(title="Subtype")
        name = f"pca_{modality}.png"
    elif kind == "boxplot":
        if feature not in df.columns:
            raise ToolError(f"Feature '{feature}' not found in {modality}.")
        ax.boxplot([df.loc[y == s, feature] for s in SUBTYPES], labels=SUBTYPES)
        ax.set(ylabel=f"{feature} ({modality})", title=f"{feature} by subtype")
        name = f"box_{modality}_{feature}.png".replace("/", "_")
    else:
        plt.close(fig)
        raise ToolError("kind must be 'pca' or 'boxplot'.")
    fig.tight_layout()
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"saved_plot": str(path.relative_to(OUTPUT_DIR.parent))}


# ----------------------------------------------------------------------------- schemas
def _mod_prop(allowed):
    return {"type": "string", "enum": allowed, "description": "Omics layer"}


def _schemas(allowed: list[str]) -> dict[str, dict]:
    mods_list = {"type": "array", "items": {"type": "string", "enum": allowed}, "minItems": 1}
    grp = {"type": "string", "enum": SUBTYPES}
    return {
        "list_modalities": {
            "description": "Overview of available omics layers, sample counts, subtype balance "
            "and which layers are missing in each split. Call this first.",
            "properties": {},
            "required": [],
        },
        "describe_modality": {
            "description": "Shape, missingness, value range and example feature names of one layer.",
            "properties": {
                "modality": _mod_prop(allowed),
                "split": {"type": "string", "enum": ["train", "test"]},
            },
            "required": ["modality"],
        },
        "differential_features": {
            "description": "Differential analysis (Welch t-test + BH FDR) between two PAM50 "
            "subtypes in one layer; returns top features ranked by |t|.",
            "properties": {
                "modality": _mod_prop(allowed),
                "group_a": grp,
                "group_b": grp,
                "top_n": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["modality", "group_a", "group_b"],
        },
        "pca_summary": {
            "description": "PCA of one layer: variance explained, subtype centroids, ANOVA "
            "separation on PC1/PC2 and top PC1 loadings.",
            "properties": {"modality": _mod_prop(allowed)},
            "required": ["modality"],
        },
        "classify_subtype": {
            "description": "5-fold cross-validated subtype classifier. Reports per-layer accuracy "
            "and, for >1 layer, fused accuracy. fusion='early' concatenates layers; "
            "fusion='late' uses uncertainty-weighted voting across per-layer models.",
            "properties": {
                "modalities": mods_list,
                "fusion": {"type": "string", "enum": ["early", "late"]},
            },
            "required": ["modalities"],
        },
        "predict_on_test": {
            "description": "Train on train split and evaluate on the held-out test split. "
            "Layers missing in the test split are skipped (late fusion on what is available).",
            "properties": {"modalities": mods_list},
            "required": ["modalities"],
        },
        "make_plot": {
            "description": "Save a figure: kind='pca' (PC1 vs PC2 coloured by subtype) or "
            "kind='boxplot' (one feature across subtypes; needs 'feature').",
            "properties": {
                "kind": {"type": "string", "enum": ["pca", "boxplot"]},
                "modality": _mod_prop(allowed),
                "feature": {"type": "string"},
            },
            "required": ["kind", "modality"],
        },
    }


FUNCTIONS: dict[str, Callable] = {
    "list_modalities": list_modalities,
    "describe_modality": describe_modality,
    "differential_features": differential_features,
    "pca_summary": pca_summary,
    "classify_subtype": classify_subtype,
    "predict_on_test": predict_on_test,
    "make_plot": make_plot,
}


class Toolbox:
    """A set of tools an agent may use. `allowed_modalities` restricts specialist agents."""

    def __init__(self, names=None, allowed_modalities=None, extra: dict | None = None):
        self.allowed = list(allowed_modalities or MODALITIES)
        schemas = _schemas(self.allowed)
        names = list(FUNCTIONS) if names is None else list(names)
        self.functions = {n: FUNCTIONS[n] for n in names}
        self.schemas = [
            {
                "name": n,
                "description": schemas[n]["description"],
                "input_schema": {
                    "type": "object",
                    "properties": schemas[n]["properties"],
                    "required": schemas[n]["required"],
                    "additionalProperties": False,
                },
            }
            for n in names
        ]
        # extra = {name: (function, schema_dict)} for tools defined elsewhere (RAG, memory...)
        for n, (fn, schema) in (extra or {}).items():
            self.functions[n] = fn
            self.schemas.append({"name": n, **schema})

    def execute(self, name: str, args: dict) -> str:
        if name not in self.functions:
            raise ToolError(f"Tool '{name}' is not available to this agent.")
        for key in ("modality",):
            if key in args and args[key] not in self.allowed:
                raise ToolError(f"This agent may only use modalities {self.allowed}.")
        if "modalities" in args and not set(args["modalities"]) <= set(self.allowed):
            raise ToolError(f"This agent may only use modalities {self.allowed}.")
        result = self.functions[name](**args)
        return result if isinstance(result, str) else json.dumps(result, default=str)
