"""Learned window-level reason classifier.

Stage 1 replacement: extracts normalized evidence features from raw window
data and predicts reason posterior probabilities via a trained classifier,
replacing the hand-tuned linear posterior in _compute_reason_posterior.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from refute_b_v2_d32.schema import reason_bucket

# ---------------------------------------------------------------------------
# Feature schema – network‑loss‑aware case‑normalised features (~50 dims)
# ---------------------------------------------------------------------------
_METRIC_BUCKET_ORDER = (
    "cpu", "memory", "disk_io", "filesystem",
    "network_latency", "network_packet_loss", "db_connection",
)

# Per‑bucket: fraction of total anomalies (density) + log‑scaled max deviation
METRIC_FIELDS = []
for b in _METRIC_BUCKET_ORDER:
    METRIC_FIELDS.append(f"{b}_density")
    METRIC_FIELDS.append(f"{b}_max_dev")

# Additional network‑loss discriminative features
METRIC_FIELDS.extend([
    "net_pkt_loss_subtype_density",   # fraction of net_pkt_loss KPIs with explicit loss tokens
    "cpu_dev_ratio",                  # cpu total dev / sum all total devs (cpu-specific strength)
])

TRACE_FIELDS = [
    "trace_available",
    "trace_has_slow_edge",
    "trace_has_dropped_edge",
    "trace_has_first_anomalous",
    "trace_slow_edge_max_ratio",
    "trace_dropped_edge_max_ratio",
    "trace_drop_vs_slow_ratio",       # dropped_edge_count / (slow_edge_count + 1)
    "trace_slow_vs_drop_ratio",       # slow_edge_count / (dropped_edge_count + 1)
]

LOG_FIELDS = [
    "log_available",
    "log_has_oom",
    "log_has_db",
    "log_has_crash",
    "log_component_count",
]

JOINT_REASONS = ("cpu", "network_latency", "network_packet_loss", "db_connection", "memory", "disk_io")
JOINT_FIELDS = []
for r in sorted(JOINT_REASONS):
    JOINT_FIELDS.append(f"joint_{r}_count")
    JOINT_FIELDS.append(f"joint_{r}_max_strength")

FEATURE_NAMES = METRIC_FIELDS + TRACE_FIELDS + LOG_FIELDS + JOINT_FIELDS

TARGET_BUCKETS = sorted({
    "cpu", "memory", "jvm_oom", "disk_io", "filesystem",
    "network_latency", "network_packet_loss", "db_connection",
    "process_termination",
})

# ---------------------------------------------------------------------------
# Feature extraction – workhorse called at both training and inference time
# ---------------------------------------------------------------------------


def _metric_features(metric_df: pd.DataFrame, baseline) -> dict[str, float]:
    """Metric features: anomaly density per bucket + log‑scaled max deviation
       + network-loss subtype discrimination + cpu deviation ratio."""
    out: dict[str, float] = {n: 0.0 for n in METRIC_FIELDS}
    if metric_df is None or metric_df.empty or "kpi_name" not in metric_df.columns:
        return out

    from refute_b_v2_d32.evidence import kpi_in_bucket

    # Tokens that indicate packet LOSS (not just latency/traffic)
    _LOSS_TOKENS = ("error", "err", "drop", "loss", "retrans", "reject", "rejected", "reset", "abort", "aborted")

    counts: dict[str, float] = {}
    max_devs: dict[str, float] = {}
    total_devs: dict[str, float] = {}
    net_loss_subtype_count = 0.0       # KPIs in net_pkt_loss bucket that have loss-specific tokens
    net_loss_total_count = 0.0
    for b in _METRIC_BUCKET_ORDER:
        counts[b] = 0.0
        max_devs[b] = 0.0
        total_devs[b] = 0.0

    for row in metric_df.itertuples(index=False):
        kpi = str(getattr(row, "kpi_name", ""))
        value = getattr(row, "value", None)
        if value is None or not np.isfinite(float(value)):
            continue
        kpi_low = kpi.lower()
        for b in _METRIC_BUCKET_ORDER:
            if kpi_in_bucket(kpi, b):
                try:
                    result = baseline.is_anomalous(
                        str(getattr(row, "cmdb_id", "")), kpi, float(value), threshold="p99",
                    )
                except Exception:
                    continue
                if result.is_anomalous:
                    dev = abs(float(result.deviation or 0.0))
                    counts[b] += 1.0
                    total_devs[b] += dev
                    if dev > max_devs[b]:
                        max_devs[b] = dev
                    # Network loss subtype: count KPIs with explicit loss tokens
                    if b == "network_packet_loss":
                        net_loss_total_count += 1.0
                        if any(t in kpi_low for t in _LOSS_TOKENS):
                            net_loss_subtype_count += 1.0

    total_anomalies = max(sum(counts.values()), 1.0)
    total_dev = max(sum(total_devs.values()), 1.0)

    for b in _METRIC_BUCKET_ORDER:
        out[f"{b}_density"] = counts[b] / total_anomalies
        out[f"{b}_max_dev"] = math.log1p(max_devs[b])

    out["net_pkt_loss_subtype_density"] = (
        net_loss_subtype_count / max(net_loss_total_count, 1.0)
        if net_loss_total_count > 0 else 0.0
    )
    out["cpu_dev_ratio"] = total_devs.get("cpu", 0.0) / total_dev
    return out


def _trace_features(trace_summary: Mapping[str, Any] | None) -> dict[str, float]:
    out = {f: 0.0 for f in TRACE_FIELDS}
    if not trace_summary or trace_summary.get("trace_status") != "present":
        return out

    out["trace_available"] = 1.0
    events = trace_summary.get("events", {}) or {}
    slow_edges = events.get("slow_edges", []) or []
    dropped_edges = events.get("dropped_edges", []) or []

    slow_count = len(slow_edges)
    drop_count = len(dropped_edges)

    out["trace_has_slow_edge"] = 1.0 if slow_count else 0.0
    out["trace_has_dropped_edge"] = 1.0 if drop_count else 0.0
    out["trace_has_first_anomalous"] = 1.0 if events.get("first_anomalous_service") else 0.0
    out["trace_slow_edge_max_ratio"] = max(
        (float(e.get("slow_ratio", 0.0) or 0.0) for e in slow_edges), default=0.0,
    )
    out["trace_dropped_edge_max_ratio"] = max(
        (float(e.get("count_drop_ratio", 0.0) or 0.0) for e in dropped_edges), default=0.0,
    )
    out["trace_drop_vs_slow_ratio"] = drop_count / (slow_count + 1.0)
    out["trace_slow_vs_drop_ratio"] = slow_count / (drop_count + 1.0)
    return out


def _log_features(log_df: pd.DataFrame) -> dict[str, float]:
    out = {f: 0.0 for f in LOG_FIELDS}
    if log_df is None or log_df.empty or "value" not in log_df.columns:
        return out

    out["log_available"] = 1.0
    values = log_df["value"].dropna().astype(str)
    text = " ".join(values).lower()
    out["log_has_oom"] = float(any(kw in text for kw in ("outofmemory", "oom", "heap space", "java heap")))
    out["log_has_db"] = float(any(kw in text for kw in ("connection", "timeout", "session", "too many")))
    out["log_has_crash"] = float(any(kw in text for kw in ("crash", "killed", "terminated", "segfault")))
    if "cmdb_id" in log_df.columns:
        out["log_component_count"] = math.log1p(log_df["cmdb_id"].nunique())
    return out


def _joint_features(candidates: list[Any]) -> dict[str, float]:
    out = {f: 0.0 for f in JOINT_FIELDS}
    if not candidates:
        return out
    from refute_b_v2_d32.joint_candidates import primary_bucket_for_reason

    bucket_signals: dict[str, list[float]] = defaultdict(list)
    for c in candidates:
        source = getattr(c, "source", "") if hasattr(c, "source") else str(getattr(c, "details", {}).get("source", ""))
        if source != "joint_generator":
            continue
        reason = getattr(c, "reason", "")
        bucket = primary_bucket_for_reason(reason)
        details = getattr(c, "details", {}) if hasattr(c, "details") else {}
        strength = float(details.get("signal_strength", 0.0) or 0.0)
        bucket_signals[bucket].append(strength)

    for bucket, vals in bucket_signals.items():
        key_count = f"joint_{bucket}_count"
        key_max = f"joint_{bucket}_max_strength"
        if key_count in out:
            out[key_count] = math.log1p(len(vals))
        if key_max in out:
            out[key_max] = math.log1p(max(vals)) if vals else 0.0
    return out


def _joint_features_from_raw(
    metric_df: pd.DataFrame,
    baseline,
) -> dict[str, float]:
    """Compute joint features directly from metric anomaly signals.

    Uses the same _metric_signals() as the joint candidate generator
    so training features match inference features without needing
    JointPrior or the full candidate generation pipeline.
    """
    from refute_b_v2_d32.joint_candidates import _metric_signals

    signals = _metric_signals(metric_df, baseline)
    bucket_strengths: dict[str, float] = defaultdict(float)
    bucket_counts: dict[str, int] = defaultdict(int)
    for _component, component_signals in signals.items():
        for bucket, signal in component_signals.items():
            bucket_counts[bucket] += 1
            if signal.strength > bucket_strengths.get(bucket, 0.0):
                bucket_strengths[bucket] = float(signal.strength)

    out = {f: 0.0 for f in JOINT_FIELDS}
    all_buckets = sorted(set(list(bucket_counts.keys()) + list(bucket_strengths.keys())))
    for bucket in all_buckets:
        key_count = f"joint_{bucket}_count"
        key_max = f"joint_{bucket}_max_strength"
        if key_count in out:
            out[key_count] = math.log1p(bucket_counts.get(bucket, 0))
        if key_max in out:
            out[key_max] = math.log1p(bucket_strengths.get(bucket, 0.0))

    return out


def extract_features(
    *,
    metric_df: pd.DataFrame,
    log_df: pd.DataFrame,
    trace_summary: Mapping[str, Any] | None,
    baseline,
    joint_candidates: list[Any] | None = None,
    raw_joint_features: dict[str, float] | None = None,
) -> np.ndarray:
    """Return a normalised 1‑D float64 array of shape (len(FEATURE_NAMES),).

    Training callers should pass *raw_joint_features* (computed from
    _metric_signals) so that joint dimensions are non-zero during
    training.  Inference callers pass *joint_candidates*.
    """
    feats: dict[str, float] = {}
    feats.update(_metric_features(metric_df, baseline))
    feats.update(_trace_features(trace_summary))
    feats.update(_log_features(log_df))
    if raw_joint_features is not None:
        feats.update(raw_joint_features)
    else:
        feats.update(_joint_features(joint_candidates or []))
    return np.array([feats.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Classifier wrapper – sklearn LogisticRegression
# ---------------------------------------------------------------------------


@dataclass
class ReasonClassifier:
    """Wrapper around a trained sklearn classifier with feature normalisation.

    load(path) / save(path) handle serialisation via pickle for any sklearn estimator.
    predict_proba(X) returns a dict of {reason_bucket: probability}.
    """

    model_: Any = None                                      # trained sklearn estimator
    classes_: list[str] = field(default_factory=list)
    feature_mean_: np.ndarray | None = None
    feature_std_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    eps: float = 1e-8

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X.astype(np.float64, copy=False))
        if self.feature_mean_ is not None:
            X = (X - self.feature_mean_) / (self.feature_std_ + self.eps)
        return X

    def predict_proba(self, X: np.ndarray) -> dict[str, float]:
        if self.model_ is None:
            return {b: 0.0 for b in self.classes_}
        Xt = self._transform(X)
        proba = self.model_.predict_proba(Xt)
        return {b: float(p) for b, p in zip(self.classes_, proba[0])}

    def predict(self, X: np.ndarray) -> str:
        proba = self.predict_proba(X)
        return max(proba, key=proba.get)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        pkl = path.with_suffix(".pkl")
        with pkl.open("wb") as f:
            pickle.dump(self.model_, f)
        meta = {
            "classes_": list(self.classes_),
            "feature_mean_": self.feature_mean_.tolist() if self.feature_mean_ is not None else None,
            "feature_std_": self.feature_std_.tolist() if self.feature_std_ is not None else None,
            "feature_names_": list(self.feature_names_),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ReasonClassifier:
        path = Path(path)
        pkl = path.with_suffix(".pkl")
        with pkl.open("rb") as f:
            model = pickle.load(f)
        meta = json.loads(path.read_text(encoding="utf-8"))
        mean = np.array(meta["feature_mean_"]) if meta["feature_mean_"] is not None else None
        std = np.array(meta["feature_std_"]) if meta["feature_std_"] is not None else None
        return cls(
            model_=model,
            classes_=list(meta["classes_"]),
            feature_mean_=mean,
            feature_std_=std,
            feature_names_=list(meta.get("feature_names_", FEATURE_NAMES)),
        )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
# pylint: disable=import-outside-toplevel


_DATASET_SPECS = {
    "bank": {
        "root": "/home/dell2/RCA513/ysj/dataset/openrca/Bank",
        "baseline": "refute/knowledge/baseline_distributions_all_metric_dates.json",
        "node_graph": "refute/knowledge/node_container_graph_2021_03_10.json",
        "trace": "trace_summaries/openrca_queries_full_batch18",
    },
    "telecom": {
        "root": "/home/dell2/RCA513/ysj/dataset/openrca/Telecom",
        "baseline": "knowledge/telecom_baseline_distributions_all_metric_dates.json",
        "node_graph": "knowledge/telecom_node_container_graph.json",
        "trace": "trace_summaries/telecom_queries_full",
    },
    "market_cb1": {
        "root": "/home/dell2/RCA513/ysj/dataset/openrca/Market/cloudbed-1",
        "baseline": "knowledge/market_cloudbed_1_baseline_distributions_all_metric_dates.json",
        "node_graph": "knowledge/market_cloudbed_1_node_container_graph.json",
        "trace": "trace_summaries/market_cloudbed_1_queries_full",
    },
    "market_cb2": {
        "root": "/home/dell2/RCA513/ysj/dataset/openrca/Market/cloudbed-2",
        "baseline": "knowledge/market_cloudbed_2_baseline_distributions_all_metric_dates.json",
        "node_graph": "knowledge/market_cloudbed_2_node_container_graph.json",
        "trace": "trace_summaries/market_cloudbed_2_queries_full",
    },
}


def _load_cases_and_extract(
    source_dir: Path,
    datasets: Iterable[str] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate all specified datasets, extract features and labels.

    Returns (X, y_str) where X is (n_cases, n_features) and y_str is string labels.
    Uses day-level caching to avoid re-reading the same date files.
    """
    from collections import defaultdict
    from refute_b_v2_d32.layer1 import case_rows_from_openrca
    from refute.src.data_loader import BankDataPaths, load_metric_day, load_log_day
    from refute_b_v2.query_windows import parse_query_window
    from refute.src.baseline_distributions import BaselineStore

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []

    names = list(datasets) if datasets else list(_DATASET_SPECS)
    for name in names:
        spec = _DATASET_SPECS[name]
        query_csv = Path(spec["root"]) / "query.csv"
        record_csv = Path(spec["root"]) / "record.csv"
        if not query_csv.exists() or not record_csv.exists():
            if verbose:
                print(f"  [{name}] skip: missing query/record CSV")
            continue
        cases = case_rows_from_openrca(str(query_csv), str(record_csv))

        # Group cases by date_key for caching
        date_groups: dict[str, list[dict]] = defaultdict(list)
        for row in cases:
            reason_str = str(row.get("reason", "")).strip()
            if not reason_str:
                continue
            label = reason_bucket(reason_str)
            if label not in TARGET_BUCKETS:
                continue
            date_groups[str(row.get("date_key", ""))].append(row)

        data_root = Path(spec["root"])
        paths = BankDataPaths.from_root(data_root)
        baseline_path = source_dir / spec["baseline"]
        baseline = BaselineStore.load_json(str(baseline_path))
        trace_dir = source_dir / spec["trace"]
        manifest: dict | None = None
        manifest_path = trace_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        n_ok_date = 0
        for date_key, day_cases in date_groups.items():
            if not date_key:
                continue
            try:
                metric_day = load_metric_day(paths, date_key)
                log_day = load_log_day(paths, date_key)
            except Exception:
                if verbose:
                    print(f"  [{name}] {date_key}: skip (load error)")
                continue

            for row in day_cases:
                instruction = str(row.get("instruction", ""))
                window = parse_query_window(instruction)
                metric_df = metric_day[(metric_day["timestamp"] >= window.start_ts) & (metric_day["timestamp"] <= window.end_ts)].copy() if not metric_day.empty else metric_day
                log_df = log_day[(log_day["timestamp"] >= window.start_ts) & (log_day["timestamp"] <= window.end_ts)].copy() if not log_day.empty else log_day

                trace_summary: dict | None = None
                if manifest is not None:
                    row_id = int(row["row_id"])
                    rows_entries = manifest.get("rows", [])
                    if row_id < len(rows_entries):
                        entry = rows_entries[row_id]
                        trace_file = entry if isinstance(entry, str) else entry.get("path", "")
                        if trace_file:
                            tp = trace_dir / Path(trace_file).name
                            if not tp.exists():
                                tp = source_dir / trace_file
                            if tp.exists():
                                trace_summary = json.loads(tp.read_text(encoding="utf-8"))

                label = reason_bucket(str(row.get("reason", "")))
                raw_joint = _joint_features_from_raw(metric_df, baseline)
                result = extract_features(
                    metric_df=metric_df,
                    log_df=log_df,
                    trace_summary=trace_summary,
                    baseline=baseline,
                    raw_joint_features=raw_joint,
                )
                all_features.append(result)
                all_labels.append(label)
                n_ok_date += 1
        if verbose:
            print(f"  [{name}] {n_ok_date} cases with features+label")

    X = np.stack(all_features, axis=0) if all_features else np.empty((0, len(FEATURE_NAMES)))
    return X, np.array(all_labels, dtype=str)


def _load_dataset_cases_and_extract(
    source_dir: Path,
    dataset: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for one dataset in original case order.

    Returns (X, y, row_ids). The GT reason is used only as y, never as a
    feature. Keeping row_ids lets evaluation choose a mutually exclusive
    classifier for each held-out case fold.
    """
    from collections import defaultdict
    from refute_b_v2_d32.layer1 import case_rows_from_openrca
    from refute.src.data_loader import BankDataPaths, load_metric_day, load_log_day
    from refute_b_v2.query_windows import parse_query_window
    from refute.src.baseline_distributions import BaselineStore

    if dataset not in _DATASET_SPECS:
        raise KeyError(f"unknown dataset: {dataset}")

    spec = _DATASET_SPECS[dataset]
    query_csv = Path(spec["root"]) / "query.csv"
    record_csv = Path(spec["root"]) / "record.csv"
    cases = case_rows_from_openrca(str(query_csv), str(record_csv))

    valid_cases: list[dict[str, Any]] = []
    for row in cases:
        reason_str = str(row.get("reason", "")).strip()
        if not reason_str:
            continue
        label = reason_bucket(reason_str)
        if label not in TARGET_BUCKETS:
            continue
        valid_cases.append(row)

    date_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_cases:
        date_groups[str(row.get("date_key", ""))].append(row)

    data_root = Path(spec["root"])
    paths = BankDataPaths.from_root(data_root)
    baseline = BaselineStore.load_json(str(source_dir / spec["baseline"]))
    trace_dir = source_dir / spec["trace"]
    manifest: dict | None = None
    manifest_path = trace_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    feature_by_row_id: dict[int, np.ndarray] = {}
    label_by_row_id: dict[int, str] = {}

    for date_key, day_cases in date_groups.items():
        if not date_key:
            continue
        try:
            metric_day = load_metric_day(paths, date_key)
            log_day = load_log_day(paths, date_key)
        except Exception:
            if verbose:
                print(f"  [{dataset}] {date_key}: skip (load error)")
            continue

        for row in day_cases:
            row_id = int(row["row_id"])
            window = parse_query_window(str(row.get("instruction", "")))
            metric_df = metric_day[(metric_day["timestamp"] >= window.start_ts) & (metric_day["timestamp"] <= window.end_ts)].copy() if not metric_day.empty else metric_day
            log_df = log_day[(log_day["timestamp"] >= window.start_ts) & (log_day["timestamp"] <= window.end_ts)].copy() if not log_day.empty else log_day

            trace_summary: dict | None = None
            if manifest is not None:
                rows_entries = manifest.get("rows", [])
                if row_id < len(rows_entries):
                    entry = rows_entries[row_id]
                    trace_file = entry if isinstance(entry, str) else entry.get("path", "")
                    if trace_file:
                        tp = trace_dir / Path(trace_file).name
                        if not tp.exists():
                            tp = source_dir / trace_file
                        if tp.exists():
                            trace_summary = json.loads(tp.read_text(encoding="utf-8"))

            raw_joint = _joint_features_from_raw(metric_df, baseline)
            feature_by_row_id[row_id] = extract_features(
                metric_df=metric_df,
                log_df=log_df,
                trace_summary=trace_summary,
                baseline=baseline,
                raw_joint_features=raw_joint,
            )
            label_by_row_id[row_id] = reason_bucket(str(row.get("reason", "")))

    ordered_row_ids = [int(row["row_id"]) for row in valid_cases if int(row["row_id"]) in feature_by_row_id]
    X = np.stack([feature_by_row_id[row_id] for row_id in ordered_row_ids], axis=0) if ordered_row_ids else np.empty((0, len(FEATURE_NAMES)))
    y = np.array([label_by_row_id[row_id] for row_id in ordered_row_ids], dtype=str)
    row_ids = np.array(ordered_row_ids, dtype=int)
    if verbose:
        print(f"  [{dataset}] {len(row_ids)} ordered cases with features+label")
    return X, y, row_ids


def contiguous_case_folds(n_cases: int, n_folds: int = 2) -> list[np.ndarray]:
    """Split original case order into contiguous held-out folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    indices = np.arange(n_cases, dtype=int)
    return [fold.astype(int) for fold in np.array_split(indices, n_folds) if len(fold) > 0]


def train_reason_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    reg_alpha: float = 0.5,
    reg_lambda: float = 1.0,
    subsample: float = 0.8,
    random_state: int = 42,
) -> ReasonClassifier:
    """Train a classifier and wrap as ReasonClassifier.

    Auto-selects model complexity based on dataset size:
      - < 60 samples: LogisticRegression (simple, regularizes well)
      - >= 60 samples: XGBoost (used as before)
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = [str(c) for c in le.classes_]
    n_classes = len(classes)
    n_samples = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float64))

    if n_samples < 60 or n_classes <= 2:
        # Tiny dataset: fall back to logistic regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=0.3,
            random_state=random_state,
        )
        clf.fit(X_scaled, y_enc)
    else:
        n_est = min(n_estimators, max(40, n_samples // 2))
        m_depth = min(max_depth, max(3, int(np.log2(n_samples))))
        clf = xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=m_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            objective="multi:softprob",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        clf.fit(X_scaled, y_enc)

    return ReasonClassifier(
        model_=clf,
        classes_=classes,
        feature_mean_=scaler.mean_.astype(np.float64),
        feature_std_=scaler.scale_.astype(np.float64),
        feature_names_=list(FEATURE_NAMES),
    )


def evaluate_classifier(
    clf: ReasonClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "",
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate a ReasonClassifier on a test set and return metrics."""
    from collections import Counter

    if len(X) == 0:
        return {"accuracy": 0.0, "n_samples": 0}

    y_pred = np.array([clf.predict(xi) for xi in X])
    acc = (y_pred == y).mean()

    # Per-class precision/recall/F1
    all_classes = sorted(set(y) | set(clf.classes_))
    per_class: dict[str, dict[str, float]] = {}
    for cls in all_classes:
        tp = sum((y_pred == cls) & (y == cls))
        fp = sum((y_pred == cls) & (y != cls))
        fn = sum((y_pred != cls) & (y == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(sum(y == cls)),
        }

    # Macro averages
    macro_precision = np.mean([m["precision"] for m in per_class.values()])
    macro_recall = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class.values()])

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "n_samples": len(X),
        "per_class": per_class,
    }

    if verbose:
        prefix = f"[{dataset_name}] " if dataset_name else ""
        print(f"{prefix}Test accuracy: {acc:.3f} ({len(X)} samples)")
        print(f"{prefix}Macro P/R/F1: {macro_precision:.3f} / {macro_recall:.3f} / {macro_f1:.3f}")
        print(f"{prefix}Prediction distribution: {dict(Counter(y_pred))}")
        print(f"{prefix}Ground truth distribution: {dict(Counter(y))}")
        if hasattr(clf.model_, 'feature_importances_'):
            imp = clf.model_.feature_importances_
            top = sorted(zip(clf.feature_names_, imp), key=lambda x: -x[1])[:10]
            print(f"{prefix}Top-10 feature importances: {', '.join(f'{n}={v:.3f}' for n,v in top)}")

    return metrics


def main_train(
    *,
    out: str = "knowledge/reason_classifier.json",
    exclude: Iterable[str] | None = None,
    verbose: bool = True,
):
    """CLI entry point: train on specified datasets, save to *out*.

    If *exclude* is given, those dataset keys are held out (LODO style).
    Default trains on all four datasets.
    """
    source_dir = Path.cwd()
    datasets = [d for d in _DATASET_SPECS if exclude is None or d not in set(exclude or [] or [])]
    X, y = _load_cases_and_extract(source_dir, datasets=datasets, verbose=verbose)
    if len(X) == 0:
        raise RuntimeError("no training cases extracted")
    if verbose:
        from collections import Counter
        print(f"Training on {len(X)} cases, {X.shape[1]} features, classes: {dict(Counter(y))}")
    clf = train_reason_classifier(X, y)
    clf.save(source_dir / out)
    if verbose:
        y_pred = np.array([clf.predict(xi) for xi in X])
        acc = (y_pred == y).mean()
        print(f"Train accuracy: {acc:.3f}")
        from collections import Counter
        print(f"Prediction distribution: {dict(Counter(y_pred))}")
        # Feature importance
        if hasattr(clf.model_, 'feature_importances_'):
            imp = clf.model_.feature_importances_
            top = sorted(zip(clf.feature_names_, imp), key=lambda x: -x[1])[:10]
            print(f"Top-10 feature importances: {', '.join(f'{n}={v:.3f}' for n,v in top)}")
        print(f"Saved to {source_dir / out} (+ .pkl)")


def main_lodo(
    *,
    out_dir: str = "knowledge/lodo_classifiers",
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Leave-One-Dataset-Out cross-validation for the reason classifier.

    For each dataset:
      1. Train on the other 3 datasets (NO access to held-out data)
      2. Evaluate on the held-out dataset
      3. Save the LODO classifier

    Returns dict of {dataset_name: metrics}.
    """
    source_dir = Path.cwd()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}

    for held_out in sorted(_DATASET_SPECS.keys()):
        train_datasets = [d for d in _DATASET_SPECS if d != held_out]

        if verbose:
            print(f"\n{'='*60}")
            print(f"LODO fold: held_out={held_out}, train={train_datasets}")
            print(f"{'='*60}")

        # Train on other datasets
        X_train, y_train = _load_cases_and_extract(source_dir, datasets=train_datasets, verbose=verbose)
        if len(X_train) == 0:
            print(f"  [{held_out}] SKIP: no training data from {train_datasets}")
            continue

        if verbose:
            from collections import Counter
            print(f"  Training on {len(X_train)} cases, classes: {dict(Counter(y_train))}")

        clf = train_reason_classifier(X_train, y_train)

        # Evaluate on held-out dataset
        X_test, y_test = _load_cases_and_extract(source_dir, datasets=[held_out], verbose=verbose)
        if len(X_test) == 0:
            print(f"  [{held_out}] SKIP: no test data")
            continue

        metrics = evaluate_classifier(clf, X_test, y_test, dataset_name=held_out, verbose=verbose)

        # Save LODO classifier
        clf_path = out_path / f"reason_classifier_exclude_{held_out}.json"
        clf.save(clf_path)
        if verbose:
            print(f"  Saved LODO classifier to {clf_path}")

        all_metrics[held_out] = metrics

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("LODO Cross-Validation Summary")
        print(f"{'='*60}")
        for ds, m in all_metrics.items():
            print(f"  {ds:15s}: acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  (n={m['n_samples']})")

        # Overall macro-average across folds
        if all_metrics:
            overall_acc = np.mean([m['accuracy'] for m in all_metrics.values()])
            overall_f1 = np.mean([m['macro_f1'] for m in all_metrics.values()])
            print(f"\n  Overall macro-average: acc={overall_acc:.3f}  f1={overall_f1:.3f}")

    return all_metrics


def main_casefold(
    *,
    out_dir: str = "knowledge/casefold_classifiers",
    n_folds: int = 2,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Train same-dataset cross-fit classifiers with disjoint case folds.

    For each dataset independently, split cases by original order into
    contiguous folds. Classifier `dataset_foldK` is trained only on cases from
    the other folds and is evaluated/used only for fold K.
    """
    source_dir = Path.cwd()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    manifest: dict[str, Any] = {
        "scheme": "same_dataset_casefold_crossfit",
        "n_folds": int(n_folds),
        "datasets": {},
    }

    for dataset in sorted(_DATASET_SPECS.keys()):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Case-fold dataset: {dataset}")
            print(f"{'='*60}")

        X, y, row_ids = _load_dataset_cases_and_extract(source_dir, dataset, verbose=verbose)
        if len(X) == 0:
            print(f"  [{dataset}] SKIP: no cases extracted")
            continue

        folds = contiguous_case_folds(len(X), n_folds=n_folds)
        dataset_metrics: dict[str, float] = {}
        manifest["datasets"][dataset] = {
            "n_cases": int(len(X)),
            "row_ids": [int(v) for v in row_ids.tolist()],
            "folds": [],
        }

        for fold_idx, test_idx in enumerate(folds):
            train_idx = np.array([i for i in range(len(X)) if i not in set(test_idx.tolist())], dtype=int)
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"  [{dataset}] fold{fold_idx}: skip empty split")
                continue

            if verbose:
                from collections import Counter
                test_rows = row_ids[test_idx]
                train_rows = row_ids[train_idx]
                print(
                    f"  fold{fold_idx}: train={len(train_idx)} cases "
                    f"rows {int(train_rows[0])}-{int(train_rows[-1])}; "
                    f"test={len(test_idx)} cases rows {int(test_rows[0])}-{int(test_rows[-1])}; "
                    f"train classes={dict(Counter(y[train_idx]))}"
                )

            clf = train_reason_classifier(X[train_idx], y[train_idx])
            metrics = evaluate_classifier(
                clf,
                X[test_idx],
                y[test_idx],
                dataset_name=f"{dataset}_fold{fold_idx}",
                verbose=verbose,
            )

            clf_path = out_path / f"reason_classifier_{dataset}_fold{fold_idx}.json"
            clf.save(clf_path)
            if verbose:
                print(f"  Saved case-fold classifier to {clf_path}")

            dataset_metrics[f"fold{fold_idx}_accuracy"] = float(metrics["accuracy"])
            dataset_metrics[f"fold{fold_idx}_macro_f1"] = float(metrics["macro_f1"])
            dataset_metrics[f"fold{fold_idx}_n_samples"] = int(metrics["n_samples"])
            manifest["datasets"][dataset]["folds"].append({
                "fold": int(fold_idx),
                "classifier": str(clf_path),
                "train_row_ids": [int(v) for v in row_ids[train_idx].tolist()],
                "test_row_ids": [int(v) for v in row_ids[test_idx].tolist()],
                "metrics": metrics,
            })

        fold_ns = [v for k, v in dataset_metrics.items() if k.endswith("_n_samples")]
        if fold_ns:
            total_n = float(sum(fold_ns))
            weighted_acc = sum(
                dataset_metrics[f"fold{i}_accuracy"] * dataset_metrics[f"fold{i}_n_samples"]
                for i in range(len(folds))
                if f"fold{i}_accuracy" in dataset_metrics
            ) / total_n
            dataset_metrics["casefold_accuracy"] = float(weighted_acc)

        all_metrics[dataset] = dataset_metrics

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        print(f"\n{'='*60}")
        print("Same-dataset case-fold classifier summary")
        print(f"{'='*60}")
        for dataset, metrics in all_metrics.items():
            acc = metrics.get("casefold_accuracy", 0.0)
            n = sum(v for k, v in metrics.items() if k.endswith("_n_samples"))
            print(f"  {dataset:15s}: casefold_acc={acc:.3f}  (n={int(n)})")
        print(f"  Manifest: {manifest_path}")

    return all_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd().parent))

    # Default: same-dataset case-fold cross-fit to prevent same-case leakage.
    main_casefold(out_dir="knowledge/casefold_classifiers", n_folds=2)
