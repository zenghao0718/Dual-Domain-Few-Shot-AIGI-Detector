"""Dependency-free R2-1 statistics, identity and atomic-file primitives.

Model mathematics remain in model/ddfsd_losses.py. The scalar helpers here
are diagnostic/test references only, never used to produce model weights.
"""

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

GENERATORS = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
SEEDS = (42, 101, 102, 103, 104)
MODES = ("rgb", "freq", "dual")
METRICS = ("acc", "real_acc", "fake_acc", "balanced_acc", "ap", "auc")
SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_bytes(path, gzip.compress(payload, mtime=0) if str(path).endswith(".gz") else payload)


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None):
    import io
    rows = list(rows)
    fields = fields or list(rows[0] if rows else {})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8"))


def alpha_reference(sigma_rgb, sigma_freq, tau_r):
    if tau_r <= 0:
        raise ValueError("tau_r must be positive")
    delta = (sigma_freq - sigma_rgb) / tau_r
    raw = 1 / (1 + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1 + math.exp(delta))
    return raw, min(0.75, max(0.25, raw))


def margin(real_distance, fake_distance, label):
    if label not in (0, 1):
        raise ValueError("Expected real=0 or fake=1")
    return (fake_distance - real_distance) * (1 if label == 0 else -1)


def sign(value):
    return (value > 0) - (value < 0)


def direction(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Paired arrays must have equal lengths")
    non_ties = [(x, y) for x, y in zip(xs, ys) if x != 0 and y != 0]
    matches = sum(sign(x) == sign(y) for x, y in non_ties)
    return {"n": len(xs), "n_non_tie": len(non_ties), "n_match": matches,
            "n_x_tie": sum(x == 0 for x in xs), "n_y_tie": sum(y == 0 for y in ys),
            "n_tie": len(xs) - len(non_ties),
            "coverage": len(non_ties) / len(xs) if xs else math.nan,
            "agreement": matches / len(non_ties) if non_ties else math.nan}


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    first = 0
    while first < len(order):
        end = first + 1
        while end < len(order) and values[order[end]] == values[order[first]]:
            end += 1
        for index in order[first:end]:
            result[index] = (first + 1 + end) / 2
        first = end
    return result


def spearman(xs, ys):
    if len(xs) != len(ys) or not all(math.isfinite(v) for v in list(xs) + list(ys)):
        raise ValueError("Spearman needs finite paired input")
    if len(xs) < 2:
        return math.nan
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cross = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return cross / den if den else math.nan


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def cluster_bootstrap(rows, y_key, repeats=10000, seed=2101):
    """Resample generators, retaining all seeds/classes within each drawn cluster."""
    groups = {}
    for row in rows:
        groups.setdefault(row["generator"], []).append(row)
    keys = sorted(groups)
    values = {"rho": [], "agreement": []}
    if len(keys) >= 2:
        rng = random.Random(seed)
        for _ in range(repeats):
            sample = [row for _ in keys for row in groups[rng.choice(keys)]]
            xs, ys = [r["X"] for r in sample], [r[y_key] for r in sample]
            for name, value in (("rho", spearman(xs, ys)),
                                ("agreement", direction(xs, ys)["agreement"])):
                if math.isfinite(value):
                    values[name].append(value)
    result = {"bootstrap_repeats": repeats, "bootstrap_seed": seed, "n_clusters": len(keys)}
    for name, samples in values.items():
        result.update({name + "_ci_low": quantile(samples, 0.025),
                       name + "_ci_high": quantile(samples, 0.975),
                       name + "_bootstrap_valid": len(samples)})
    return result


def binary_metrics_reference(labels, scores, predictions):
    """Offline audit: exact tied-score AP and Mann-Whitney AUC, fake positive.

    Inference uses util.ddfsd_eval.binary_metrics. This implementation permits
    local standard-library tests and offline analysis without importing torch.
    """
    if not (len(labels) == len(scores) == len(predictions)) or set(labels) != {0, 1}:
        raise ValueError("Need equally sized arrays with both binary classes")
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite probabilities")
    if not set(predictions).issubset({0, 1}):
        raise ValueError("Invalid predictions")
    counts = [labels.count(0), labels.count(1)]
    correct = [sum(y == p and y == cls for y, p in zip(labels, predictions)) for cls in (0, 1)]
    score_ranks = ranks(scores)
    auc = (sum(r for r, y in zip(score_ranks, labels) if y == 1)
           - counts[1] * (counts[1] + 1) / 2) / (counts[0] * counts[1])
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    ap, tp, start = 0.0, 0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        increment = sum(labels[index] for index in order[start:end])
        tp += increment
        ap += increment / counts[1] * tp / end
        start = end
    real, fake = correct[0] / counts[0], correct[1] / counts[1]
    return {"acc": sum(correct) / len(labels), "real_acc": real, "fake_acc": fake,
            "balanced_acc": (real + fake) / 2, "ap": ap, "auc": auc}


def validate_split(support, query, size, shot, seed):
    from util.ddfsd_sampling import sample_support_query_indices
    if (support, query) != sample_support_query_indices(size, shot, seed):
        raise ValueError("Support/query differ from official sampling/order")
    if len(support) != shot or set(support) & set(query):
        raise ValueError("Support/query leakage")
    if sorted(support + query) != list(range(size)):
        raise ValueError("Support/query do not partition ALL validation images")


def validate_marker(root, marker, identity):
    if marker.get("identity") != identity or marker.get("identity_sha256") != canonical_hash(identity):
        raise ValueError("Task identity changed; use a new OUTPUT_DIR")
    if marker.get("status") != "DONE" or not marker.get("files"):
        raise ValueError("No atomic DONE marker")
    root = Path(root).resolve()
    for relative, digest in marker["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Missing/corrupt task artifact: {relative}")
    return True


def class_rows(identity, support, arrays):
    """Aggregate only each class's true-label queries; one alpha per class."""
    labels = arrays["labels"].tolist()
    result = []
    for cls in (0, 1):
        indices = [i for i, label in enumerate(labels) if label == cls]
        if not indices:
            raise ValueError("No queries for a class")
        row = {key: identity[key] for key in ("generator", "seed", "ckpt_sha256", "stats_sha256")}
        row.update({"class_label": cls, "class_name": ("Real", "Fake")[cls],
                    "n_support": identity["protocol"]["support_shot"], "n_query": len(indices)})
        for key in ("sigma_rgb", "sigma_freq", "alpha_raw", "alpha_rgb"):
            row[key] = support[key][cls]
        row["alpha_freq"] = 1 - row["alpha_rgb"]
        row["alpha_clipped"] = int(row["alpha_raw"] < 0.25 or row["alpha_raw"] > 0.75)
        row["alpha_at_bound"] = int(row["alpha_rgb"] in (0.25, 0.75))
        for mode in ("rgb", "freq"):
            row["proto_sep_" + mode] = support["proto_sep_" + mode]
            row["mean_margin_" + mode] = statistics.mean(
                margin(float(arrays["d_" + mode][i, 0]), float(arrays["d_" + mode][i, 1]), cls)
                for i in indices)
        for mode in MODES:
            row["recall_" + mode] = sum(int(arrays["pred_" + mode][i]) == cls for i in indices) / len(indices)
        row.update(X=row["alpha_rgb"] - 0.5,
                   Y=row["mean_margin_rgb"] - row["mean_margin_freq"],
                   delta_recall=row["recall_rgb"] - row["recall_freq"])
        result.append(row)
    return result
