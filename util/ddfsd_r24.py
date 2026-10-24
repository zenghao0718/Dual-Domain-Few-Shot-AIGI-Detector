"""R2-4 deterministic support selection and provenance, without torch imports."""

from collections.abc import Mapping
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile

from util.ddfsd_sampling import sample_support_query_indices

GENERATORS = ("ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM")
SEEDS = (42, 101, 102, 103, 104)
SHOT = 10
NOISE_RATES = (0.1, 0.2, 0.3)
CLIP_ID = "openai/clip-vit-base-patch32"
SCHEMA = 1


def stable_seed(*parts):
    payload = json.dumps(["ddfsd-r24-v1", *parts], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
    atomic_write(path, gzip.compress(data, mtime=0) if str(path).endswith(".gz") else data)


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0] if rows else {})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode())


def seeds_from_string(value):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("Seeds must be comma-separated integers") from error
    if not values or len(values) != len(set(values)) or any(s < 0 or s >= 2 ** 32 for s in values):
        raise ValueError("Seeds must be unique and in [0, 2**32)")
    return values


def noise_rates_from_string(value):
    try:
        rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("Noise rates must be 0,0.1,0.2,0.3") from error
    if not rates or len(rates) != len(set(rates)) or any(p not in (0.0, *NOISE_RATES) for p in rates):
        raise ValueError("Only unique noise rates 0,0.1,0.2,0.3 are supported; zero is parity verification only")
    return rates


def condition_name(rate):
    return "clean_parity_only" if rate == 0 else f"label_noise_{round(100 * rate)}pct"


def require_sizes(real_size, fake_size):
    if real_size <= SHOT or fake_size <= SHOT:
        raise ValueError(f"Each val class needs at least {SHOT + 1} images; got real={real_size}, fake={fake_size}")


def label_noise_selection(real_size, fake_size, seed, generator, noise_rate):
    require_sizes(real_size, fake_size)
    if generator not in GENERATORS or noise_rate not in (0.0, *NOISE_RATES):
        raise ValueError("Invalid generator/noise rate; expected 0,0.1,0.2,0.3")
    rs, rq = sample_support_query_indices(real_size, SHOT, seed)
    fs, fq = sample_support_query_indices(fake_size, SHOT, seed)
    # Shared ordered corruption schedule: rate selects a prefix of length k.
    # Hence the stream is separate from official sampling and errors nest.
    real_order, fake_order = list(rs), list(fs)
    rng = random.Random(stable_seed("label_noise_schedule", generator, seed))
    rng.shuffle(real_order)
    rng.shuffle(fake_order)
    k = round(SHOT * noise_rate)
    swapped_real, swapped_fake = real_order[:k], fake_order[:k]
    if k == 0:
        groups = [[(0, i) for i in rs], [(1, i) for i in fs]]
    else:
        groups = [[(0, i) for i in rs if i not in swapped_real] + [(1, i) for i in swapped_fake],
                  [(1, i) for i in fs if i not in swapped_fake] + [(0, i) for i in swapped_real]]
    selection = {"groups": groups, "query_indices": [rq, fq], "original_support_indices": [rs, fs],
                 "swapped_indices": [swapped_real, swapped_fake], "noise_rate": noise_rate,
                 "swapped_per_class": k, "mislabeled_total": 2 * k,
                 "schedule_seed": stable_seed("label_noise_schedule", generator, seed),
                 "condition_seed": stable_seed("label_noise_condition", generator, seed, noise_rate)}
    validate_selection(selection, real_size, fake_size)
    return selection


def validate_selection(selection, real_size, fake_size):
    require_sizes(real_size, fake_size)
    groups, queries = selection["groups"], selection["query_indices"]
    if len(groups) != 2 or any(len(group) != SHOT for group in groups) or len(queries) != 2:
        raise ValueError("Need exactly 10 observed Real + 10 observed Fake support entries")
    support = [tuple(item) for group in groups for item in group]
    if len(set(support)) != 2 * SHOT:
        raise ValueError("Duplicate support source/index")
    for cls, size in enumerate((real_size, fake_size)):
        source_support = [index for source, index in support if source == cls]
        query = queries[cls]
        if len(source_support) != SHOT or len(set(query)) != len(query):
            raise ValueError("Each source needs 10 support images and unique queries")
        if set(source_support) & set(query) or sorted(source_support + list(query)) != list(range(size)):
            raise ValueError("Support/query must partition ALL val images without overlap")
    if any(source not in (0, 1) for source, _ in support):
        raise ValueError("Support true source labels must be real=0/fake=1")
    return True


def cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Cosine requires equal nonempty vectors")
    norm = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(x) ** 2 for x in b))
    if not norm or not math.isfinite(norm):
        raise ValueError("Invalid CLIP vector")
    value = sum(float(x) * float(y) for x, y in zip(a, b)) / norm
    if not math.isfinite(value):
        raise ValueError("Non-finite cosine")
    return max(-1.0, min(1.0, value))


def similarities(features, anchor):
    # Runtime uses a vectorized float64 dot/norm operation; the pure fallback
    # makes sampling unit tests independent of numpy/torch installations.
    if hasattr(features, "shape"):
        import numpy as np
        values = np.asarray(features, dtype=np.float64)
        target = np.asarray(anchor, dtype=np.float64)
        scores = values @ target / (np.linalg.norm(values, axis=1) * np.linalg.norm(target))
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite CLIP similarities")
        return np.clip(scores, -1, 1).tolist()
    return [cosine(row, anchor) for row in features]


def top_indices(scores, count=SHOT, mandatory=None):
    if count > len(scores) or count <= 0 or any(not math.isfinite(s) for s in scores):
        raise ValueError("Invalid top-k input")
    # Source indices follow official path order. The anchor is a mandatory
    # member if identical-vector ties would otherwise exclude it; all other
    # ties are ordered by ascending source index.
    ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    chosen = ordered[:count]
    if mandatory is not None:
        if not 0 <= mandatory < len(scores):
            raise ValueError("Anchor index out of range")
        if mandatory not in chosen:
            chosen[-1] = mandatory
        chosen.sort(key=lambda i: (-scores[i], i))
    return chosen


def pairwise_mean(features, indices):
    return statistics.mean(cosine(list(features[a]), list(features[b])) for position, a in enumerate(indices)
                           for b in indices[position + 1:])


def clip_selection(real_features, fake_features, seed, generator):
    real_size, fake_size = len(real_features), len(fake_features)
    require_sizes(real_size, fake_size)
    if generator not in GENERATORS:
        raise ValueError("Unknown generator")
    anchor_seed = stable_seed("clip_real_anchor", generator, seed)
    anchor = random.Random(anchor_seed).randrange(real_size)
    target = list(real_features[anchor])
    rscores, fscores = similarities(real_features, target), similarities(fake_features, target)
    # The anchor's self-cosine is exactly the theoretical maximum. It is an
    # explicit mandatory support member, including degenerate equal-vector ties.
    rscores[anchor] = 1.0
    rs = top_indices(rscores, mandatory=anchor)
    fs = top_indices(fscores)
    selection = {"groups": [[(0, i) for i in rs], [(1, i) for i in fs]],
                 "query_indices": [[i for i in range(real_size) if i not in set(rs)],
                                   [i for i in range(fake_size) if i not in set(fs)]],
                 "anchor_index": anchor, "anchor_seed": anchor_seed,
                 "support_similarities": [[rscores[i] for i in rs], [fscores[i] for i in fs]]}
    diagnostics = {}
    verified = True
    for cls, (features, selected, scores) in enumerate(((real_features, rs, rscores), (fake_features, fs, fscores))):
        random_support, _ = sample_support_query_indices(len(features), SHOT, seed)
        name = ("real", "fake")[cls]
        concentrated = pairwise_mean(features, selected)
        random_mean = pairwise_mean(features, random_support)
        diagnostics.update({name + "_pairwise_concentrated": concentrated,
                            name + "_pairwise_random": random_mean,
                            name + "_pairwise_delta": concentrated - random_mean,
                            name + "_anchor_mean_concentrated": statistics.mean(scores[i] for i in selected),
                            name + "_anchor_mean_random": statistics.mean(scores[i] for i in random_support)})
        verified &= concentrated > random_mean
    diagnostics["concentration_status"] = "CONCENTRATION_VERIFIED" if verified else "CONCENTRATION_NOT_VERIFIED"
    selection["diagnostics"] = diagnostics
    validate_selection(selection, real_size, fake_size)
    return selection


def support_rows(generator, seed, experiment, condition, selection, paths):
    anchor = selection.get("anchor_index")
    result = []
    for observed, group in enumerate(selection["groups"]):
        for rank, (source, index) in enumerate(group):
            result.append({"generator": generator, "seed": seed, "experiment": experiment, "condition": condition,
                           "observed_label": observed, "true_label": source, "source_class": "real" if source == 0 else generator,
                           "source_index": index, "relative_image_path": paths[source][index], "is_swapped": int(source != observed),
                           "anchor_id": paths[0][anchor] if anchor is not None else "", "anchor_index": anchor if anchor is not None else "",
                           "support_rank": rank + 1, "similarity_to_anchor": selection.get("support_similarities", [[""] * SHOT] * 2)[observed][rank]})
    return result


def inventory(paths, data_root):
    root = Path(data_root)
    return [{"index": i, "relative_path": Path(path).relative_to(root).as_posix(),
             "bytes": Path(path).stat().st_size, "sha256": file_sha256(path)} for i, path in enumerate(paths)]


def outside_repo(path, repo_root):
    path, root = Path(path).expanduser().resolve(), Path(repo_root).resolve()
    if path == root or root in path.parents:
        raise ValueError(f"Output/cache/model directory must be outside repository: {path}")
    return path


def checkpoint_metadata(checkpoint, generator):
    if int(checkpoint.get("step", -1)) != 15000:
        raise ValueError("Checkpoint metadata step must be 15000")
    sources = {"checkpoint": checkpoint}
    for name in ("config", "args"):
        value = checkpoint.get(name)
        if value is not None:
            sources[name] = value if isinstance(value, Mapping) else vars(value)
    declarations = {name: source["model_mode"] for name, source in sources.items() if "model_mode" in source}
    if any(not isinstance(value, str) or value != "dual" for value in declarations.values()):
        raise ValueError("Explicit checkpoint model_mode must be dual without conflicts: " + repr(declarations))
    excluded = [source["exclude_class"] for source in sources.values() if "exclude_class" in source]
    if not excluded or any(value != generator for value in excluded):
        raise ValueError(f"Checkpoint exclude_class metadata must consistently equal {generator}")
    return {"step": 15000, "model_mode": "dual", "exclude_class": generator,
            "mode_confirmation": "explicit_dual" if declarations else "legacy_dual_strict_load_required",
            "model_mode_declarations": declarations}


def validate_cache(metadata, records, features, expected_model=None):
    if metadata.get("schema") != SCHEMA or metadata.get("inventory_sha256") != digest(records):
        raise ValueError("CLIP cache path/order/content inventory mismatch; rebuild in a new cache directory")
    if metadata.get("n_images") != len(records) or len(features) != len(records):
        raise ValueError("CLIP cache file/feature count mismatch")
    if expected_model is not None and metadata.get("clip_identity") != expected_model:
        raise ValueError("CLIP cache model/revision/processor identity mismatch")
    dimension = metadata.get("feature_dim")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Invalid CLIP feature dimension")
    if hasattr(features, "shape"):
        import numpy as np
        if features.shape != (len(records), dimension) or features.dtype != np.float32:
            raise ValueError("CLIP features must have shape [image_count,feature_dim] and dtype float32")
        if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=2e-5, rtol=0):
            raise ValueError("CLIP features must be finite L2 unit vectors")
    else:
        for row in features:
            if len(row) != dimension or any(not math.isfinite(x) for x in row) or abs(sum(x * x for x in row) - 1) > 4e-5:
                raise ValueError("CLIP features must be complete finite L2 unit vectors")
    return True


def summarize(rows):
    if not rows:
        return []
    result = []
    for condition in sorted({row["condition"] for row in rows}):
        values = [row for row in rows if row["condition"] == condition]
        row = {"condition": condition, "n_seeds": len(values), "std_definition": "seed_population_ddof0"}
        for metric in ("acc", "ap", "auc"):
            samples = [float(value[metric]) for value in values]
            if any(not math.isfinite(x) for x in samples):
                raise ValueError("Non-finite metric")
            row[metric + "_mean"] = statistics.mean(samples)
            row[metric + "_std"] = statistics.pstdev(samples)
        result.append(row)
    return result
