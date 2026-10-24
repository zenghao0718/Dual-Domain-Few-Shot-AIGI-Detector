"""Revalidate saved R2-1 distances and build descriptive statistics/report.

No model inference or torch import. Requires numpy and matplotlib on AutoDL.
Never treats a query image as an independent reliability-weight observation.
"""

import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.r21_common import (  # noqa: E402
    GENERATORS, METRICS, MODES, SEEDS, alpha_reference, atomic_json,
    binary_metrics_reference, canonical_hash, class_rows, cluster_bootstrap,
    direction, quantile, read_json, sha256_file, spearman, validate_marker,
    validate_split, write_csv,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, tolerance=1e-6):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and abs(float(actual) - float(expected)) <= tolerance,
            f"Recomputation mismatch {name}: {actual} != {expected}")


def validate_task(output, key, identity, marker=None):
    """Validate checksum, full sampling/order, finite arrays and all metrics."""
    import numpy as np
    output = Path(output)
    task = output / "per_task" / key
    marker = read_json(task / "DONE.json") if marker is None else marker
    validate_marker(output, marker, identity)
    expected_files = {f"per_query/{key}.npz"} | {
        f"per_task/{key}/{name}" for name in
        ("support.npz", "support.json", "sampling.json.gz", "metrics.json", "class_metrics.json")}
    require(set(marker["files"]) == expected_files, "Incomplete task artifact checksum set")
    sampling = read_json(task / "sampling.json.gz")
    support = read_json(task / "support.json")
    require(support["identity_sha256"] == canonical_hash(identity), "Support identity mismatch")
    protocol = identity["protocol"]
    labels, indices, paths = [], [], []
    for cls, entry in enumerate(sampling["classes"]):
        require(entry["class_name"] == ("real" if cls == 0 else identity["generator"]), "Class order mismatch")
        dataset_manifest = read_json(output / "manifests" / (entry["class_name"] + ".json.gz"))
        require(canonical_hash(dataset_manifest) == identity["dataset_sha256"][entry["class_name"]],
                "Dataset inventory hash mismatch")
        validate_split(entry["support_indices"], entry["query_indices"], entry["dataset_size"],
                       protocol["support_shot"], identity["seed"])
        require(len(dataset_manifest) == entry["dataset_size"], "Dataset size mismatch")
        for subset in ("support", "query"):
            expected = [dataset_manifest[i]["relative_path"] for i in entry[subset + "_indices"]]
            require(entry[subset + "_paths"] == expected, "Manifest paths do not match official indices")
        labels.extend([cls] * len(entry["query_indices"]))
        indices.extend(entry["query_indices"])
        paths.extend(entry["query_paths"])
    require(len(sampling["classes"]) == 2, "Need exactly two classes")
    with np.load(task / "support.npz", allow_pickle=False) as saved:
        for mode in ("rgb", "freq"):
            z, proto = saved["support_" + mode], saved["proto_" + mode]
            require(z.ndim == 3 and z.shape[:2] == (protocol["support_shot"], 2)
                    and proto.shape == (2, z.shape[-1]), "Invalid support/prototype dimensions")
            require(np.isfinite(z).all() and np.isfinite(proto).all(), "Non-finite support")
            average = z.mean(axis=0)
            expected = average / np.maximum(np.linalg.norm(average, axis=-1, keepdims=True), 1e-12)
            require(np.allclose(proto, expected, atol=2e-6, rtol=1e-6), "Prototype mismatch")
            sigma = ((z - proto[None]) ** 2).sum(axis=-1).mean(axis=0)
            require(np.allclose(sigma, support["sigma_" + mode], atol=2e-6, rtol=1e-6), "Sigma mismatch")
            close(support["proto_sep_" + mode], max(float(((proto[0] - proto[1]) ** 2).sum()), 1e-12) ** 0.5,
                  "prototype separation", 2e-6)
    for cls in (0, 1):
        raw, alpha = alpha_reference(support["sigma_rgb"][cls], support["sigma_freq"][cls], protocol["tau_r"])
        close(support["alpha_raw"][cls], raw, "alpha_raw")
        close(support["alpha_rgb"][cls], alpha, "alpha")
    with np.load(output / "per_query" / (key + ".npz"), allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    require(str(arrays["identity_sha256"].item()) == canonical_hash(identity), "NPZ identity mismatch")
    require(str(arrays["generator"].item()) == identity["generator"] and int(arrays["seed"]) == identity["seed"],
            "NPZ generator/seed mismatch")
    require(arrays["labels"].tolist() == labels and arrays["source_index"].tolist() == indices
            and arrays["relative_path"].tolist() == paths, "Missing, reordered or mislabeled query")
    n = len(labels)
    require(marker.get("logit_parity") == "PASS" and marker.get("query_count") == n
            and marker.get("processed_query_count") == n, "Missing full-query/logit parity validation")
    alpha = np.asarray(support["alpha_rgb"], dtype=np.float32)
    for mode in MODES:
        distance, score, pred = (arrays[prefix + mode] for prefix in ("d_", "score_", "pred_"))
        require(distance.shape == (n, 2) and score.shape == pred.shape == (n,), "Invalid query dimensions")
        require(np.isfinite(distance).all() and (distance >= 0).all() and np.isfinite(score).all(),
                "Non-finite/negative distances or scores")
        logits = -distance.astype(np.float32) / protocol["tau"]
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        require(np.allclose(score, prob[:, 1], atol=2e-7, rtol=1e-6), "Saved fake score differs from softmax")
        require(set(pred.tolist()).issubset({0, 1}), "Invalid prediction label")
        # CPU numpy vs CUDA softmax can round a mathematical tie differently.
        # Saved prediction must maximize probability, within float32 roundoff.
        require((prob[np.arange(n), pred] >= prob.max(axis=1) - 2e-7).all(), "Saved prediction differs from softmax")
    expected_dual = alpha[None] * arrays["d_rgb"] + (1 - alpha[None]) * arrays["d_freq"]
    require(np.allclose(arrays["d_dual"], expected_dual, atol=2e-7, rtol=1e-6), "Class-specific fusion mismatch")
    stored_metrics = read_json(task / "metrics.json")
    recomputed = {}
    for mode in MODES:
        recomputed[mode] = binary_metrics_reference(labels, arrays["score_" + mode].tolist(),
                                                   arrays["pred_" + mode].tolist())
        for metric in METRICS:
            close(stored_metrics[mode][metric], recomputed[mode][metric], mode + "_" + metric, 1e-10)
    rows = class_rows(identity, support, arrays)
    stored_rows = read_json(task / "class_metrics.json")
    require(len(stored_rows) == 2, "Need two stored class rows")
    for row, stored in zip(rows, stored_rows):
        require(set(row) == set(stored), "Class columns mismatch")
        for name, value in row.items():
            if isinstance(value, (int, float)):
                close(stored[name], value, name, 1e-10)
            else:
                require(stored[name] == value, "Class identity mismatch")
    return rows, recomputed, marker


def parity_rows(output, manifest, tasks):
    """Reference outputs come exclusively from unmodified test_ddfsd.py."""
    rows = []
    for generator in GENERATORS:
        reference_dir = output / "formal" / generator
        try:
            marker = read_json(reference_dir / "REFERENCE_DONE.json")
            expected = {key: identity for key, identity in manifest["tasks"].items()
                        if identity["generator"] == generator}
            validate_marker(output, marker, {"tasks": expected})
            required = {f"formal/{generator}/{name}" for name in
                        ("ddfsd_eval_per_seed.csv", "ddfsd_eval_summary.csv", "config.json", "reference.log")}
            require(set(marker["files"]) == required, "Incomplete official reference artifacts")
            with open(reference_dir / "ddfsd_eval_per_seed.csv", encoding="utf-8", newline="") as handle:
                reference = list(csv.DictReader(handle))
            seeds = manifest["protocol"]["seeds"]
            require(len(reference) == len(seeds) and {int(r["seed"]) for r in reference} == set(seeds),
                    "Official reference seed coverage mismatch")
            by_seed = {int(r["seed"]): r for r in reference}
            protocol = manifest["protocol"]
            for seed, row in by_seed.items():
                task_identity = next(i for i in expected.values() if i["seed"] == seed)
                require(row["exclude_class"] == generator and row["model_mode"] == "dual"
                        and row["branch_mode"] == "dual" and int(row["ckpt_step"]) == 15000
                        and int(row["support_shot"]) == protocol["support_shot"]
                        and row["ckpt_path"] == task_identity["ckpt_path"]
                        and row["freq_stats_path"] == task_identity["stats_path"], "Official CSV protocol/path mismatch")
                for name, data_class in (("real", "real"), ("fake", generator)):
                    require(int(row["num_" + name + "_support"]) == protocol["support_shot"]
                            and int(row["num_" + name + "_query"]) == manifest["dataset_inventories"][data_class]["n_images"] - protocol["support_shot"],
                            "Official support/query count mismatch")
            config = read_json(reference_dir / "config.json")
            require(config["git_commit"] == manifest["git_commit"] and config["seeds"] == protocol["seeds"]
                    and config["shot"] == protocol["support_shot"] and config["tau"] == protocol["tau"]
                    and config["tau_r"] == protocol["tau_r"] and config["max_eval_query_per_class"] == 0
                    and config["strict_formal_eval_images"] is True, "Official config mismatch")
            error = ""
        except (OSError, ValueError, KeyError) as exc:
            by_seed, error = {}, str(exc)
        for key, metrics in tasks.items():
            identity = manifest["tasks"][key]
            if identity["generator"] != generator:
                continue
            row = {"generator": generator, "seed": identity["seed"], "logit_parity": "PASS",
                   "official_entrypoint_parity": "MISSING", "error": error}
            if identity["seed"] in by_seed:
                reference = by_seed[identity["seed"]]
                matches = True
                for metric in METRICS:
                    difference = abs(metrics["dual"][metric] - float(reference[metric]))
                    row[metric + "_diagnostic"] = metrics["dual"][metric]
                    row[metric + "_official"] = float(reference[metric])
                    row[metric + "_abs_diff"] = difference
                    matches &= math.isfinite(difference) and difference <= 1e-8
                row["official_entrypoint_parity"] = "PASS" if matches else "MISMATCH"
            rows.append(row)
    return rows


def correlation_rows(rows, repeats, bootstrap_seed):
    result = []
    groups = [("Real", [r for r in rows if r["class_label"] == 0]),
              ("Fake", [r for r in rows if r["class_label"] == 1]), ("Overall", rows)]
    groups.extend((generator, [r for r in rows if r["generator"] == generator]) for generator in GENERATORS)
    for name, group in groups:
        for target in ("Y", "delta_recall"):
            xs, ys = [r["X"] for r in group], [r[target] for r in group]
            row = {"group": name, "target": target, "rho": spearman(xs, ys), **direction(xs, ys)}
            row.update(cluster_bootstrap(group, target, repeats, bootstrap_seed))
            row["undefined_reason"] = "fewer than two observations or constant ranks" if math.isnan(row["rho"]) else ""
            result.append(row)
    return result


def failures(rows):
    """Publish all directional failures and predefined compact/weak/poor cases."""
    thresholds = {}
    for mode in ("rgb", "freq"):
        thresholds[mode] = {
            "compact_sigma_q25": quantile([r["sigma_" + mode] for r in rows], 0.25),
            "weak_separation_q25": quantile([r["proto_sep_" + mode] for r in rows], 0.25)}
    result = []
    for row in rows:
        reasons = []
        if row["X"] * row["Y"] < 0:
            reasons.append("weight_margin_disagreement")
        for mode in ("rgb", "freq"):
            if (row["sigma_" + mode] <= thresholds[mode]["compact_sigma_q25"]
                    and row["proto_sep_" + mode] <= thresholds[mode]["weak_separation_q25"]
                    and (row["mean_margin_" + mode] <= 0 or row["recall_" + mode] <= 0.5)):
                reasons.append(mode + "_compact_weak_poor")
        if reasons:
            result.append({**row, "reasons": ";".join(reasons)})
    return result, thresholds


def plot_figures(output, rows, correlations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    colors = dict(zip(GENERATORS, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(9, 6))
    for generator in GENERATORS:
        for cls, symbol in ((0, "o"), (1, "^")):
            group = [r for r in rows if r["generator"] == generator and r["class_label"] == cls]
            ax.scatter([r["X"] for r in group], [r["Y"] for r in group], marker=symbol,
                       color=colors[generator], label=generator + "/" + ("Real", "Fake")[cls], alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(xlabel="X = alpha_RGB - 0.5", ylabel="Y = mean margin_RGB - mean margin_Frequency",
           title=f"Support reliability vs class query margin ({len(rows)} observations)")
    ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1))
    fig.savefig(folder / "alpha_margin_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ("rho", "agreement"), ("Spearman rho (descriptive)", "Non-tie direction agreement")):
        values = [next(r[key] for r in correlations if r["group"] == g and r["target"] == "Y") for g in GENERATORS]
        ax.bar(GENERATORS, values, color=[colors[g] for g in GENERATORS])
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(title)
        ax.set_ylim((-1, 1) if key == "rho" else (0, 1))
        for i, value in enumerate(values):
            if not math.isfinite(value):
                ax.text(i, 0.03, "undefined", ha="center", rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "generator_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cls in (0, 1):
        group = [r for r in rows if r["class_label"] == cls]
        axes[0].hist([r["alpha_raw"] for r in group], bins=np.linspace(0, 1, 21), alpha=0.5, label=("Real", "Fake")[cls])
        axes[1].hist([r["alpha_rgb"] for r in group], bins=np.linspace(0.25, 0.75, 16), alpha=0.5)
    axes[0].set_title("Raw sigmoid weights")
    axes[0].legend()
    axes[1].set_title("Applied clipped RGB weights")
    ratios = []
    for generator in GENERATORS:
        group = [r for r in rows if r["generator"] == generator]
        ratios.append(statistics.mean(r["alpha_clipped"] for r in group) if group else math.nan)
    axes[2].bar(GENERATORS, ratios, color=[colors[g] for g in GENERATORS])
    axes[2].set(title="Strict clipping fraction (raw < .25 or > .75)", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(folder / "alpha_clipping.png", dpi=180)
    plt.close(fig)


def markdown_table(rows, fields):
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.6f}" if math.isfinite(value) else "undefined/NaN"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
                     + ["| " + " | ".join(format_value(row.get(f, "")) for f in fields) + " |" for row in rows])


def verify_csv(path, expected):
    with open(path, encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    require(len(actual) == len(expected), "CSV row count mismatch")
    for saved, row in zip(actual, expected):
        require(set(saved) == set(row), "CSV columns mismatch")
        for key, value in row.items():
            require(saved[key] == str(value), "CSV roundtrip mismatch: " + key)


def package_results(output):
    """Small review bundle, excluding raw arrays, inventories and sampling lists."""
    output = Path(output).resolve()
    require(output != ROOT and ROOT not in output.parents, "OUTPUT_DIR must be outside the repository")
    paths = [p for p in output.iterdir() if p.suffix in {".json", ".csv", ".md"} and p.is_file()
             and p.name != "review_checksums.json"]
    paths += list((output / "figures").glob("*.png"))
    paths += [p for p in (output / "formal").glob("*/*") if p.suffix in {".csv", ".json", ".log"}]
    paths += [p for p in (output / "per_task").glob("*/*.json") if p.is_file()]
    paths += [output / "logs" / name for name in ("preflight_validation.json", "last_failure.json", "last_traceback.txt")
              if (output / "logs" / name).is_file()]
    checksums = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(paths)}
    atomic_json(output / "review_checksums.json", checksums)
    paths.append(output / "review_checksums.json")
    archive = output.parent / (output.name + "_review.tar.gz")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in sorted(set(paths)):
            bundle.add(path, arcname=output.name + "/" + path.relative_to(output).as_posix(), recursive=False)
    temporary.replace(archive)
    return archive


def analyze(output, bootstrap_repeats=10000, bootstrap_seed=2101, package=False):
    from tools.r21_common import atomic_bytes
    output = Path(output).resolve()
    require(output != ROOT and ROOT not in output.parents, "OUTPUT_DIR must be outside the repository")
    manifest = read_json(output / "run_manifest.json")
    require(bootstrap_repeats > 0, "bootstrap-repeats must be positive")
    protocol = manifest["protocol"]
    expected_keys = {f"{g}_seed{s}" for g in GENERATORS for s in protocol["seeds"]}
    require(set(manifest["tasks"]) == expected_keys and protocol["generators"] == list(GENERATORS)
            and protocol["all_queries"] is True and protocol["max_query_per_class"] == 0,
            "Run manifest omits expected generators/seeds or truncates queries")
    for key, identity in manifest["tasks"].items():
        require(key == f"{identity['generator']}_seed{identity['seed']}" and identity["protocol"] == protocol,
                "Task/run protocol mismatch")
    rows, tasks, errors = [], {}, []
    for key, identity in manifest["tasks"].items():
        if not (output / "per_task" / key / "DONE.json").exists():
            continue
        try:
            classes, metrics, _ = validate_task(output, key, identity)
            rows.extend(classes)
            tasks[key] = metrics
        except (OSError, ValueError, KeyError) as error:
            errors.append({"task": key, "error": str(error)})
    if not rows:
        raise ValueError("No validated task results to analyze; no report was generated")
    task_rows = []
    for key, metrics in tasks.items():
        identity = manifest["tasks"][key]
        row = {name: identity[name] for name in ("generator", "seed", "ckpt_sha256", "stats_sha256")}
        row.update({mode + "_" + metric: metrics[mode][metric] for mode in MODES for metric in METRICS})
        task_rows.append(row)
    parity = parity_rows(output, manifest, tasks)
    correlations = correlation_rows(rows, bootstrap_repeats, bootstrap_seed)
    failed, thresholds = failures(rows)
    distributions = []
    for name, subset in (("Real", [r for r in rows if r["class_label"] == 0]),
                         ("Fake", [r for r in rows if r["class_label"] == 1]), ("Overall", rows)):
        for field in ("sigma_rgb", "sigma_freq", "alpha_raw", "alpha_rgb", "proto_sep_rgb", "proto_sep_freq"):
            values = [r[field] for r in subset]
            distributions.append({"group": name, "variable": field, "n": len(values),
                                  "mean": statistics.mean(values) if values else math.nan,
                                  **{key: quantile(values, q) for key, q in
                                     (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1))}})
    summaries = []
    for generator in GENERATORS:
        subset = [r for r in rows if r["generator"] == generator]
        measurements = [r for r in task_rows if r["generator"] == generator]
        if not subset:
            continue
        summary = {"generator": generator, "n_tasks": len(measurements), "n_class_rows": len(subset),
                   "alpha_clip_ratio": statistics.mean(r["alpha_clipped"] for r in subset),
                   "alpha_bound_ratio": statistics.mean(r["alpha_at_bound"] for r in subset),
                   "alpha_mean": statistics.mean(r["alpha_rgb"] for r in subset)}
        for mode in MODES:
            for metric in METRICS:
                values = [r[mode + "_" + metric] for r in measurements]
                summary[mode + "_" + metric + "_mean"] = statistics.mean(values)
                summary[mode + "_" + metric + "_std"] = statistics.pstdev(values)
        summaries.append(summary)
    fields = list(rows[0]) + ["reasons"]
    for name, values, columns in (("episode_class_metrics.csv", rows, None),
                                  ("task_metrics.csv", task_rows, None),
                                  ("generator_summary.csv", summaries, None),
                                  ("distribution_summary.csv", distributions, None),
                                  ("correlation_results.csv", correlations, None),
                                  ("mismatched_cases.csv", failed, fields)):
        write_csv(output / name, values, columns)
        if values:
            verify_csv(output / name, values)
    parity_fields = ["generator", "seed", "logit_parity", "official_entrypoint_parity", "error"]
    parity_fields += [metric + suffix for metric in METRICS for suffix in ("_diagnostic", "_official", "_abs_diff")]
    write_csv(output / "parity_validation.csv", parity, parity_fields)
    protocol = manifest["protocol"]
    canonical = (protocol["seeds"] == list(SEEDS) and protocol["support_shot"] == 10
                 and protocol["tau"] == 0.2 and protocol["tau_r"] == 0.1 and protocol["use_fp16"] is True)
    expected = len(manifest["tasks"])
    all_done = len(tasks) == expected and len(rows) == 2 * expected and not errors
    matching = all_done and len(parity) == expected and all(r["official_entrypoint_parity"] == "PASS" for r in parity)
    status = ("COMPLETE_AND_MATCHING" if canonical and expected == 30 else "CUSTOM_COMPLETE_AND_MATCHING") if matching else (
        f"PARTIAL_{len(tasks)}_OF_{expected}" if not all_done else "PARITY_PENDING_OR_MISMATCH")
    plot_figures(output, rows, correlations)
    macro = {metric: statistics.mean(r["dual_" + metric + "_mean"] for r in summaries) for metric in ("acc", "ap", "auc")}
    reference = {"acc": 0.9110, "ap": 0.9221, "auc": 0.9561}
    paper_check = [{"metric": k, "actual": macro[k], "paper_reference": reference[k],
                    "difference_percentage_points": 100 * (macro[k] - reference[k])} for k in reference]
    validation = {"status": status, "completed_tasks": len(tasks), "expected_tasks": expected,
                  "class_rows": len(rows), "formal_protocol": canonical, "errors": errors,
                  "all_queries_checked": True, "csv_recomputed_and_roundtripped": True,
                  "parity_matching": matching, "failure_thresholds": thresholds,
                  "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": bootstrap_seed,
                  "paper_reference_check": paper_check,
                  "artifact_sha256": {p.name: sha256_file(p) for p in output.glob("*.csv")}}
    report = ["# R2-1 支持集可靠性分析", "", f"状态：**{status} ({len(tasks)}/{expected})**。",
              "只有 COMPLETE_AND_MATCHING (30/30) 表示默认完整实验且已通过原正式入口核查。自定义协议另行标记。",
              "", "## Material Passport", "原始证据：本目录经校验的逐图距离、支持集特征和任务记录；不包含本机合成测试。",
              "验证级别：距离与指标复算；正式入口核查状态见下表。结果解释限于描述性关联。",
              "", "## 审稿意见与动机",
              "Reviewer 2 Comment 1 质疑：低类内离散度不保证可判别性，紧凑的 Real/Fake 簇仍可能重叠。",
              "本实验检验支持集计算的类别权重与同一类别全量 Query 的实际判别间隔是否对应。不预设结果支持设计。",
              "", "## 定义与协议",
              "Real margin=d_fake−d_real；Fake margin=d_real−d_fake。正值表示该分支更支持真类。",
              "X=alpha_rgb−0.5；Y=mean_margin_rgb−mean_margin_freq；每类只统计该真标签 Query。",
              "类别 Recall 基于正式 softmax/argmax，精确平局预测 Real；间隔零不计为正间隔。",
              "方向一致率仅以 X、Y 均非零记录为分母；平局、覆盖率独立列出。RGB/Frequency 来自同一最终 Dual checkpoint。",
              "权重、sigma、原型、距离均复用仓库现有函数，clip 固定 [.25,.75]，fake=1，分数为 softmax(-distance/tau)。",
              "完整配置、代码版本、环境、命令与全部任务身份见 [run_manifest.json](run_manifest.json)。",
              "```json", __import__("json").dumps(protocol, indent=2, ensure_ascii=False), "```",
              "```json", __import__("json").dumps(manifest["environment"], indent=2, ensure_ascii=False), "```",
              "", "## 全部任务指标（比例单位，含同模型分支推理）",
              markdown_table(task_rows, ["generator", "seed"] + [m + "_" + k for m in MODES for k in ("acc", "ap", "auc")]),
              "", "## Real/Fake/Overall 及生成器相关性与方向一致率",
              "Overall 为描述性统计。Spearman 使用平均秩处理并列；常数输入返回 undefined/NaN，绝不填 0。",
              f"按生成器 cluster bootstrap {bootstrap_repeats} 次，固定 seed={bootstrap_seed}；保留整个 seed/类别簇，报告百分位 95% 区间及有效重采样数。",
              "只有六个生成器，区间稳定性有限；单生成器无法估计生成器层区间。未计算独立样本显著性 p 值。",
              markdown_table(correlations, ["group", "target", "n", "rho", "rho_ci_low", "rho_ci_high", "rho_bootstrap_valid",
                                             "agreement", "agreement_ci_low", "agreement_ci_high", "n_non_tie", "n_tie", "n_x_tie", "n_y_tie", "coverage", "undefined_reason"]),
              "", "![alpha 与间隔](figures/alpha_margin_scatter.png)",
              "![生成器对照](figures/generator_comparison.png)", "", "## 六个生成器分表"]
    for generator in GENERATORS:
        report += ["", "### " + generator, markdown_table([r for r in rows if r["generator"] == generator],
                   ["seed", "class_name", "n_query", "sigma_rgb", "sigma_freq", "alpha_raw", "alpha_rgb",
                    "proto_sep_rgb", "proto_sep_freq", "mean_margin_rgb", "mean_margin_freq", "recall_rgb", "recall_freq", "recall_dual", "X", "Y"])]
    report += ["", "## 权重分布和裁剪", markdown_table(summaries, ["generator", "n_tasks", "alpha_mean", "alpha_clip_ratio", "alpha_bound_ratio"]),
               "clip_ratio 指 raw 严格落于区间外；bound_ratio 指实际 alpha 等于边界。原始分布保留在类别行中。",
               markdown_table(distributions, ["group", "variable", "n", "mean", "min", "q25", "median", "q75", "max"]),
               "![权重裁剪](figures/alpha_clipping.png)", "", "## 反例与紧凑但弱分离筛选",
               "全部 X*Y<0 的方向失败均列出，未挑选正面数据。每域独立用所有已验证类别行的 sigma≤Q25、原型分离≤Q25，且平均间隔≤0 或 Recall≤.5 筛选紧凑/弱分离/差判别。",
               "分位数使用线性插值；重复于两个类别的原型分离均保留，部分运行阈值仅代表已完成子集。",
               "```json", __import__("json").dumps(thresholds, indent=2, ensure_ascii=False), "```",
               markdown_table(failed, ["generator", "seed", "class_name", "X", "Y", "reasons"]),
               f"筛选记录共 {len(failed)} 条；无满足阈值记录时不虚构失败例。完整字段见 mismatched_cases.csv。",
               "", "## Parity 与论文参考值", markdown_table(parity, ["generator", "seed", "logit_parity", "official_entrypoint_parity", "error"]),
               "每个 Query 批次的 Dual logits 在保存前与 compute_query_logits(branch_mode='dual') 严格核查；再运行原 test_ddfsd.py 比较每 seed 的六项指标，绝对容差 1e-8。",
               "论文 Full Model 的 mACC=91.10%、mAP=92.21%、mAUC=95.61% 仅作参考；以下从真实结果先按 seed、再按生成器宏平均。",
               markdown_table(paper_check, ["metric", "actual", "paper_reference", "difference_percentage_points"]),
               "出现差异应核对 checkpoint/stats SHA、数据版本及数量、支持抽样、预处理、精度和硬件；目前没有证据将差异归因于某一个因素。",
               "", "## 局限与复现",
               "Query 跨 seed 重叠，支持集记录也相关；不能当作独立显著性证据。仅分析最终六模型的五次支持抽样，不构成因果检验；均值间隔受各域距离尺度影响。",
               "自定义参数、未完成任务或 parity 缺失不构成默认完整结果。小归档可复核汇总，但逐图独立复算还需原目录中的 per_query/per_task/manifests。",
               "原命令（路径为本次实际路径）：", "```sh", manifest["reproduction_command"], "```",
               "", "## R2-1 英文证据摘要（据真实输出生成）"]
    core = [r for r in correlations if r["target"] == "Y" and r["group"] in ("Real", "Fake", "Overall")]
    report += [f"We analyzed {len(tasks)} validated generator-seed episodes ({len(rows)} class-level observations; status: {status}). "
               "Each observation pairs the support-derived RGB preference with the RGB-minus-frequency mean query margin for the same true class. "
               + "; ".join(f"{r['group']}: Spearman rho={r['rho']:.4f}, non-tie agreement={r['agreement']:.4f}, coverage={r['coverage']:.4f}" for r in core)
               + ". Overall results are descriptive. Generator-cluster bootstrap intervals and all directional counterexamples are reported above. "
               "These observations quantify alignment and failure cases without assuming that compactness guarantees discrimination. "
               "RGB-only and frequency-only inference use branches of the same dual checkpoint, not separately trained ablations.", ""]
    atomic_bytes(output / "R21_REPORT.md", "\n".join(report).encode("utf-8"))
    validation["artifact_sha256"].update({p.relative_to(output).as_posix(): sha256_file(p) for p in
                                         [output / "run_manifest.json", output / "R21_REPORT.md"]
                                         + list((output / "figures").glob("*.png"))})
    atomic_json(output / "completion_validation.json", validation)
    # Global completion is published after all CSVs, figures and report exist.
    atomic_json(output / "run_status.json", {k: validation[k] for k in ("status", "completed_tasks", "expected_tasks", "class_rows")})
    if package:
        print("Review archive:", package_results(output))
    print(status, f"({len(tasks)}/{expected});", output / "R21_REPORT.md")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2101)
    parser.add_argument("--package", action="store_true", help="Write small review archive next to OUTPUT_DIR")
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")
    result = analyze(args.output_dir, args.bootstrap_repeats, args.bootstrap_seed, args.package)
    return 0 if "COMPLETE_AND_MATCHING" in result["status"] else 2


if __name__ == "__main__":
    sys.exit(main())
