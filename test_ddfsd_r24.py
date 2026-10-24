"""Frozen DDFSD R2-4: label-noise and CLIP-concentrated supports, ALL queries."""

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent

from util.ddfsd_r24 import (CLIP_ID, GENERATORS, NOISE_RATES, SEEDS, SHOT, condition_name,
                            clip_selection, digest, file_sha256, label_noise_selection,
                            noise_rates_from_string, outside_repo, read_json, seeds_from_string,
                            paper_macro_by_seed, summarize, support_rows, validate_selection, write_csv, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "label_noise", "clip_bias"), default="all")
    parser.add_argument("--exclude-class", choices=("all",) + GENERATORS, default="all")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT"))
    parser.add_argument("--ckpt-path", default=os.environ.get("CKPT_PATH"), help="Single-generator exact path")
    parser.add_argument("--freq-stats-path", default=os.environ.get("FREQ_STATS_PATH"), help="Single-generator paired stats")
    parser.add_argument("--checkpoint-template", default="exclude_{generator}/ckpt/ddfsd_step[15000].pth")
    parser.add_argument("--stats-template", default="exclude_{generator}/freq_stats.pt")
    parser.add_argument("--clip-cache-root", default=os.environ.get("CLIP_CACHE_ROOT"))
    parser.add_argument("--clip-model-id", default=CLIP_ID, help="Must match cache metadata; model is not loaded by this evaluator")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--noise-rates", default=",".join(map(str, NOISE_RATES)), help="0,0.1,0.2,0.3; zero is optional clean parity only")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-fp16", choices=("true", "false"), default="true")
    parser.add_argument("--baseline-reference", default="NOT_PROVIDED", help="Citation/path to existing paper clean baseline; never triggers inference")
    parser.add_argument("--baseline-csv", help="Optional existing formal per-seed CSV, one generator or concatenated six; never reruns baseline")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--validate-only", action="store_true", help="Audit all resources and caches/load checkpoints, no forward")
    args = parser.parse_args(argv)
    if not args.data_root or not args.output_dir or (not args.ckpt_root and not args.ckpt_path):
        parser.error("Set DATA_ROOT/OUTPUT_DIR and CKPT_ROOT or exact --ckpt-path with --freq-stats-path")
    if args.exclude_class == "all" and (args.ckpt_path or args.freq_stats_path):
        parser.error("Exact checkpoint/stats paths require one --exclude-class; use CKPT_ROOT for all six")
    if bool(args.ckpt_path) != bool(args.freq_stats_path):
        parser.error("--ckpt-path and --freq-stats-path must be provided together")
    if args.experiment in ("all", "clip_bias") and not args.clip_cache_root:
        parser.error("CLIP bias requires --clip-cache-root / CLIP_CACHE_ROOT; build the offline cache first")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("Positive batch size and nonnegative workers required")
    try:
        args.seeds = seeds_from_string(args.seeds)
        args.noise_rates = noise_rates_from_string(args.noise_rates)
        args.output_dir = outside_repo(args.output_dir, ROOT)
        args.data_root = Path(args.data_root).expanduser().resolve()
        for name in ("ckpt_root", "clip_cache_root"):
            if getattr(args, name):
                setattr(args, name, outside_repo(getattr(args, name), ROOT))
        if args.ckpt_path:
            args.ckpt_path, args.freq_stats_path = outside_repo(args.ckpt_path, ROOT), outside_repo(args.freq_stats_path, ROOT)
        for source in (args.data_root, args.ckpt_root, args.clip_cache_root):
            if source and (source == args.output_dir or source in args.output_dir.parents or args.output_dir in source.parents):
                parser.error("Output must be disjoint from data/checkpoint/cache roots")
    except ValueError as error:
        parser.error(str(error))
    args.generators = list(GENERATORS) if args.exclude_class == "all" else [args.exclude_class]
    args.use_fp16 = args.use_fp16 == "true"
    return args


def checkpoint_paths(args, generator):
    if args.ckpt_path:
        paths = args.ckpt_path, args.freq_stats_path
    else:
        task_root = (args.ckpt_root / ("exclude_" + generator)).resolve()
        paths = tuple((args.ckpt_root / template.format(generator=generator)).resolve()
                      for template in (args.checkpoint_template, args.stats_template))
        if any(task_root not in p.parents for p in paths):
            raise ValueError("Checkpoint/stats templates must resolve within the matching exclude_" + generator)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/stats: {path}. Check CKPT_ROOT nesting and exact --checkpoint-template")
    if paths[1].parent != paths[0].parent and not (paths[0].parent.name == "ckpt" and paths[1].parent == paths[0].parent.parent):
        raise ValueError("Checkpoint/freq_stats task directories do not match")
    return paths


def existing_baseline(args):
    if not args.baseline_csv:
        return [], {"reference": args.baseline_reference, "status": "NOT_PROVIDED_NO_CLEAN_RUN_REQUESTED"}
    path = Path(args.baseline_csv).expanduser().resolve()
    with open(path, encoding="utf-8-sig", newline="") as handle:
        values = list(csv.DictReader(handle))
    rows = []
    for generator in args.generators:
        source = [r for r in values if r.get("generator", r.get("exclude_class")) == generator]
        if len(source) != 5 or {int(r["seed"]) for r in source} != set(SEEDS):
            raise ValueError("Existing clean baseline CSV needs exactly the five formal seeds for " + generator)
        for row in source:
            if (row.get("model_mode", "dual") != "dual" or row.get("branch_mode", "dual") != "dual"
                    or int(row.get("support_shot", "10")) != 10 or int(row.get("ckpt_step", "15000")) != 15000):
                raise ValueError("Existing baseline CSV conflicts with the formal clean protocol")
            if any(not math.isfinite(float(row[k])) or not 0 <= float(row[k]) <= 1 for k in ("acc", "ap", "auc")):
                raise ValueError("Existing clean baseline CSV has invalid metrics")
        summary = summarize([{**row, "condition": "EXTERNAL_EXISTING_CLEAN_BASELINE"} for row in source])[0]
        rows.append({"generator": generator, **summary})
    return rows, {"reference": args.baseline_reference, "csv_path": str(path), "csv_sha256": file_sha256(path),
                  "status": "EXTERNAL_EXISTING_FORMAL_CSV_NOT_RERUN"}


def load_case(path, identity, selection, sizes, marker=None):
    marker = read_json(path / "DONE.json") if marker is None else marker
    if marker.get("identity") != identity:
        raise ValueError("Case identity changed; use a NEW OUTPUT_DIR")
    if marker.get("status") != "DONE" or marker.get("sha256") != file_sha256(path / "result.json.gz"):
        raise ValueError("Corrupt/incomplete task result")
    result = read_json(path / "result.json.gz")
    if result.get("selection") != read_json_value(selection) or result.get("identity") != identity:
        raise ValueError("Saved support/query selection differs from deterministic recomputation")
    validate_selection(result["selection"], *sizes)
    metrics = result["metrics"]
    for key in ("acc", "ap", "auc", "alpha_real", "alpha_fake"):
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError("Invalid recorded metric: " + key)
    if any(not .25 <= metrics[key] <= .75 for key in ("alpha_real", "alpha_fake")):
        raise ValueError("Recorded adaptive alpha is outside [.25,.75]")
    for cls, name in enumerate(("real", "fake")):
        if metrics["num_" + name + "_support"] != SHOT or metrics["num_" + name + "_query"] != sizes[cls] - SHOT:
            raise ValueError("Incomplete support/query counts in result")
    return result


def read_json_value(value):
    # Canonicalize tuple source/index entries for comparisons after JSON load.
    import json
    return json.loads(json.dumps(value, allow_nan=False))


def summarize_run(args, jobs, completed, baseline):
    root = args.output_dir / "r24"
    summaries = []
    raw_rows = [{"generator": result["generator"], "seed": result["seed"], "condition": result["condition"],
                 **{metric: result["metrics"][metric] for metric in ("acc", "ap", "auc")}}
                for result in completed.values()]
    macro_per_seed = []
    diagnostics_all = []
    for generator in args.generators:
        for experiment in ("label_noise", "clip_bias"):
            group = [r for r in completed.values() if r["generator"] == generator and r["experiment"] == experiment]
            if not group:
                continue
            folder = root / ("exclude_" + generator) / experiment
            per_seed, manifests, concentration = [], [], []
            for result in group:
                metadata, metrics = result["identity"], result["metrics"]
                row = {"generator": generator, "seed": result["seed"], "experiment": experiment, "condition": result["condition"],
                       "noise_rate": result["selection"].get("noise_rate", ""),
                       "swapped_per_class": result["selection"].get("swapped_per_class", 0),
                       "mislabeled_total": result["selection"].get("mislabeled_total", 0),
                       **{key: metrics[key] for key in ("num_real_support", "num_fake_support", "num_real_query", "num_fake_query",
                                                       "acc", "ap", "auc", "alpha_real", "alpha_fake")},
                       "ckpt_sha256": metadata["checkpoint"]["ckpt_sha256"], "stats_sha256": metadata["checkpoint"]["stats_sha256"],
                       "sampling_sha256": digest(result["selection"]), "cache_sha256": metadata["cache_sha256"],
                       "concentration_status": result["selection"].get("diagnostics", {}).get("concentration_status", ""),
                       "is_new_clean_experiment": False}
                per_seed.append(row)
                manifests.extend(result["support_manifest"])
                if "diagnostics" in result["selection"]:
                    concentration.append({"generator": generator, "seed": result["seed"], **result["selection"]["diagnostics"]})
            summary = [{"generator": generator, **r, "status": "COMPLETE_REQUESTED_SEEDS" if r["n_seeds"] == len(args.seeds) else "PARTIAL"}
                       for r in summarize(per_seed)]
            write_csv(folder / "per_seed.csv", per_seed)
            write_csv(folder / "summary.csv", summary)
            write_csv(folder / "support_manifest.csv", manifests)
            write_json(folder / "config.json", {"experiment": experiment, "generator": generator,
                       "requested_seeds": args.seeds, "support_shot": 10, "tau": .2, "tau_r": .1,
                       "class_mapping": {"real": 0, "target_fake": 1, "valid_generators": list(GENERATORS)},
                       "alpha_clip": [.25, .75], "all_queries": True, "model_mode": "dual", "use_fp16": args.use_fp16,
                       "baseline": baseline, "task_identities": [r["identity"] for r in group],
                       "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            if concentration:
                write_csv(folder / "clip_concentration.csv", concentration)
                diagnostics_all.extend(concentration)
            summaries.extend({"scope": "GENERATOR", **row, "n_generators": 1} for row in summary)
    for condition in sorted({row["condition"] for row in raw_rows}):
        macro = paper_macro_by_seed(raw_rows, condition)
        if macro is not None and args.generators == list(GENERATORS):
            summaries.append(macro["summary"])
            macro_per_seed.extend(macro["per_seed"])
    # Rewrite even empty tables so partial/empty input cannot leave stale macros.
    write_csv(root / "all_generators_summary.csv", summaries,
              ["scope", "generator", "condition", "n_seeds", "std_definition", "acc_mean", "acc_std",
               "ap_mean", "ap_std", "auc_mean", "auc_std", "status", "n_generators"])
    write_csv(root / "macro_per_seed.csv", macro_per_seed,
              ["condition", "seed", "n_generators", "acc", "ap", "auc"])
    comparisons = []
    for row in summaries:
        if row["scope"] != "GENERATOR":
            continue
        clean = next((r for r in baseline.get("summary", []) if r["generator"] == row["generator"]), None)
        if clean is not None:
            comparison = {"generator": row["generator"], "condition": row["condition"],
                          "baseline_source": "EXTERNAL_EXISTING_FORMAL_CSV_NOT_RERUN"}
            for metric in ("acc", "ap", "auc"):
                comparison.update({metric + "_clean_existing_mean": clean[metric + "_mean"],
                                   metric + "_condition_mean": row[metric + "_mean"],
                                   metric + "_delta": row[metric + "_mean"] - clean[metric + "_mean"]})
            comparisons.append(comparison)
    if comparisons:
        write_csv(root / "baseline_comparison.csv", comparisons)
    full_protocol = (args.generators == list(GENERATORS) and args.seeds == list(SEEDS) and args.experiment == "all"
                     and args.noise_rates == list(NOISE_RATES))
    status = ("COMPLETE_R24_A_B" if full_protocol else "COMPLETE_REQUESTED_SCOPE") if len(completed) == len(jobs) else f"PARTIAL_{len(completed)}_OF_{len(jobs)}"
    text = ["# R2-4 支持集质量敏感性实验", "", f"状态：{status} ({len(completed)}/{len(jobs)})。", "",
            "A：按原采样构建 10 Real+10 Fake；10/20/30% 分别交换每类 1/2/3 张，共误标 2/4/6 张。共享稳定随机排列前缀保证嵌套，Query 和真值均不变。",
            "B：每 seed 从目标任务 real/val 稳定随机选一张真实锚点，同锚点余弦 Top10 Real/Fake。Real 必含锚点，其余相似度平局按官方路径索引升序。",
            "每条件各自剩余全部 val 为 Query；CLIP 组与随机组不人为共用 Query 池。CLIP 不参与 DDFSD 特征或预测。",
            "冻结最终双分支 step15000 checkpoint，tau=.2、tau_r=.1、alpha clip=[.25,.75]；RGB/Haar 和频域统计使用原正式代码。",
            "原论文 Clean Baseline 仅引用已有证据，不重新运行；0%（若显式选用）只是一致性检查。",
            "Baseline 来源：" + repr(baseline), "",
            "每生成器自身的 mean±std 仍在其五个 seeds 上计算，使用 statistics.pstdev(ddof=0)。",
            "论文口径宏统计：对每个条件、每个 seed 先跨六生成器等权平均，再对五个 seed 宏平均值求 mean±pstdev(ddof=0)，分别得到 mACC、mAP、mAUC。",
            "宏统计只发布六类×正式五 seeds 的完整唯一有效网格；局部/未完成条件不输出宏行。逐 seed 宏值见 macro_per_seed.csv。",
            "R2-4 复用每生成器的同一最终 checkpoint，五次波动来自支持集采样 seeds；统计聚合口径对齐论文，不声称使用五个独立训练 checkpoint。",
            "CSV 保留原始浮点精度；下表以百分比显示两位小数。", "", "|scope|generator|condition|n_seeds|ACC mean±std|AP mean±std|AUC mean±std|",
            "|---|---|---|---|---|---|---|"]
    for row in summaries:
        text.append("|" + "|".join([str(row[k]) for k in ("scope", "generator", "condition", "n_seeds")]
                                  + [f"{100 * row[m + '_mean']:.2f} ± {100 * row[m + '_std']:.2f}" for m in ("acc", "ap", "auc")]) + "|")
    text += ["", "## 已有 Clean Baseline 引用", "不重新跑基线；以下仅在提供可信正式 CSV 时引用其真实数据，差值=本条件均值−已有 clean 均值。未提供 CSV 时不生成下降幅度。",
             "|generator|condition|ΔACC (pp)|ΔAP (pp)|ΔAUC (pp)|", "|---|---|---|---|---|"]
    for row in comparisons:
        text.append("|" + "|".join([row["generator"], row["condition"]] + [f"{100 * row[m + '_delta']:.2f}" for m in ("acc", "ap", "auc")]) + "|")
    text += ["", "## 语义集中诊断", "支持集内部排除自身的平均两两 CLIP cosine，与同 seed 正式随机 Support 比较；Real/Fake 任一类未严格高于 Random，则标记 CONCENTRATION_NOT_VERIFIED 并保留。",
             "|generator|seed|Real Δ pairwise|Fake Δ pairwise|status|", "|---|---|---|---|---|"]
    for row in diagnostics_all:
        text.append(f"|{row['generator']}|{row['seed']}|{row['real_pairwise_delta']:.6f}|{row['fake_pairwise_delta']:.6f}|{row['concentration_status']}|")
    text += ["", "CLIP 语义相似不等于人工类别标注，不据此声称图片全部属于某个物体类别。未确认集中程度的 seed 不重选锚点、不筛除。",
             "各条件 Query 因 Support 不同而可能不同，此处保留正式协议；跨 seed Query 重叠，结果为描述性敏感性分析。",
             "支持来源、观测标签、交换索引、锚点及其相似度见 support_manifest.csv；完整原始 Support/Query 索引与统计见 per_task 中的 result.json.gz。",
             "环境、权重 SHA、数据/缓存身份、原命令与 Git commit 见 run_config.json 以及各实验 config.json。", ""]
    from util.ddfsd_r24 import atomic_write
    atomic_write(root / "R24_REPORT.md", "\n".join(text).encode())
    write_json(root / "run_status.json", {"status": status, "completed": len(completed), "expected": len(jobs),
               "full_six_generator_five_seed_both_experiments": full_protocol,
               "concentration_not_verified": sum(r["concentration_status"] == "CONCENTRATION_NOT_VERIFIED" for r in diagnostics_all)})


def execute(args):
    import torch
    from util.ddfsd_eval import _evaluate_binary_few_shot_impl
    from util.ddfsd_r24_runtime import audited_datasets, code_identity, environment, load_cache, load_model
    from util.utils import set_seed
    root = args.output_dir / "r24"
    root.mkdir(parents=True, exist_ok=True)
    code = code_identity(ROOT)
    baseline_rows, baseline = existing_baseline(args)
    if baseline_rows:
        baseline["summary"] = baseline_rows
    # Resolve every requested checkpoint/stats pair before scanning images.
    paths = {g: checkpoint_paths(args, g) for g in args.generators}
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = {**environment(), "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    datasets, records = audited_datasets(args.data_root, ["real", *args.generators], root / "logs")
    features, cache_metadata, cache_ids = {}, {}, {}
    if args.experiment in ("all", "clip_bias"):
        common_model = None
        for name in ("real", *args.generators):
            values, metadata, cache_id = load_cache(args.clip_cache_root, name, records[name], common_model)
            if metadata["clip_identity"]["model_id"] != args.clip_model_id:
                raise ValueError("CLIP cache model_id differs from --clip-model-id")
            if common_model is None:
                common_model = metadata["clip_identity"]
            elif metadata["feature_dim"] != cache_metadata["real"]["feature_dim"]:
                raise ValueError("Real/Fake CLIP cache feature dimensions differ")
            features[name], cache_metadata[name], cache_ids[name] = values, metadata, cache_id
    checkpoint_info = {}
    for generator in args.generators:
        print("Preflight dual checkpoint:", generator, flush=True)
        model, info = load_model(*paths[generator], generator, torch.device("cpu"))
        del model
        checkpoint_info[generator] = info
    protocol = {"seeds": args.seeds, "shot": 10, "tau": .2, "tau_r": .1, "alpha_clip": [.25, .75], "all_queries": True,
                "generators": args.generators, "experiment": args.experiment, "noise_rates": args.noise_rates,
                "use_fp16": args.use_fp16, "effective_precision": "fp16_autocast" if args.use_fp16 and device.type == "cuda" else "fp32",
                "batch_size": args.batch_size, "num_workers": args.num_workers}
    run_identity = {**code, "protocol": protocol, "environment": env, "data_root": str(args.data_root),
                    "dataset_sha256": {name: digest(value) for name, value in records.items()},
                    "checkpoints": checkpoint_info, "cache_sha256": cache_ids, "baseline": baseline}
    config_path = root / "run_config.json"
    if config_path.exists() and read_json(config_path)["identity"] != run_identity:
        raise ValueError("Run identity changed (code/data/model/cache/protocol/environment); use NEW OUTPUT_DIR")
    if not config_path.exists():
        write_json(config_path, {"identity": run_identity, "command": sys.argv, "clip_cache_metadata": cache_metadata,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    for name, manifest in records.items():
        write_json(root / "dataset_manifests" / (name + ".json.gz"), manifest)
    if baseline_rows:
        write_csv(root / "baseline_reference_summary.csv", baseline_rows)
    if args.validate_only:
        write_json(root / "validation.json", {"status": "VALIDATION_ONLY_PASS", "identity": run_identity, "forward_run": False})
        print("VALIDATION_ONLY_PASS; no DDFSD forward", flush=True)
        return 0
    jobs = {}
    for generator in args.generators:
        for seed in args.seeds:
            conditions = [("label_noise", condition_name(p), p) for p in args.noise_rates] if args.experiment in ("all", "label_noise") else []
            if args.experiment in ("all", "clip_bias"):
                conditions.append(("clip_bias", "clip_concentrated", None))
            for experiment, condition, rate in conditions:
                key = f"{generator}/{experiment}/seed{seed}_{condition}"
                jobs[key] = (generator, seed, experiment, condition, rate)
    completed = {}
    write_json(root / "run_status.json", {"status": f"PARTIAL_0_OF_{len(jobs)}", "completed": 0, "expected": len(jobs)})
    for generator in args.generators:
        model = None
        for key, (name, seed, experiment, condition, rate) in jobs.items():
            if name != generator:
                continue
            selection = (label_noise_selection(len(datasets["real"]), len(datasets[generator]), seed, generator, rate)
                         if experiment == "label_noise" else clip_selection(features["real"], features[generator], seed, generator))
            identity = {**code, "generator": generator, "seed": seed, "experiment": experiment, "condition": condition,
                        "checkpoint": checkpoint_info[generator], "protocol": protocol, "environment": env,
                        "dataset_sha256": {name: run_identity["dataset_sha256"][name] for name in ("real", generator)},
                        "cache_sha256": digest({name: cache_ids[name] for name in ("real", generator)}) if experiment == "clip_bias" else "",
                        "selection_sha256": digest(selection)}
            folder = root / "per_task" / key
            if (folder / "DONE.json").exists():
                old_marker = read_json(folder / "DONE.json")
                if old_marker.get("identity") != identity:
                    raise ValueError("Saved task identity changed; use NEW OUTPUT_DIR")
                completed[key] = load_case(folder, identity, selection, (len(datasets["real"]), len(datasets[generator])))
                print("Resume verified:", key, flush=True)
                continue
            if model is None:
                set_seed(42)
                model, info = load_model(*paths[generator], generator, device)
                if info != checkpoint_info[generator]:
                    raise ValueError("Checkpoint/stats changed after preflight")
            print(f"Running {key}: {len(selection['query_indices'][0])}+{len(selection['query_indices'][1])} ALL queries", flush=True)
            metrics = _evaluate_binary_few_shot_impl(
                model=model, data_root=str(args.data_root), fake_class=generator, support_shot=10, seed=seed,
                batch_size=args.batch_size, num_workers=args.num_workers, device=device, use_fp16=args.use_fp16,
                tau=.2, tau_r=.1, max_query_per_class=0, model_mode="dual", branch_mode="dual", strict_images=True,
                selection=selection, datasets=(datasets["real"], datasets[generator]), return_alpha=True)
            relative_paths = [[entry["relative_path"] for entry in records[name]] for name in ("real", generator)]
            result = {"identity": identity, "generator": generator, "seed": seed, "experiment": experiment, "condition": condition,
                      "selection": selection, "metrics": metrics,
                      "support_manifest": support_rows(generator, seed, experiment, condition, selection, relative_paths)}
            write_json(folder / "result.json.gz", result)
            marker = {"status": "DONE", "identity": identity, "sha256": file_sha256(folder / "result.json.gz")}
            completed[key] = load_case(folder, identity, selection, (len(datasets["real"]), len(datasets[generator])), marker)
            write_json(folder / "DONE.json", marker)
            summarize_run(args, jobs, completed, baseline)
        if model is not None:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summarize_run(args, jobs, completed, baseline)
    return 0


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / ".r24.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run lock exists: {lock}; confirm previous process ended before removing")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        return execute(args)
    except (Exception, KeyboardInterrupt):
        error = traceback.format_exc()
        write_json(args.output_dir / "r24" / "logs" / "last_failure.json", {"traceback": error, "command": sys.argv})
        status_path = args.output_dir / "r24" / "run_status.json"
        prior = read_json(status_path) if status_path.exists() else {"completed": 0}
        prior.update(status="FAILED_OR_INTERRUPTED_SEE_LAST_FAILURE", last_failure=error)
        write_json(status_path, prior)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
