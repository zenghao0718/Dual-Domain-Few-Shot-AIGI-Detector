"""One command: six final dual checkpoints x five seeds, ALL validation queries.

Run on AutoDL's existing environment only. Never downloads, trains, or uploads.
The --help and --analyze-only paths do not import the deep-learning runtime.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.r21_common import (  # noqa: E402
    GENERATORS, METRICS, MODES, SCHEMA, SEEDS, atomic_bytes, atomic_json,
    canonical_hash, class_rows, read_json, sha256_file, validate_marker,
    validate_split, write_csv,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--ckpt-root", default=os.environ.get("CKPT_ROOT"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"), required=not os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--checkpoint-template", default="exclude_{generator}/ddfsd_step[15000].pth",
                        help="Exact path relative to CKPT_ROOT; no glob/recursive checkpoint selection. For training layout use exclude_{generator}/ckpt/ddfsd_step[15000].pth")
    parser.add_argument("--stats-template", default="exclude_{generator}/freq_stats.pt")
    parser.add_argument("--checksum-manifest", help="Optional JSON: checkpoints.{generator}.{ckpt_sha256,stats_sha256}, datasets.{class_name}: inventory hash")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--support-shot", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tau-r", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-fp16", choices=("true", "false"), default="true")
    parser.add_argument("--all-queries", action="store_true", default=True,
                        help="Always enabled; no query truncation option exists")
    parser.add_argument("--resume", action="store_true", default=True, help="Always verify and reuse only identical valid tasks")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Full resource audit/load checks; no model forward or experiments")
    modes.add_argument("--analyze-only", action="store_true", help="Recompute saved results, parity and report; no resource scan or inference")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2101)
    parser.add_argument("--package", action="store_true", help="Write compact review archive alongside OUTPUT_DIR")
    args = parser.parse_args(argv)
    try:
        args.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2 ** 32 for s in args.seeds):
        parser.error("Need unique seeds in [0, 2**32)")
    if (args.support_shot <= 0 or args.batch_size <= 0 or args.num_workers < 0
            or not 0 < args.tau < float("inf") or not 0 < args.tau_r < float("inf") or args.bootstrap_repeats <= 0):
        parser.error("Positive finite temperatures/shot/batch/bootstrap and nonnegative workers required")
    if not args.analyze_only and (not args.data_root or not args.ckpt_root):
        parser.error("Set --data-root / DATA_ROOT and --ckpt-root / CKPT_ROOT to actual AutoDL directories")
    args.use_fp16 = args.use_fp16 == "true"
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.output_dir == ROOT or ROOT in args.output_dir.parents:
        parser.error("OUTPUT_DIR must be outside the repository")
    if not args.analyze_only:
        args.data_root = Path(args.data_root).expanduser().resolve()
        args.ckpt_root = Path(args.ckpt_root).expanduser().resolve()
        for resource in (args.data_root, args.ckpt_root):
            if resource == args.output_dir or resource in args.output_dir.parents or args.output_dir in resource.parents:
                parser.error("OUTPUT_DIR and DATA_ROOT/CKPT_ROOT must be disjoint directories")
    return args


def load_runtime():
    global np, torch, DDFSDDualDomainNet, load_ddfsd_class_dataset, make_subset_loader
    global compute_alpha, compute_prototypes, compute_query_logits, compute_support_sigmas
    global binary_metrics, encode_batch, load_frequency_stats, set_seed
    global audit_image_paths, load_checkpoint, checkpoint_model_mode
    from test_ddfsd import audit_image_paths, checkpoint_model_mode, load_checkpoint
    import numpy as np
    import torch
    from datasets.ddfsd_datasets import load_ddfsd_class_dataset, make_subset_loader
    from model.ddfsd import DDFSDDualDomainNet
    from model.ddfsd_losses import compute_alpha, compute_prototypes, compute_query_logits, compute_support_sigmas
    from util.ddfsd_eval import binary_metrics, encode_batch
    from util.ddfsd_frequency import load_frequency_stats
    from util.utils import set_seed


def git_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Tracked repository files differ from HEAD; pull a clean committed version before running")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    hashes = {p: sha256_file(ROOT / p) for p in paths if p and (ROOT / p).is_file()}
    return {"git_commit": commit, "source_sha256": canonical_hash(hashes)}


def runtime_environment(device):
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "scikit-learn", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    try:
        nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
                                         text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        nvidia = "UNAVAILABLE"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages, "nvidia_smi": nvidia,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")}


def task_paths(args, generator):
    task_root = (args.ckpt_root / ("exclude_" + generator)).resolve()
    paths = []
    for template in (args.checkpoint_template, args.stats_template):
        relative = Path(template.format(generator=generator))
        if relative.is_absolute():
            raise ValueError("Templates must be relative to CKPT_ROOT")
        path = (args.ckpt_root / relative).resolve()
        if task_root not in path.parents:
            raise ValueError(f"Checkpoint/stats must belong to the same exclude_{generator} task directory: {path}")
        paths.append(path)
    return paths


def checkpoint_config(checkpoint):
    config = checkpoint.get("config", checkpoint.get("args", {}))
    return config if isinstance(config, dict) else vars(config)


def checked_model(checkpoint_path, stats_path, generator):
    checkpoint = load_checkpoint(str(checkpoint_path))
    config = checkpoint_config(checkpoint)
    if int(checkpoint.get("step", -1)) != 15000:
        raise ValueError("Checkpoint metadata step must be exactly 15000 (filename is insufficient)")
    recorded_mode = checkpoint.get("model_mode", config.get("model_mode"))
    if recorded_mode != "dual" or checkpoint_model_mode(checkpoint) != "dual":
        raise ValueError("Checkpoint metadata must explicitly declare model_mode=dual")
    excluded = checkpoint.get("exclude_class", config.get("exclude_class"))
    if excluded != generator:
        raise ValueError(f"Checkpoint exclude_class={excluded!r}, expected {generator}")
    for name, expected in (("exclude_class", generator), ("model_mode", "dual")):
        if name in config and config[name] != expected:
            raise ValueError(f"Conflicting checkpoint config {name}={config[name]!r}")
    stats = load_frequency_stats(str(stats_path))
    if not all(torch.isfinite(stats[name]).all().item() for name in ("mean", "std")) or (stats["std"] < 0).any().item():
        raise ValueError("Stats must be finite, std nonnegative, shape exactly [3]")
    model = DDFSDDualDomainNet(pretrained=False, model_mode="dual")
    model.load_state_dict(checkpoint["model"], strict=True)
    # Same order as test_ddfsd.py: checkpoint buffers first, selected stats last.
    model.set_freq_stats(stats["mean"], stats["std"])
    model.eval()
    model.requires_grad_(False)
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError("Non-finite checkpoint tensor: " + name)
    return model, {"step": 15000, "model_mode": "dual", "exclude_class": excluded,
                   "recorded_freq_stats_path": str(checkpoint.get("freq_stats_path", config.get("freq_stats_path", ""))),
                   "stats_mean": stats["mean"].tolist(), "stats_std": stats["std"].tolist()}


def preflight(args, device):
    """Audit every dataset and every checkpoint before ANY formal task forward."""
    print("Preflight: strict-decode + content-hash all seven val classes, then load all six checkpoint/stats pairs", flush=True)
    code = git_identity()
    errors, invalid, datasets, inventories, checkpoints = [], [], {}, {}, {}
    optional = read_json(args.checksum_manifest) if args.checksum_manifest else None
    for name in ("real",) + GENERATORS:
        try:
            dataset = load_ddfsd_class_dataset(str(args.data_root), name, "val", strict_images=True)
            if len(dataset) <= args.support_shot:
                raise ValueError(f"Need > {args.support_shot} images; found {len(dataset)}")
            invalid.extend(audit_image_paths(dataset.paths, name, "val"))
            inventory = [{"index": i, "relative_path": Path(p).relative_to(args.data_root).as_posix(),
                          "bytes": Path(p).stat().st_size, "sha256": sha256_file(p)} for i, p in enumerate(dataset.paths)]
            digest = canonical_hash(inventory)
            if optional is not None and optional["datasets"][name] != digest:
                raise ValueError("Provided dataset inventory checksum differs")
            datasets[name] = dataset
            inventories[name] = {"sha256": digest, "records": inventory, "n_images": len(dataset)}
            print(f"  {name}/val: {len(dataset)} images, inventory SHA256={digest}", flush=True)
        except Exception as error:
            errors.append(f"{name}/val: {type(error).__name__}: {error}")
    audit_columns = ["data_class", "split", "dataset_index", "filepath", "error_type", "error"]
    write_csv(args.output_dir / "logs" / "formal_val_invalid_images.csv", invalid, audit_columns)
    if invalid:
        errors.append(f"{len(invalid)} invalid images; see logs/formal_val_invalid_images.csv. No substitutions allowed.")
    for generator in GENERATORS:
        try:
            ckpt_path, stats_path = task_paths(args, generator)
            missing = [str(p) for p in (ckpt_path, stats_path) if not p.is_file()]
            if missing:
                raise FileNotFoundError("Missing: " + ", ".join(missing)
                                        + "; choose CKPT_ROOT containing all exclude_* directories; if ckpt/ is present, set --checkpoint-template")
            hashes = {"ckpt_sha256": sha256_file(ckpt_path), "stats_sha256": sha256_file(stats_path)}
            if optional is not None and any(optional["checkpoints"][generator][k] != v for k, v in hashes.items()):
                raise ValueError("Provided checkpoint/stats checksum differs")
            model, metadata = checked_model(ckpt_path, stats_path, generator)
            del model
            checkpoints[generator] = {**hashes, "ckpt_path": str(ckpt_path), "stats_path": str(stats_path), **metadata}
            print(f"  exclude_{generator}: step=15000 dual, strict load PASS", flush=True)
        except Exception as error:
            errors.append(f"exclude_{generator}: {type(error).__name__}: {error}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    except Exception as error:
        errors.append(f"Plot dependency: {error}; install matplotlib in existing AutoDL environment")
    preflight_result = {"status": "PASS" if not errors else "FAIL", "errors": errors,
                        "invalid_images": len(invalid), "checkpoints": checkpoints,
                        "dataset_hashes": {name: value["sha256"] for name, value in inventories.items()}}
    atomic_json(args.output_dir / "logs" / "preflight_validation.json", preflight_result)
    if errors:
        raise RuntimeError("Preflight FAILED; no experiment was run:\n" + "\n".join(errors))
    protocol = {"generators": list(GENERATORS), "seeds": args.seeds, "support_shot": args.support_shot,
                "tau": args.tau, "tau_r": args.tau_r, "alpha_clip": [0.25, 0.75], "model_mode": "dual",
                "all_queries": True, "max_query_per_class": 0, "split": "val", "strict_images": True,
                "preprocessing": "official Resize(256)/CenterCrop(224)/ToTensor; RGB ImageNet + signed Y Haar, selected task stats",
                "use_fp16": args.use_fp16, "effective_precision": "cuda_autocast_fp16" if args.use_fp16 and device.type == "cuda" else "fp32",
                "batch_size": args.batch_size, "num_workers": args.num_workers, "initialization_seed": 42}
    environment = runtime_environment(device)
    tasks = {}
    for generator in GENERATORS:
        ckpt = checkpoints[generator]
        for seed in args.seeds:
            key = f"{generator}_seed{seed}"
            tasks[key] = {"schema": SCHEMA, "generator": generator, "seed": seed, **code,
                          "ckpt_sha256": ckpt["ckpt_sha256"], "stats_sha256": ckpt["stats_sha256"],
                          "ckpt_path": ckpt["ckpt_path"], "stats_path": ckpt["stats_path"],
                          "data_root": str(args.data_root),
                          "dataset_sha256": {name: inventories[name]["sha256"] for name in ("real", generator)},
                          "protocol": protocol, "environment": environment}
    reproduction = ["python", "tools/run_r21_reliability.py", "--data-root", str(args.data_root), "--ckpt-root", str(args.ckpt_root),
                    "--output-dir", str(args.output_dir), "--checkpoint-template", args.checkpoint_template,
                    "--stats-template", args.stats_template, "--seeds", ",".join(map(str, args.seeds)), "--support-shot", str(args.support_shot),
                    "--tau", str(args.tau), "--tau-r", str(args.tau_r), "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers), "--use-fp16", str(args.use_fp16).lower(), "--all-queries", "--resume"]
    reproduction += ["--bootstrap-repeats", str(args.bootstrap_repeats), "--bootstrap-seed", str(args.bootstrap_seed)]
    if args.checksum_manifest:
        reproduction += ["--checksum-manifest", str(Path(args.checksum_manifest).resolve())]
    if args.package:
        reproduction.append("--package")
    manifest = {"schema": SCHEMA, **code, "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "environment": environment, "tasks": tasks,
                "checkpoints": checkpoints, "dataset_inventories": {name: {k: v for k, v in entry.items() if k != "records"}
                                                                         for name, entry in inventories.items()},
                "reproduction_command": shlex.join(reproduction), "command": sys.argv,
                "checksum_manifest_sha256": sha256_file(args.checksum_manifest) if args.checksum_manifest else None}
    existing_path = args.output_dir / "run_manifest.json"
    if existing_path.exists():
        existing = read_json(existing_path)
        if existing["tasks"] != tasks:
            raise RuntimeError("Run identity changed (code/checkpoint/stats/data/protocol/environment/path). Use a new OUTPUT_DIR; existing outputs preserved.")
        manifest = existing
    else:
        atomic_json(existing_path, manifest)
    for name, entry in inventories.items():
        atomic_json(args.output_dir / "manifests" / (name + ".json.gz"), entry["records"])
    return manifest, datasets


def atomic_npz(path, **arrays):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_task(args, model, datasets, key, identity, device):
    from util.ddfsd_sampling import sample_support_query_indices
    task_dir = args.output_dir / "per_task" / key
    task_dir.mkdir(parents=True, exist_ok=True)
    marker_path = task_dir / "DONE.json"
    if marker_path.exists():
        marker_path.unlink()  # only this pending, identity-matching task's obsolete marker
    sampling = {"classes": []}
    support_images = []
    labels, source_indices, relative_paths = [], [], []
    for cls, name in enumerate(("real", identity["generator"])):
        dataset = datasets[name]
        support, query = sample_support_query_indices(len(dataset), args.support_shot, identity["seed"])
        validate_split(support, query, len(dataset), args.support_shot, identity["seed"])
        entry = {"class_name": name, "dataset_size": len(dataset), "support_indices": support, "query_indices": query,
                 "support_paths": [Path(dataset.paths[i]).relative_to(args.data_root).as_posix() for i in support],
                 "query_paths": [Path(dataset.paths[i]).relative_to(args.data_root).as_posix() for i in query]}
        sampling["classes"].append(entry)
        support_images.append(torch.stack([dataset[i][0] for i in support]))
        labels.extend([cls] * len(query))
        source_indices.extend(query)
        relative_paths.extend(entry["query_paths"])
    collected = {prefix + mode: [] for mode in MODES for prefix in ("d_", "score_", "pred_")}
    with torch.no_grad():
        rgb, freq = encode_batch(model, torch.cat(support_images, dim=0), device, args.use_fp16)
        support_rgb = rgb.reshape(2, args.support_shot, -1).permute(1, 0, 2).unsqueeze(0)
        support_freq = freq.reshape(2, args.support_shot, -1).permute(1, 0, 2).unsqueeze(0)
        proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode="dual")
        sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=args.tau_r)
        raw = torch.sigmoid((sigma_freq - sigma_rgb) / args.tau_r)
        support_data = {"identity_sha256": canonical_hash(identity), "alpha_rgb": alpha[0].cpu().tolist(),
                        "alpha_raw": raw[0].cpu().tolist(), "sigma_rgb": sigma_rgb[0].cpu().tolist(),
                        "sigma_freq": sigma_freq[0].cpu().tolist()}
        for mode, proto in (("rgb", proto_rgb), ("freq", proto_freq)):
            support_data["proto_sep_" + mode] = float(torch.sqrt((proto[0, 0] - proto[0, 1]).pow(2).sum().clamp_min(1e-12)).cpu())
        atomic_npz(task_dir / "support.npz", support_rgb=support_rgb[0].cpu().numpy(), support_freq=support_freq[0].cpu().numpy(),
                   proto_rgb=proto_rgb[0].cpu().numpy(), proto_freq=proto_freq[0].cpu().numpy())
        processed = 0
        for entry in sampling["classes"]:
            loader = make_subset_loader(datasets[entry["class_name"]], entry["query_indices"], batch_size=args.batch_size,
                                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
            for images, _ in loader:
                rgb, freq = encode_batch(model, images, device, args.use_fp16)
                result = compute_query_logits(rgb.unsqueeze(0), freq.unsqueeze(0), proto_rgb, proto_freq, alpha,
                                              tau=args.tau, branch_mode="dual", model_mode="dual")
                distances = {"rgb": result["rgb_dist"].squeeze(0), "freq": result["freq_dist"].squeeze(0),
                             "dual": result["fused_dist"].squeeze(0)}
                reconstructed = alpha[:, None, :] * result["rgb_dist"] + (1 - alpha[:, None, :]) * result["freq_dist"]
                if not torch.equal(-reconstructed / args.tau, result["logits"]):
                    raise RuntimeError("Adaptive Dual reconstructed logits differ from official compute_query_logits")
                for mode, distance in distances.items():
                    logits = -distance / args.tau
                    prob = logits.softmax(dim=-1)
                    if mode == "dual" and not torch.equal(prob, result["logits"].squeeze(0).softmax(dim=-1)):
                        raise RuntimeError("Adaptive Dual softmax parity failed")
                    collected["d_" + mode].append(distance.cpu().numpy())
                    collected["score_" + mode].append(prob[:, 1].cpu().numpy())
                    collected["pred_" + mode].append(prob.argmax(dim=-1).cpu().numpy())
                processed += images.shape[0]
                if processed % (args.batch_size * 50) == 0:
                    print(f"  {key}: {processed}/{len(labels)} queries", flush=True)
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in collected.items()}
    arrays.update(labels=np.asarray(labels, dtype=np.int64), source_index=np.asarray(source_indices, dtype=np.int64),
                  relative_path=np.asarray(relative_paths), generator=np.asarray(identity["generator"]),
                  seed=np.asarray(identity["seed"]), identity_sha256=np.asarray(canonical_hash(identity)))
    metrics = {mode: binary_metrics(labels, arrays["score_" + mode].tolist(), arrays["pred_" + mode].tolist()) for mode in MODES}
    atomic_json(task_dir / "sampling.json.gz", sampling)
    atomic_json(task_dir / "support.json", support_data)
    atomic_npz(args.output_dir / "per_query" / (key + ".npz"), **arrays)
    atomic_json(task_dir / "metrics.json", metrics)
    atomic_json(task_dir / "class_metrics.json", class_rows(identity, support_data, arrays))
    relative_files = [f"per_query/{key}.npz"] + [f"per_task/{key}/{name}" for name in
                     ("support.npz", "support.json", "sampling.json.gz", "metrics.json", "class_metrics.json")]
    marker = {"status": "DONE", "identity": identity, "identity_sha256": canonical_hash(identity),
              "query_count": len(labels), "processed_query_count": processed, "logit_parity": "PASS",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "files": {path: sha256_file(args.output_dir / path) for path in relative_files}}
    if processed != len(labels):
        raise RuntimeError("Not all queries were processed")
    from tools.analyze_r21_reliability import validate_task
    validate_task(args.output_dir, key, identity, marker=marker)
    # DONE is the last atomic write, only after full semantic recomputation.
    atomic_json(marker_path, marker)


def official_reference(args, generator, manifest):
    """One automatic official evaluation per generator; all five seeds in one subprocess."""
    identities = {k: i for k, i in manifest["tasks"].items() if i["generator"] == generator}
    identity = {"tasks": identities}
    directory = args.output_dir / "formal" / generator
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / "REFERENCE_DONE.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("identity") != identity:
            raise ValueError("Reference identity mismatch; use new OUTPUT_DIR")
        try:
            validate_marker(args.output_dir, marker, identity)
            print(f"Official {generator}: validated existing reference", flush=True)
            return
        except ValueError:
            marker_path.unlink()
    ckpt = manifest["checkpoints"][generator]
    command = [sys.executable, str(ROOT / "test_ddfsd.py"), "--data_root", str(args.data_root),
               "--output_dir", str(directory), "--ckpt_path", ckpt["ckpt_path"], "--ckpt_step", "15000",
               "--freq_stats_path", ckpt["stats_path"], "--exclude_class", generator, "--model_mode", "dual", "--branch_mode", "dual",
               "--num_support_test", str(args.support_shot), "--eval_seeds", ",".join(map(str, args.seeds)),
               "--eval_repeats", str(len(args.seeds)), "--max_eval_query_per_class", "0", "--eval_batch_size", str(args.batch_size),
               "--num_workers", str(args.num_workers), "--tau", str(args.tau), "--tau_r", str(args.tau_r),
               "--use_fp16", str(args.use_fp16), "--pretrained", "False", "--seed", "42"]
    print(f"Official {generator}: test_ddfsd.py full-query parity for {len(args.seeds)} seeds; log={directory / 'reference.log'}", flush=True)
    with open(directory / "reference.log", "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Official test_ddfsd.py failed ({completed.returncode}); see {directory / 'reference.log'}; rerun same command after diagnosis")
    files = [f"formal/{generator}/{name}" for name in ("ddfsd_eval_per_seed.csv", "ddfsd_eval_summary.csv", "config.json", "reference.log")]
    atomic_json(marker_path, {"status": "DONE", "identity": identity, "identity_sha256": canonical_hash(identity),
                              "command": command, "files": {p: sha256_file(args.output_dir / p) for p in files}})


def update_progress(output, manifest, completed):
    expected = len(manifest["tasks"])
    atomic_json(output / "run_status.json", {"status": f"PARTIAL_{len(completed)}_OF_{expected}",
                "completed_tasks": len(completed), "expected_tasks": expected, "validated_tasks": completed,
                "class_rows": 2 * len(completed), "official_parity": "PENDING_UNTIL_ANALYSIS"})


def execute(args):
    from tools.analyze_r21_reliability import analyze, validate_task
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        result = analyze(args.output_dir, args.bootstrap_repeats, args.bootstrap_seed, args.package)
        return 0 if "COMPLETE_AND_MATCHING" in result["status"] else 2
    load_runtime()
    set_seed(42)  # exactly as formal entrypoint; official sampler has its own RNG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, datasets = preflight(args, device)
    if args.validate_only:
        print("VALIDATION_ONLY_PASS; no model inference; GPU_VALIDATION_PENDING_AUTODL until full run", flush=True)
        return 0
    completed = []
    update_progress(args.output_dir, manifest, completed)
    # Check all pre-existing markers before overwriting any task artifact.
    for key, identity in manifest["tasks"].items():
        path = args.output_dir / "per_task" / key / "DONE.json"
        if path.exists() and read_json(path).get("identity") != identity:
            raise ValueError("Existing task identity differs: " + key + "; use new OUTPUT_DIR")
    for generator in GENERATORS:
        pending = []
        for key, identity in manifest["tasks"].items():
            if identity["generator"] != generator:
                continue
            try:
                validate_task(args.output_dir, key, identity)
                completed.append(key)
                print(f"Resume {key}: identity/checksum/full-data recomputation PASS", flush=True)
            except (OSError, ValueError, KeyError) as error:
                print(f"Pending {key}: {error}", flush=True)
                pending.append((key, identity))
        update_progress(args.output_dir, manifest, completed)
        if pending:
            ckpt = manifest["checkpoints"][generator]
            if sha256_file(ckpt["ckpt_path"]) != ckpt["ckpt_sha256"] or sha256_file(ckpt["stats_path"]) != ckpt["stats_sha256"]:
                raise ValueError("Checkpoint/stats changed after preflight")
            set_seed(42)
            model, _ = checked_model(ckpt["ckpt_path"], ckpt["stats_path"], generator)
            model = model.to(device).eval()
            for key, identity in pending:
                print(f"Running {key}: full-query dual encoding ({len(completed)}/{len(manifest['tasks'])} completed)", flush=True)
                run_task(args, model, datasets, key, identity, device)
                completed.append(key)
                update_progress(args.output_dir, manifest, completed)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        official_reference(args, generator, manifest)
    result = analyze(args.output_dir, args.bootstrap_repeats, args.bootstrap_seed, args.package)
    if not result["parity_matching"]:
        raise RuntimeError("Official parity missing/mismatched; inspect parity_validation.csv and reference logs. Do not claim completion.")
    return 0


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two invocations from mixing writes. A crashed
    # process leaves a lock deliberately: check PID before removing it manually.
    lock = args.output_dir / ".r21.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Run lock exists: {lock}; check recorded PID; remove only after confirming no active writer", file=sys.stderr)
        return 2
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": platform.node(), "command": sys.argv}, handle)
        return execute(args)
    except (Exception, KeyboardInterrupt) as error:
        detail = traceback.format_exc()
        atomic_bytes(args.output_dir / "logs" / "last_traceback.txt", detail.encode("utf-8"))
        atomic_json(args.output_dir / "logs" / "last_failure.json", {"error_type": type(error).__name__, "error": str(error),
                    "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "recovery": "Inspect traceback/config; rerun identical command to verify and resume. For code fixes pull new commit and use a NEW output directory."})
        status_path = args.output_dir / "run_status.json"
        if not status_path.exists():
            atomic_json(status_path, {"status": f"PARTIAL_0_OF_{6 * len(args.seeds)}", "completed_tasks": 0,
                                      "expected_tasks": 6 * len(args.seeds), "error": str(error)})
        else:
            progress = read_json(status_path)
            count = progress.get("completed_tasks", 0)
            if "COMPLETE_AND_MATCHING" in progress.get("status", ""):
                count = 0  # a new attempt failed before revalidating the old completion
            progress.update(status=f"PARTIAL_{count}_OF_{6 * len(args.seeds)}", completed_tasks=count,
                            last_attempt_error=str(error), last_attempt_status="FAILED_OR_INTERRUPTED")
            atomic_json(status_path, progress)
        print(detail, file=sys.stderr)
        print(f"Results preserved. Traceback: {args.output_dir / 'logs' / 'last_traceback.txt'}", file=sys.stderr)
        return 2
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
