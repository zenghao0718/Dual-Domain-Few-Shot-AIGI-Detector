"""AutoDL-only runtime adapters. Pure selection lives in ddfsd_r24.py."""

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from util.ddfsd_r24 import (SCHEMA, checkpoint_metadata, digest, file_sha256,
                            inventory, read_json, validate_cache, write_json)


def code_identity(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if dirty:
        raise RuntimeError("Tracked checkout differs from HEAD; run committed code and use a new output/cache directory after updates")
    return {"git_commit": commit}


def environment():
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "scikit-learn", "transformers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def load_model(ckpt_path, stats_path, generator, device):
    import torch
    from model.ddfsd import DDFSDDualDomainNet
    from test_ddfsd import load_checkpoint
    from util.ddfsd_frequency import load_frequency_stats
    ckpt_path, stats_path = Path(ckpt_path), Path(stats_path)
    if not ckpt_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/freq_stats pair: {ckpt_path}, {stats_path}")
    if stats_path.parent != ckpt_path.parent and not (ckpt_path.parent.name == "ckpt" and stats_path.parent == ckpt_path.parent.parent):
        raise ValueError("freq_stats must belong to the checkpoint task: same directory, or sibling of ckpt/")
    checkpoint = load_checkpoint(str(ckpt_path))
    metadata = checkpoint_metadata(checkpoint, generator)
    stats = load_frequency_stats(str(stats_path))
    if not all(torch.isfinite(stats[key]).all().item() for key in ("mean", "std")) or (stats["std"] < 0).any().item():
        raise ValueError("Invalid frequency stats: finite [3] mean/std and nonnegative std required")
    model = DDFSDDualDomainNet(pretrained=False, model_mode="dual")
    expected = set(model.state_dict())
    for prefix in ("rgb_backbone.", "rgb_projector.", "freq_backbone.", "freq_projector."):
        required = {key for key in expected if key.startswith(prefix)}
        if not required or not required.issubset(checkpoint["model"]):
            raise ValueError("Incomplete dual checkpoint branch: " + prefix)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_freq_stats(stats["mean"], stats["std"])
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError("Non-finite model tensor: " + name)
    model.eval().requires_grad_(False)
    metadata.update(ckpt_path=str(ckpt_path), freq_stats_path=str(stats_path),
                    ckpt_sha256=file_sha256(ckpt_path), stats_sha256=file_sha256(stats_path), strict_load=True)
    return model.to(device), metadata


def audited_datasets(data_root, classes, log_dir):
    from datasets.ddfsd_datasets import load_ddfsd_class_dataset
    from test_ddfsd import audit_image_paths
    datasets, records, errors = {}, {}, []
    for name in classes:
        print(f"Audit/hash {name}/val", flush=True)
        try:
            dataset = load_ddfsd_class_dataset(str(data_root), name, "val", strict_images=True)
            if len(dataset) <= 10:
                raise ValueError(f"Need at least 11 images; found {len(dataset)}")
            invalid = audit_image_paths(dataset.paths, name, "val")
            if invalid:
                errors.extend(invalid)
                continue
            datasets[name] = dataset
            records[name] = inventory(dataset.paths, data_root)
        except Exception as error:
            errors.append({"class": name, "error": str(error)})
    write_json(Path(log_dir) / "image_audit.json", {"errors": errors, "strict_images": True})
    if errors:
        raise ValueError(f"Strict val audit failed; no substitution permitted. See {log_dir}/image_audit.json")
    return datasets, records


def load_cache(cache_root, class_name, records, expected_clip=None):
    folder = Path(cache_root) / class_name
    marker_path = folder / "DONE.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing completed CLIP cache: {marker_path}; run tools/build_r24_clip_cache.py first")
    marker = read_json(marker_path)
    required = ("features.npy", "metadata.json", "manifest.json.gz")
    if marker.get("status") != "DONE" or set(marker.get("sha256", {})) != set(required):
        raise ValueError("Incomplete CLIP cache completion marker")
    for name in required:
        if not (folder / name).is_file() or file_sha256(folder / name) != marker["sha256"][name]:
            raise ValueError("Corrupt CLIP cache artifact: " + str(folder / name))
    metadata = read_json(folder / "metadata.json")
    manifest = read_json(folder / "manifest.json.gz")
    if manifest != records or metadata.get("class_name") != class_name:
        raise ValueError("CLIP cache image paths/order/content/class mismatch; rebuild with current official inventory")
    import numpy as np
    features = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
    validate_cache(metadata, records, features, expected_clip)
    return features, metadata, digest(marker)


def save_cache(folder, features, metadata, records):
    import numpy as np
    import tempfile
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    validate_cache(metadata, records, features, metadata["clip_identity"])
    descriptor, path = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, features, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(path, folder / "features.npy")
    finally:
        if os.path.exists(path):
            os.unlink(path)
    write_json(folder / "manifest.json.gz", records)
    write_json(folder / "metadata.json", metadata)
    write_json(folder / "DONE.json", {"status": "DONE", "schema": SCHEMA,
               "sha256": {name: file_sha256(folder / name) for name in ("features.npy", "metadata.json", "manifest.json.gz")}})
