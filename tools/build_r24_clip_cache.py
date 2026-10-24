"""Build once: real/val + all six fake/val CLIP image caches, offline by default."""

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

sys.dont_write_bytecode = True  # keep runtime bytecode out of source bookkeeping
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.ddfsd_r24 import (CLIP_ID, GENERATORS, SCHEMA, digest, file_sha256,
                            outside_repo, write_json)


class IdentityTransform:
    def __call__(self, image):
        return image


def clip_collate(batch, processor):
    # Independent official CLIP preprocessing; never reused as DDFSD input.
    return processor(images=[image for image, _ in batch], return_tensors="pt")["pixel_values"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--classes", default="all", help="all or comma-separated real,ADM,BigGAN,glide,Midjourney,SD,VQDM")
    parser.add_argument("--clip-model-id", default=CLIP_ID)
    parser.add_argument("--clip-model-path", help="Existing external directory; otherwise use existing Hugging Face cache")
    parser.add_argument("--clip-revision", default="main", help="Requested HF revision; resolved commit is recorded")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--resume", "--skip-existing", action="store_true", default=True,
                        help="Default: strictly validate and reuse matching completed classes")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("Positive batch size and nonnegative workers required")
    args.classes = list(("real",) + GENERATORS) if args.classes == "all" else args.classes.split(",")
    if not args.classes or len(set(args.classes)) != len(args.classes) or any(c not in ("real",) + GENERATORS for c in args.classes):
        parser.error("Invalid or duplicate class IDs (glide is lowercase)")
    try:
        args.cache_root = outside_repo(args.cache_root, ROOT)
        if args.clip_model_path:
            args.clip_model_path = outside_repo(args.clip_model_path, ROOT)
            if not args.clip_model_path.is_dir():
                parser.error("--clip-model-path must be an existing local model directory")
    except ValueError as error:
        parser.error(str(error))
    args.data_root = Path(args.data_root).expanduser().resolve()
    if args.cache_root == args.data_root or args.data_root in args.cache_root.parents or args.cache_root in args.data_root.parents:
        parser.error("Cache and data directories must be disjoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    lock = args.cache_root / ".r24.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise RuntimeError(f"Cache writer lock exists: {lock}; confirm prior process ended before removing")
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from datasets.ddfsd_datasets import load_ddfsd_class_dataset
        from torch.utils.data import DataLoader
        from util.ddfsd_r24_runtime import audited_datasets, code_identity, environment, load_cache, save_cache
        code = code_identity(ROOT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
        # Resolve a predownloaded immutable snapshot; do not contact the Hub.
        try:
            location = args.clip_model_path or Path(snapshot_download(args.clip_model_id, revision=args.clip_revision, local_files_only=True))
            processor = CLIPImageProcessor.from_pretrained(str(location), local_files_only=True)
            model, loading = CLIPVisionModelWithProjection.from_pretrained(
                str(location), local_files_only=True, attn_implementation="eager",
                torch_dtype=torch.float32, output_loading_info=True)
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError("CLIP vision/projection weights did not load completely: " + repr(loading))
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError("Cannot load pretrained CLIP offline. Prepare weights first or use --clip-model-path; "
                               "inspect the chained error for missing files, weight format or torch/transformers compatibility. "
                               "See docs/R2-4_support_quality_experiments.md") from error
        location = Path(location).resolve()
        model_files = [p for p in location.rglob("*") if p.is_file() and (p.suffix in {".json", ".bin", ".safetensors"})
                       and ".cache" not in p.relative_to(location).parts]
        if not any(p.suffix in {".bin", ".safetensors"} for p in model_files):
            raise ValueError("Local CLIP directory contains no model weights")
        model_identity = {"model_id": args.clip_model_id, "requested_revision": args.clip_revision,
                          "resolved_revision": getattr(model.config, "_commit_hash", None) or (location.name if not args.clip_model_path else "local-content-sha256"),
                          "file_sha256": {p.relative_to(location).as_posix(): file_sha256(p) for p in sorted(model_files)},
                          "processor": processor.to_dict(), "encoder": "CLIPVisionModelWithProjection.image_embeds",
                          "precision": "float32", "attention": "eager", **code,
                          "environment": environment(), "device": str(device),
                          "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                          "batch_size": args.batch_size}
        model.eval().requires_grad_(False)
        model = model.to(device)
        datasets, records = audited_datasets(args.data_root, args.classes, args.cache_root)
        for name in args.classes:
            folder = args.cache_root / name
            if (folder / "DONE.json").exists():
                load_cache(args.cache_root, name, records[name], model_identity)
                print(f"Cache {name}: validated existing features; skipped encoding", flush=True)
                continue
            dataset = load_ddfsd_class_dataset(str(args.data_root), name, "val", transform=IdentityTransform(), strict_images=True)
            if dataset.paths != datasets[name].paths:
                raise ValueError("CLIP raw-image paths differ from official DDFSD path order")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=device.type == "cuda", collate_fn=partial(clip_collate, processor=processor))
            features = np.empty((len(dataset), int(model.config.projection_dim)), dtype=np.float32)
            offset = 0
            with torch.inference_mode():
                for pixels in loader:
                    embedding = model(pixel_values=pixels.to(device=device, dtype=torch.float32)).image_embeds.float()
                    norms = embedding.norm(dim=-1, keepdim=True)
                    if not torch.isfinite(embedding).all() or (norms <= 0).any():
                        raise ValueError("CLIP returned non-finite or zero embeddings")
                    array = (embedding / norms).cpu().numpy()
                    features[offset:offset + len(array)] = array
                    offset += len(array)
                    if offset % (args.batch_size * 50) == 0:
                        print(f"CLIP {name}: {offset}/{len(dataset)}", flush=True)
            if offset != len(dataset):
                raise ValueError("Incomplete CLIP feature traversal")
            metadata = {"schema": SCHEMA, "class_name": name, "n_images": len(dataset), "feature_dim": features.shape[1],
                        "inventory_sha256": digest(records[name]), "clip_identity": model_identity,
                        "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv}
            save_cache(folder, features, metadata, records[name])
            print(f"Cache {name}: DONE ({offset} images)", flush=True)
        write_json(args.cache_root / "cache_status.json", {"status": "COMPLETE_REQUESTED_CLASSES", "classes": args.classes,
                   "clip_identity": model_identity, "data_root": str(args.data_root)})
        return 0
    except Exception:
        import traceback
        write_json(args.cache_root / "last_failure.json", {"command": sys.argv, "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
