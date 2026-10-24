"""Synthetic R2-1 tests only: no real images/checkpoints, training or inference.

Default Python runs standard-library tests. If numpy already exists, small
loss-function adapters and on-disk 30-task fixtures are also exercised.
"""

import ast
import copy
import csv
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r21_common as common
from tools import run_r21_reliability as runner
from tools import analyze_r21_reliability as analyzer
from util.ddfsd_sampling import sample_support_query_indices

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class StandardLibraryTests(unittest.TestCase):
    def test_alpha_raw_clipping_and_symmetry(self):
        self.assertEqual(common.alpha_reference(.2, .2, .1), (.5, .5))
        self.assertEqual(common.alpha_reference(0, 1000, .1), (1, .75))
        self.assertEqual(common.alpha_reference(1000, 0, .1), (0, .25))
        raw, alpha = common.alpha_reference(.1, .15, .1)
        self.assertAlmostEqual(raw, 1 / (1 + math.exp(-.5)))
        self.assertEqual(alpha, raw)
        with self.assertRaises(ValueError):
            common.alpha_reference(0, 1, 0)

    def test_margin_label_signs(self):
        self.assertEqual(common.margin(.2, .8, 0), .8 - .2)
        self.assertEqual(common.margin(.2, .8, 1), .2 - .8)
        self.assertEqual(common.margin(1, 1, 0), 0)
        with self.assertRaises(ValueError):
            common.margin(1, 2, 2)

    def test_ties_are_excluded_and_coverage_is_explicit(self):
        result = common.direction([0, 1, -1, 2, 0], [1, 2, 1, 0, 0])
        self.assertEqual(result["n_non_tie"], 2)
        self.assertEqual(result["n_tie"], 3)
        self.assertEqual(result["n_match"], 1)
        self.assertEqual(result["coverage"], .4)
        self.assertEqual(result["agreement"], .5)
        self.assertTrue(math.isnan(common.direction([0], [1])["agreement"]))

    def test_spearman_average_ranks_constants_and_negative(self):
        self.assertEqual(common.ranks([1, 3, 3, 2]), [1, 3.5, 3.5, 2])
        self.assertAlmostEqual(common.spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertAlmostEqual(common.spearman([1, 1, 3], [2, 2, 4]), 1)
        self.assertTrue(math.isnan(common.spearman([1, 1], [1, 2])))
        self.assertTrue(math.isnan(common.spearman([], [])))
        with self.assertRaises(ValueError):
            common.spearman([1], [math.nan])

    def test_metrics_fake_positive_and_score_ties(self):
        labels = [0, 0, 1, 1]
        result = common.binary_metrics_reference(labels, [.1, .4, .35, .8], [0, 0, 0, 1])
        self.assertEqual(result["acc"], .75)
        self.assertEqual(result["auc"], .75)
        self.assertAlmostEqual(result["ap"], 5 / 6)
        tied = common.binary_metrics_reference(labels, [.5] * 4, [0] * 4)
        self.assertEqual(tied["ap"], .5)
        self.assertEqual(tied["auc"], .5)
        self.assertEqual(tied["real_acc"], 1)
        self.assertEqual(tied["fake_acc"], 0)
        reverse = common.binary_metrics_reference(labels, [.9, .8, .2, .1], [1, 1, 0, 0])
        self.assertEqual(reverse["auc"], 0)
        with self.assertRaises(ValueError):
            common.binary_metrics_reference([0], [.5], [0])

    def test_bootstrap_is_clustered_reproducible_and_constant_undefined(self):
        rows = [{"generator": g, "X": x, "Y": x} for g in common.GENERATORS for x in (-.1, .1)]
        result = common.cluster_bootstrap(rows, "Y", 100, 21)
        self.assertEqual(result, common.cluster_bootstrap(rows, "Y", 100, 21))
        self.assertEqual(result["n_clusters"], 6)
        self.assertEqual(result["rho_ci_low"], 1)
        constants = [{"generator": g, "X": 0, "Y": 1} for g in common.GENERATORS]
        result = common.cluster_bootstrap(constants, "Y", 10, 21)
        self.assertTrue(math.isnan(result["rho_ci_low"]))
        self.assertEqual(result["rho_bootstrap_valid"], 0)

    def test_official_sampling_full_partition_and_order(self):
        for seed in common.SEEDS:
            support, query = sample_support_query_indices(35, 10, seed)
            common.validate_split(support, query, 35, 10, seed)
            with self.assertRaises(ValueError):
                common.validate_split(support, query[:-1], 35, 10, seed)
            with self.assertRaises(ValueError):
                common.validate_split(support, list(reversed(query)), 35, 10, seed)

    def test_atomic_marker_hash_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="r21_synthetic_") as directory:
            root = Path(directory)
            common.atomic_json(root / "file.json", {"synthetic": True})
            common.atomic_json(root / "sampling.json.gz", {"support": [1]})
            self.assertEqual(common.read_json(root / "sampling.json.gz"), {"support": [1]})
            identity = {"seed": 42, "ckpt_sha256": "synthetic"}
            marker = {"status": "DONE", "identity": identity, "identity_sha256": common.canonical_hash(identity),
                      "files": {"file.json": common.sha256_file(root / "file.json")}}
            self.assertTrue(common.validate_marker(root, marker, identity))
            with self.assertRaises(ValueError):
                common.validate_marker(root, marker, {**identity, "seed": 101})
            common.atomic_json(root / "file.json", {"synthetic": False})
            with self.assertRaises(ValueError):
                common.validate_marker(root, marker, identity)
            with self.assertRaises(ValueError):
                common.validate_marker(root, {**marker, "status": "WRITING"}, identity)

    def test_parser_defaults_all_queries_and_overrides(self):
        with tempfile.TemporaryDirectory(prefix="r21_synthetic_") as directory:
            args = runner.parse_args(["--data-root", str(Path(directory) / "data"), "--ckpt-root", str(Path(directory) / "ckpt"),
                                      "--output-dir", str(Path(directory) / "out")])
            self.assertEqual(args.seeds, list(common.SEEDS))
            self.assertEqual(args.support_shot, 10)
            self.assertEqual((args.tau, args.tau_r), (.2, .1))
            self.assertTrue(args.all_queries and args.use_fp16 and args.resume)
            args = runner.parse_args(["--output-dir", str(Path(directory) / "out"), "--analyze-only", "--use-fp16", "false"])
            self.assertFalse(args.use_fp16)

    def test_exact_checkpoint_path_and_same_task_enforcement(self):
        with tempfile.TemporaryDirectory(prefix="r21_synthetic_") as directory:
            args = types.SimpleNamespace(ckpt_root=Path(directory), checkpoint_template="exclude_{generator}/ckpt/ddfsd_step[15000].pth",
                                         stats_template="exclude_{generator}/freq_stats.pt")
            ckpt, stats = runner.task_paths(args, "ADM")
            self.assertEqual(ckpt.name, "ddfsd_step[15000].pth")
            self.assertEqual(stats.parent, ckpt.parent.parent)
            args.stats_template = "exclude_SD/freq_stats.pt"
            with self.assertRaises(ValueError):
                runner.task_paths(args, "ADM")

    def test_checkpoint_rejects_wrong_step_mode_generator_and_conflicts(self):
        for change in ({"step": 14999}, {"model_mode": "rgb-only"}, {"exclude_class": "SD"},
                       {"model_mode": None}, {"config": {"model_mode": "freq-only"}}):
            checkpoint = {"model": {}, "step": 15000, "model_mode": "dual", "exclude_class": "ADM", **change}
            with self.subTest(change=change), mock.patch.object(runner, "load_checkpoint", return_value=checkpoint, create=True):
                with mock.patch.object(runner, "checkpoint_model_mode", return_value="dual", create=True):
                    with self.assertRaises(ValueError):
                        runner.checked_model("synthetic.pth", "synthetic.pt", "ADM")

    def test_help_needs_no_torch(self):
        for name in ("run_r21_reliability.py", "analyze_r21_reliability.py"):
            completed = subprocess.run([sys.executable, str(ROOT / "tools" / name), "--help"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_shell_passes_literal_templates_and_overrides(self):
        bash = shutil.which("bash")
        git = shutil.which("git")
        if not bash and git:
            candidate = Path(git).parent.parent / "bin" / "bash.exe"
            bash = str(candidate) if candidate.is_file() else None
        if not bash:
            self.skipTest("NOT_RUN_ENV_MISSING: bash")
        with tempfile.TemporaryDirectory(prefix="r21_synthetic_") as temporary:
            root = Path(temporary)
            executable, arguments = root / "mock_python.sh", root / "arguments.txt"
            executable.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$R21_TEST_ARGV_OUT"\n', encoding="utf-8")
            executable.chmod(0o755)
            environment = {**os.environ, "DATA_ROOT": "/synthetic/data path", "CKPT_ROOT": "/synthetic/ckpt path",
                           "OUTPUT_DIR": "/synthetic/output path", "PYTHON": executable.as_posix(),
                           "R21_TEST_ARGV_OUT": arguments.as_posix()}
            environment.pop("CHECKPOINT_TEMPLATE", None)
            environment.pop("STATS_TEMPLATE", None)
            for custom in (False, True):
                if custom:
                    environment["CHECKPOINT_TEMPLATE"] = "exclude_{generator}/ckpt/ddfsd_step[15000].pth"
                completed = subprocess.run([bash, (ROOT / "scripts" / "run_r21_reliability.sh").as_posix(), "--validate-only"],
                                           env=environment, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                argv = arguments.read_text(encoding="utf-8").splitlines()
                expected = "exclude_{generator}/" + ("ckpt/" if custom else "") + "ddfsd_step[15000].pth"
                self.assertEqual(argv[argv.index("--checkpoint-template") + 1], expected)
                self.assertEqual(argv[argv.index("--stats-template") + 1], "exclude_{generator}/freq_stats.pt")
                self.assertEqual(argv[argv.index("--data-root") + 1], "/synthetic/data path")
                self.assertEqual(argv[-1], "--validate-only")

    def test_preflight_checks_all_six_pairs_before_any_experiment(self):
        class SyntheticDataset(list):
            @property
            def paths(self):
                return self

        with tempfile.TemporaryDirectory(prefix="r21_synthetic_") as temporary:
            root = Path(temporary)
            args = runner.parse_args(["--data-root", str(root / "data"), "--ckpt-root", str(root / "ckpt"),
                                      "--output-dir", str(root / "output")])
            datasets = {}
            for name in ("real",) + common.GENERATORS:
                folder = args.data_root / name / "val"
                folder.mkdir(parents=True)
                paths = []
                for i in range(11):
                    path = folder / f"synthetic_{i}.png"
                    path.write_bytes(b"SYNTHETIC: image audit is mocked, no inference")
                    paths.append(str(path))
                datasets[name] = SyntheticDataset(paths)
            for generator in common.GENERATORS:
                for path in runner.task_paths(args, generator):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"SYNTHETIC: checkpoint load is mocked")
            matplotlib = types.ModuleType("matplotlib")
            matplotlib.__path__ = []
            matplotlib.use = lambda _: None
            modules = {"matplotlib": matplotlib, "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot")}
            with mock.patch.object(runner, "git_identity", return_value={"git_commit": "synthetic", "source_sha256": "synthetic"}), \
                 mock.patch.object(runner, "load_ddfsd_class_dataset", side_effect=lambda root, name, split, **kw: datasets[name], create=True), \
                 mock.patch.object(runner, "audit_image_paths", return_value=[], create=True), \
                 mock.patch.object(runner, "checked_model", return_value=(object(), {"step": 15000})) as load, \
                 mock.patch.object(runner, "runtime_environment", return_value={"synthetic": True}), \
                 mock.patch.dict(sys.modules, modules):
                manifest, _ = runner.preflight(args, types.SimpleNamespace(type="cpu"))
                self.assertEqual(len(manifest["tasks"]), 30)
                self.assertEqual(load.call_count, 6)
                for generator in common.GENERATORS:
                    runner.task_paths(args, generator)[0].unlink()
                load.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "Preflight FAILED"):
                    runner.preflight(args, types.SimpleNamespace(type="cpu"))
                validation = common.read_json(args.output_dir / "logs" / "preflight_validation.json")
                self.assertEqual(len(validation["errors"]), 6)
                self.assertEqual(load.call_count, 0)


def numpy_loss_adapter():
    """Execute unchanged loss-function AST with a tiny NumPy tensor adapter.

    This tests formulas/shape contracts without claiming a PyTorch/CUDA test.
    """
    import numpy as np

    class Tensor(np.ndarray):
        def __new__(cls, value):
            return np.asarray(value, dtype=np.float32).view(cls)

        def float(self):
            return Tensor(self)

        def pow(self, exponent):
            return Tensor(np.asarray(self) ** exponent)

        def sum(self, dim=None):
            return Tensor(np.asarray(self).sum(axis=dim))

        def mean(self, dim=None):
            return Tensor(np.asarray(self).mean(axis=dim))

        def clamp_min(self, lower):
            return Tensor(np.maximum(self, lower))

        def clamp(self, lower, upper):
            return Tensor(np.clip(self, lower, upper))

        def detach(self):
            return self.copy()

    def normalize(value, p, dim):
        return Tensor(value / np.maximum(np.linalg.norm(value, ord=p, axis=dim, keepdims=True), 1e-12))

    from typing import Dict, Optional
    names = {"_validate_model_mode", "ordinary_euclidean_distance", "compute_prototypes", "compute_support_sigmas",
             "compute_alpha", "fuse_distances", "compute_query_logits"}
    tree = ast.parse((ROOT / "model" / "ddfsd_losses.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": types.SimpleNamespace(Tensor=Tensor, sqrt=lambda x: Tensor(np.sqrt(x)),
                                                sigmoid=lambda x: Tensor(1 / (1 + np.exp(-np.asarray(x))))),
                 "F": types.SimpleNamespace(normalize=normalize), "Dict": Dict, "Optional": Optional,
                 "VALID_MODEL_MODES": {"dual", "rgb-only", "freq-only"}, "VALID_BRANCH_MODES": {"dual", "rgb-only", "freq-only"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "synthetic_numpy_loss_adapter", "exec"), namespace)
    namespace["Tensor"] = Tensor
    return namespace


@unittest.skipUnless(HAS_NUMPY, "NOT_RUN_ENV_MISSING: numpy (no installation required)")
class NumpySyntheticTests(unittest.TestCase):
    def test_actual_loss_functions_alpha_sigmas_distances_and_parity(self):
        import numpy as np
        loss = numpy_loss_adapter()
        tensor = loss["Tensor"]
        rgb = tensor([[[[1, 0], [0, 1]], [[.8, .6], [.6, .8]]]])
        freq = tensor([[[[0, 1], [1, 0]], [[.6, .8], [.8, .6]]]])
        proto_rgb, proto_freq = loss["compute_prototypes"](rgb, freq)
        sr, sf = loss["compute_support_sigmas"](rgb, freq, proto_rgb, proto_freq)
        np.testing.assert_allclose(sr, np.mean((np.asarray(rgb) - np.asarray(proto_rgb)[:, None]) ** 2, axis=(1, 3)) * 2, atol=1e-6)
        alpha = loss["compute_alpha"](sr, sf, .1)
        for i in (0, 1):
            self.assertAlmostEqual(float(alpha[0, i]), common.alpha_reference(float(sr[0, i]), float(sf[0, i]), .1)[1], places=6)
        np.testing.assert_allclose(loss["compute_alpha"](tensor([[0, 1]]), tensor([[1, 0]]), .1), [[.75, .25]])
        query = tensor([[[1, 0], [0, 1]]])
        result = loss["compute_query_logits"](query, query, proto_rgb, proto_freq, alpha)
        np.testing.assert_array_equal(result["logits"], -result["fused_dist"] / .2)
        np.testing.assert_allclose(result["fused_dist"], alpha[:, None] * result["rgb_dist"] + (1 - alpha[:, None]) * result["freq_dist"])
        for mode, field in (("rgb-only", "rgb_dist"), ("freq-only", "freq_dist")):
            branch = loss["compute_query_logits"](query, query, proto_rgb, proto_freq, alpha, branch_mode=mode)
            np.testing.assert_array_equal(branch["logits"], -result[field] / .2)

    def test_class_isolation_and_xy(self):
        import numpy as np
        identity = {"generator": "ADM", "seed": 42, "ckpt_sha256": "synthetic", "stats_sha256": "synthetic",
                    "protocol": {"support_shot": 10}}
        support = {"sigma_rgb": [.1, .3], "sigma_freq": [.2, .1], "alpha_raw": [.7, .2], "alpha_rgb": [.7, .25],
                   "proto_sep_rgb": .6, "proto_sep_freq": .4}
        arrays = {"labels": np.array([0, 0, 1]), "d_rgb": np.array([[0, 1], [0, 3], [5, 0]]),
                  "d_freq": np.array([[0, 2], [0, 4], [1, 0]]), **{"pred_" + mode: np.array([0, 0, 1]) for mode in common.MODES}}
        real, fake = common.class_rows(identity, support, arrays)
        self.assertEqual(real["mean_margin_rgb"], 2)
        self.assertEqual(real["Y"], -1)
        self.assertAlmostEqual(real["X"], .2)
        self.assertEqual(fake["mean_margin_rgb"], 5)
        self.assertEqual(fake["Y"], 4)
        self.assertEqual(fake["X"], -.25)
        self.assertEqual((real["n_query"], fake["n_query"]), (2, 1))

    def make_fixture(self, output):
        import numpy as np
        protocol = {"generators": list(common.GENERATORS), "seeds": list(common.SEEDS), "support_shot": 10,
                    "tau": .2, "tau_r": .1, "use_fp16": True, "all_queries": True, "max_query_per_class": 0}
        inventories = {}
        for name in ("real",) + common.GENERATORS:
            records = [{"index": i, "relative_path": f"{name}/val/synthetic_{i}.png", "bytes": 1, "sha256": "synthetic"} for i in range(14)]
            common.atomic_json(output / "manifests" / (name + ".json.gz"), records)
            inventories[name] = records
        tasks = {}
        for generator in common.GENERATORS:
            for seed in common.SEEDS:
                key = f"{generator}_seed{seed}"
                identity = {"generator": generator, "seed": seed, "ckpt_sha256": "synthetic_" + generator,
                            "stats_sha256": "synthetic_" + generator, "protocol": protocol,
                            "ckpt_path": f"/synthetic/exclude_{generator}/ddfsd_step[15000].pth",
                            "stats_path": f"/synthetic/exclude_{generator}/freq_stats.pt",
                            "dataset_sha256": {name: common.canonical_hash(inventories[name]) for name in ("real", generator)}}
                tasks[key] = identity
                task = output / "per_task" / key
                sampling = {"classes": []}
                labels, indices, paths = [], [], []
                for cls, name in enumerate(("real", generator)):
                    support, query = sample_support_query_indices(14, 10, seed)
                    entry = {"class_name": name, "dataset_size": 14, "support_indices": support, "query_indices": query,
                             "support_paths": [inventories[name][i]["relative_path"] for i in support],
                             "query_paths": [inventories[name][i]["relative_path"] for i in query]}
                    sampling["classes"].append(entry)
                    labels.extend([cls] * len(query))
                    indices.extend(query)
                    paths.extend(entry["query_paths"])
                z = np.tile(np.array([[[1, 0], [0, 1]]], dtype=np.float32), (10, 1, 1))
                proto = z.mean(axis=0)
                support_data = {"identity_sha256": common.canonical_hash(identity), "sigma_rgb": [0, 0], "sigma_freq": [0, 0],
                                "alpha_raw": [.5, .5], "alpha_rgb": [.5, .5], "proto_sep_rgb": math.sqrt(2), "proto_sep_freq": math.sqrt(2)}
                distances = np.array([[.1, .9]] * 4 + [[.9, .1]] * 4, dtype=np.float32)
                logits = -distances / .2
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                arrays = {"labels": np.asarray(labels), "source_index": np.asarray(indices), "relative_path": np.asarray(paths),
                          "generator": np.asarray(generator), "seed": np.asarray(seed), "identity_sha256": np.asarray(common.canonical_hash(identity))}
                for mode in common.MODES:
                    arrays.update({"d_" + mode: distances, "score_" + mode: probabilities[:, 1], "pred_" + mode: probabilities.argmax(axis=1)})
                metrics = {mode: common.binary_metrics_reference(labels, arrays["score_" + mode].tolist(), arrays["pred_" + mode].tolist()) for mode in common.MODES}
                with mock.patch.object(runner, "np", np, create=True):
                    runner.atomic_npz(task / "support.npz", support_rgb=z, support_freq=z, proto_rgb=proto, proto_freq=proto)
                    runner.atomic_npz(output / "per_query" / (key + ".npz"), **arrays)
                for name, value in (("support.json", support_data), ("sampling.json.gz", sampling), ("metrics.json", metrics),
                                    ("class_metrics.json", common.class_rows(identity, support_data, arrays))):
                    common.atomic_json(task / name, value)
                files = [f"per_query/{key}.npz"] + [f"per_task/{key}/{name}" for name in
                         ("support.npz", "support.json", "sampling.json.gz", "metrics.json", "class_metrics.json")]
                common.atomic_json(task / "DONE.json", {"status": "DONE", "identity": identity, "identity_sha256": common.canonical_hash(identity),
                                   "query_count": 8, "processed_query_count": 8, "logit_parity": "PASS",
                                   "files": {p: common.sha256_file(output / p) for p in files}})
        manifest = {"tasks": tasks, "protocol": protocol, "environment": {"synthetic": True},
                    "git_commit": "synthetic", "reproduction_command": "SYNTHETIC TEST FIXTURE ONLY",
                    "dataset_inventories": {name: {"n_images": len(records)} for name, records in inventories.items()}}
        common.atomic_json(output / "run_manifest.json", manifest)
        for generator in common.GENERATORS:
            directory = output / "formal" / generator
            identities = {k: i for k, i in tasks.items() if i["generator"] == generator}
            rows = []
            for identity in identities.values():
                rows.append({"exclude_class": generator, "seed": identity["seed"], "model_mode": "dual", "branch_mode": "dual",
                             "ckpt_step": 15000, "support_shot": 10, "ckpt_path": identity["ckpt_path"], "freq_stats_path": identity["stats_path"],
                             "num_real_support": 10, "num_fake_support": 10, "num_real_query": 4, "num_fake_query": 4,
                             **common.read_json(output / "per_task" / f"{generator}_seed{identity['seed']}" / "metrics.json")["dual"]})
            common.write_csv(directory / "ddfsd_eval_per_seed.csv", rows)
            common.write_csv(directory / "ddfsd_eval_summary.csv", [{"synthetic": True}])
            common.atomic_json(directory / "config.json", {"git_commit": "synthetic", "seeds": list(common.SEEDS), "shot": 10,
                               "tau": .2, "tau_r": .1, "max_eval_query_per_class": 0, "strict_formal_eval_images": True})
            common.atomic_bytes(directory / "reference.log", b"SYNTHETIC TEST FIXTURE ONLY\n")
            files = [f"formal/{generator}/{name}" for name in ("ddfsd_eval_per_seed.csv", "ddfsd_eval_summary.csv", "config.json", "reference.log")]
            common.atomic_json(directory / "REFERENCE_DONE.json", {"status": "DONE", "identity": {"tasks": identities},
                               "identity_sha256": common.canonical_hash({"tasks": identities}), "files": {p: common.sha256_file(output / p) for p in files}})
        return manifest

    def test_synthetic_30_tasks_recomputation_report_archive_partial_and_parity(self):
        with tempfile.TemporaryDirectory(prefix="r21_SYNTHETIC_") as temporary:
            output = Path(temporary) / "SYNTHETIC_RESULTS"
            manifest = self.make_fixture(output)
            key = "ADM_seed42"
            classes, metrics, marker = analyzer.validate_task(output, key, manifest["tasks"][key])
            self.assertEqual(len(classes), 2)
            self.assertEqual(metrics["dual"]["auc"], 1)
            with mock.patch.object(analyzer, "plot_figures"):
                result = analyzer.analyze(output, bootstrap_repeats=10, package=True)
                self.assertEqual(result["status"], "COMPLETE_AND_MATCHING")
                self.assertEqual(result["class_rows"], 60)
                self.assertEqual(result["completed_tasks"], 30)
                with tarfile.open(output.parent / (output.name + "_review.tar.gz")) as bundle:
                    names = bundle.getnames()
                    self.assertFalse(any(name.endswith((".npz", ".json.gz")) for name in names))
                    self.assertTrue(any(name.endswith("R21_REPORT.md") for name in names))
                reference = output / "formal" / "ADM" / "REFERENCE_DONE.json"
                csv_path = reference.parent / "ddfsd_eval_per_seed.csv"
                with open(csv_path, encoding="utf-8", newline="") as handle:
                    official_rows = list(csv.DictReader(handle))
                official_rows[0]["acc"] = 0
                common.write_csv(csv_path, official_rows)
                reference_marker = common.read_json(reference)
                reference_marker["files"]["formal/ADM/ddfsd_eval_per_seed.csv"] = common.sha256_file(csv_path)
                common.atomic_json(reference, reference_marker)
                result = analyzer.analyze(output, bootstrap_repeats=10)
                self.assertEqual(result["status"], "PARITY_PENDING_OR_MISMATCH")
                reference.unlink()
                result = analyzer.analyze(output, bootstrap_repeats=10)
                self.assertEqual(result["status"], "PARITY_PENDING_OR_MISMATCH")
                (output / "per_task" / key / "DONE.json").unlink()
                result = analyzer.analyze(output, bootstrap_repeats=10)
                self.assertEqual(result["status"], "PARTIAL_29_OF_30")

    def test_synthetic_corruption_missing_query_nan_wrong_shape_and_metrics(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="r21_SYNTHETIC_") as temporary:
            output = Path(temporary)
            manifest = self.make_fixture(output)
            key = "ADM_seed42"
            marker_path = output / "per_task" / key / "DONE.json"
            original_marker = common.read_json(marker_path)
            npz_path = output / "per_query" / (key + ".npz")
            with np.load(npz_path, allow_pickle=False) as source:
                original = {k: source[k] for k in source.files}
            for mutate in (lambda a: a.update(d_rgb=np.full((8, 2), np.nan)),
                           lambda a: a.update(d_rgb=np.zeros((8, 3))),
                           lambda a: a.update(labels=a["labels"][:-1]),
                           lambda a: a.update(d_dual=np.ones((8, 2))),
                           lambda a: a.update(source_index=np.zeros(8, dtype=int))):
                arrays = copy.deepcopy(original)
                mutate(arrays)
                with mock.patch.object(runner, "np", np, create=True):
                    runner.atomic_npz(npz_path, **arrays)
                marker = copy.deepcopy(original_marker)
                marker["files"][f"per_query/{key}.npz"] = common.sha256_file(npz_path)
                common.atomic_json(marker_path, marker)
                with self.assertRaises(ValueError):
                    analyzer.validate_task(output, key, manifest["tasks"][key])
            with mock.patch.object(runner, "np", np, create=True):
                runner.atomic_npz(npz_path, **original)
            marker = copy.deepcopy(original_marker)
            marker["files"][f"per_query/{key}.npz"] = common.sha256_file(npz_path)
            metrics_path = output / "per_task" / key / "metrics.json"
            metrics = common.read_json(metrics_path)
            metrics["rgb"]["ap"] = .9
            common.atomic_json(metrics_path, metrics)
            marker["files"][f"per_task/{key}/metrics.json"] = common.sha256_file(metrics_path)
            common.atomic_json(marker_path, marker)
            with self.assertRaises(ValueError):
                analyzer.validate_task(output, key, manifest["tasks"][key])


if __name__ == "__main__":
    unittest.main()
