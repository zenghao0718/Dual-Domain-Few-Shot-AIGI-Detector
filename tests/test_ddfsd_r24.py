"""SYNTHETIC tests: no GenImage, checkpoint, CLIP download or GPU inference."""

import ast
from contextlib import nullcontext
import copy
import csv
import importlib.util
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import types
from typing import Any, Dict, List, Optional
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util import ddfsd_r24 as logic
from util import ddfsd_r24_runtime as runtime

spec = importlib.util.spec_from_file_location("r24_test_entry", ROOT / "test_ddfsd_r24.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


SYNTHETIC_BASES = {"acc": [.123456789 + .13 * g for g in range(6)],
                   "ap": [.234567891 + .09 * g for g in range(6)],
                   "auc": [.612345678 + .05 * g for g in range(6)]}
SYNTHETIC_OFFSETS = {"acc": [-.004, -.002, 0, .002, .004],
                     "ap": [.009, -.006, .003, -.012, .006],
                     "auc": [-.005, .010, -.015, .020, -.010]}


def synthetic_metrics(generator, seed):
    g, s = logic.GENERATORS.index(generator), logic.SEEDS.index(seed)
    return {metric: SYNTHETIC_BASES[metric][g] + SYNTHETIC_OFFSETS[metric][s]
            for metric in ("acc", "ap", "auc")}


def synthetic_raw_rows(condition="label_noise_10pct"):
    return [{"generator": generator, "seed": seed, "condition": condition,
             **synthetic_metrics(generator, seed)} for generator in logic.GENERATORS for seed in logic.SEEDS]


class SelectionTests(unittest.TestCase):
    def test_noise_zero_exact_official_indices_and_order(self):
        from util.ddfsd_sampling import sample_support_query_indices
        for seed in logic.SEEDS:
            selection = logic.label_noise_selection(37, 51, seed, "ADM", 0)
            for cls, size in enumerate((37, 51)):
                support, query = sample_support_query_indices(size, 10, seed)
                self.assertEqual(selection["groups"][cls], [(cls, i) for i in support])
                self.assertEqual(selection["query_indices"][cls], query)
            self.assertEqual(selection["mislabeled_total"], 0)

    def test_noise_counts_conservation_queries_reproducibility_and_nesting(self):
        for generator in logic.GENERATORS:
            for seed in logic.SEEDS:
                clean = logic.label_noise_selection(53, 47, seed, generator, 0)
                previous = [set(), set()]
                for p, k in ((.1, 1), (.2, 2), (.3, 3)):
                    with self.subTest(generator=generator, seed=seed, p=p):
                        noisy = logic.label_noise_selection(53, 47, seed, generator, p)
                        self.assertEqual(noisy, logic.label_noise_selection(53, 47, seed, generator, p))
                        self.assertEqual(noisy["query_indices"], clean["query_indices"])
                        self.assertEqual([len(g) for g in noisy["groups"]], [10, 10])
                        self.assertEqual(set(sum(noisy["groups"], [])), set(sum(clean["groups"], [])))
                        self.assertEqual(sum(source != observed for observed, group in enumerate(noisy["groups"]) for source, _ in group), 2 * k)
                        self.assertEqual(noisy["mislabeled_total"], 2 * k)
                        for cls in (0, 1):
                            errors = set(noisy["swapped_indices"][cls])
                            self.assertEqual(len(errors), k)
                            self.assertTrue(previous[cls].issubset(errors))
                            previous[cls] = errors

    def test_stable_rng_ignores_python_hash_randomization(self):
        code = "from util.ddfsd_r24 import stable_seed; print(stable_seed('label_noise_schedule','glide',42))"
        results = [subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT,
                   env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip() for seed in ("1", "2")]
        self.assertEqual(results[0], results[1])

    def vectors(self, count=31):
        return [[math.cos(i * .19), math.sin(i * .19), 0.] for i in range(count)]

    def test_clip_same_anchor_true_top10_and_full_query(self):
        real, fake = self.vectors(), self.vectors(27)
        result = logic.clip_selection(real, fake, 42, "glide")
        anchor = result["anchor_index"]
        self.assertIn((0, anchor), result["groups"][0])
        for cls, values in enumerate((real, fake)):
            scores = [logic.cosine(v, real[anchor]) for v in values]
            if cls == 0:
                scores[anchor] = 1.
            expected = sorted(range(len(values)), key=lambda i: (-scores[i], i))[:10]
            self.assertEqual([i for _, i in result["groups"][cls]], expected)
        self.assertEqual(result, logic.clip_selection(real, fake, 42, "glide"))
        self.assertEqual(sum(map(len, result["query_indices"])), 31 + 27 - 20)
        logic.validate_selection(result, 31, 27)

    def test_clip_ties_are_deterministic_and_anchor_never_disappears(self):
        vectors = [[1., 0.]] * 40
        result = logic.clip_selection(vectors, vectors, 104, "ADM")
        self.assertEqual([i for _, i in result["groups"][1]], list(range(10)))
        self.assertIn((0, result["anchor_index"]), result["groups"][0])
        self.assertEqual(result["diagnostics"]["concentration_status"], "CONCENTRATION_NOT_VERIFIED")
        self.assertEqual(logic.top_indices([.5, .5, .8, .5], 3), [2, 0, 1])

    def test_pairwise_excludes_self_and_unverified_is_retained(self):
        self.assertEqual(logic.pairwise_mean([[1, 0], [0, 1], [-1, 0]], [0, 1, 2]), -1 / 3)
        result = logic.clip_selection([[1., 0.]] * 12, [[1., 0.]] * 13, 42, "SD")
        for cls in ("real", "fake"):
            self.assertEqual(result["diagnostics"][cls + "_pairwise_delta"], 0)
            self.assertEqual(result["diagnostics"][cls + "_anchor_mean_concentrated"], 1)

    def test_support_manifest_preserves_true_and_observed_labels(self):
        selected = logic.label_noise_selection(30, 30, 42, "ADM", .2)
        paths = [[f"{name}/val/SYNTHETIC_{i}.png" for i in range(30)] for name in ("real", "ADM")]
        rows = logic.support_rows("ADM", 42, "label_noise", "label_noise_20pct", selected, paths)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r["is_swapped"] for r in rows), 4)
        self.assertEqual(sum(r["observed_label"] == 0 for r in rows), 10)
        for row in rows:
            self.assertEqual(row["relative_image_path"], paths[row["true_label"]][row["source_index"]])

    def test_errors_small_class_invalid_rate_duplicate_seeds_leakage(self):
        for size in (0, 10):
            with self.assertRaisesRegex(ValueError, "at least 11"):
                logic.label_noise_selection(size, 20, 42, "ADM", .1)
        for p in (-.1, .15, 1, math.nan):
            with self.assertRaises(ValueError):
                logic.label_noise_selection(20, 20, 42, "ADM", p)
        for s in ("42,42", "", "x", "-1"):
            with self.assertRaises(ValueError):
                logic.seeds_from_string(s)
        selection = logic.label_noise_selection(25, 25, 42, "ADM", 0)
        selection["query_indices"][0].append(selection["groups"][0][0][1])
        with self.assertRaises(ValueError):
            logic.validate_selection(selection, 25, 25)

    def test_cache_validation_detects_order_count_model_dimension_nonfinite_and_missing(self):
        records = [{"index": i, "relative_path": f"SYNTHETIC_{i}.png", "sha256": str(i)} for i in range(12)]
        features = [[1., 0.]] * 12
        identity = {"model_id": "SYNTHETIC", "revision": "one"}
        metadata = {"schema": 1, "inventory_sha256": logic.digest(records), "n_images": 12,
                    "feature_dim": 2, "clip_identity": identity}
        self.assertTrue(logic.validate_cache(metadata, records, features, identity))
        for bad_records, bad_features, model in ((list(reversed(records)), features, identity),
                 (records[:-1], features, identity), (records, features[:-1], identity),
                 (records, [[math.nan, 0.]] * 12, identity), (records, [[1.]] * 12, identity),
                 (records, [[0., 0.]] * 12, identity), (records, features, {**identity, "revision": "two"})):
            with self.subTest(model=model), self.assertRaises(ValueError):
                logic.validate_cache(metadata, bad_records, bad_features, model)

    def test_checkpoint_metadata_legacy_dual_single_branch_and_conflicts(self):
        base = {"step": 15000, "exclude_class": "ADM"}
        self.assertEqual(logic.checkpoint_metadata(base, "ADM")["mode_confirmation"], "legacy_dual_strict_load_required")
        self.assertEqual(logic.checkpoint_metadata({**base, "model_mode": "dual"}, "ADM")["mode_confirmation"], "explicit_dual")
        for change in ({"step": 14000}, {"exclude_class": "SD"}, {"model_mode": "rgb-only"},
                       {"model_mode": "dual", "args": types.SimpleNamespace(model_mode="freq-only")},
                       {"config": {"model_mode": "dual"}, "args": {"model_mode": "rgb-only"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                logic.checkpoint_metadata({**base, **change}, "ADM")

    def test_summary_uses_unrounded_mean_and_population_std(self):
        rows = [{"condition": "SYNTHETIC", "acc": v, "ap": v, "auc": v} for v in (.123456789, .234567891)]
        result = logic.summarize(rows)[0]
        self.assertEqual(result["acc_mean"], (.123456789 + .234567891) / 2)
        self.assertAlmostEqual(result["acc_std"], (.234567891 - .123456789) / 2)
        self.assertEqual(result["std_definition"], "seed_population_ddof0")

    def test_cli_defaults_errors_and_missing_resource_paths(self):
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_") as temp:
            args = entry.parse_args(["--data-root", temp + "/data", "--ckpt-root", temp + "/ckpt",
                                    "--clip-cache-root", temp + "/cache", "--output-dir", temp + "/out"])
            self.assertEqual(args.seeds, list(logic.SEEDS))
            self.assertEqual(args.noise_rates, [.1, .2, .3])
            self.assertEqual(args.generators, list(logic.GENERATORS))
            with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint/stats"):
                entry.checkpoint_paths(args, "ADM")
            with self.assertRaisesRegex(FileNotFoundError, "Missing completed CLIP cache"):
                runtime.load_cache(Path(temp) / "cache", "real", [], None)

    def test_help_and_shell_literal_arguments(self):
        for script in ("test_ddfsd_r24.py", "tools/build_r24_clip_cache.py"):
            completed = subprocess.run([sys.executable, "-B", str(ROOT / script), "--help"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        git, bash = shutil.which("git"), shutil.which("bash")
        if not bash and git:
            guess = Path(git).parent.parent / "bin" / "bash.exe"
            bash = str(guess) if guess.exists() else None
        if not bash:
            self.skipTest("NOT_RUN_ENV_MISSING: bash")
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_") as temp:
            root = Path(temp)
            helper, captured = root / "mock_python.sh", root / "argv.txt"
            helper.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$R24_TEST_ARGV"\n')
            helper.chmod(0o755)
            env = {**os.environ, "DATA_ROOT": "/SYNTHETIC/data path", "OUTPUT_DIR": "/SYNTHETIC/out path",
                   "PYTHON": helper.as_posix(), "R24_TEST_ARGV": captured.as_posix()}
            env.pop("CHECKPOINT_TEMPLATE", None)
            completed = subprocess.run([bash, (ROOT / "scripts/eval_r24.sh").as_posix()], env=env, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            args = captured.read_text().splitlines()
            self.assertEqual(args[args.index("--checkpoint-template") + 1], "exclude_{generator}/ckpt/ddfsd_step[15000].pth")
            self.assertEqual(args[args.index("--data-root") + 1], "/SYNTHETIC/data path")

    def test_saved_case_identity_completion_and_counts(self):
        selected = logic.label_noise_selection(21, 22, 42, "ADM", .1)
        identity = {"SYNTHETIC": True, "condition": "label_noise_10pct"}
        result = {"identity": identity, "selection": selected,
                  "metrics": {"acc": .5, "ap": .5, "auc": .5, "alpha_real": .5, "alpha_fake": .5,
                              "num_real_support": 10, "num_fake_support": 10, "num_real_query": 11, "num_fake_query": 12}}
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_") as temp:
            root = Path(temp)
            logic.write_json(root / "result.json.gz", result)
            marker = {"status": "DONE", "identity": identity, "sha256": logic.file_sha256(root / "result.json.gz")}
            # Semantic verification is possible before publishing DONE.
            self.assertEqual(entry.load_case(root, identity, selected, (21, 22), marker)["metrics"]["num_fake_query"], 12)
            self.assertFalse((root / "DONE.json").exists())
            logic.write_json(root / "DONE.json", marker)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                entry.load_case(root, {**identity, "changed": True}, selected, (21, 22))
            result["metrics"]["num_fake_query"] = 11
            logic.write_json(root / "result.json.gz", result)
            marker["sha256"] = logic.file_sha256(root / "result.json.gz")
            with self.assertRaisesRegex(ValueError, "Incomplete support/query counts"):
                entry.load_case(root, identity, selected, (21, 22), marker)

    def test_synthetic_120_case_csv_summary_partial_status_and_no_fake_clean_baseline(self):
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_") as temp:
            args = types.SimpleNamespace(output_dir=Path(temp), generators=list(logic.GENERATORS), seeds=list(logic.SEEDS),
                                         experiment="all", noise_rates=list(logic.NOISE_RATES), use_fp16=True)
            completed, jobs = {}, {}
            for generator in logic.GENERATORS:
                for seed in logic.SEEDS:
                    selections = [("label_noise", logic.condition_name(rate), logic.label_noise_selection(21, 22, seed, generator, rate))
                                  for rate in logic.NOISE_RATES]
                    selections.append(("clip_bias", "clip_concentrated", logic.clip_selection([[1., 0.]] * 21, [[1., 0.]] * 22, seed, generator)))
                    for experiment, condition, selection in selections:
                        key = f"{generator}/{seed}/{condition}"
                        jobs[key] = None
                        completed[key] = {"generator": generator, "seed": seed, "experiment": experiment, "condition": condition,
                                          "identity": {"SYNTHETIC": True, "checkpoint": {"ckpt_sha256": "SYNTHETIC", "stats_sha256": "SYNTHETIC"},
                                                       "cache_sha256": "SYNTHETIC"}, "selection": selection, "support_manifest": [],
                                          "metrics": {**synthetic_metrics(generator, seed), "alpha_real": .5, "alpha_fake": .5,
                                                      "num_real_support": 10, "num_fake_support": 10, "num_real_query": 11, "num_fake_query": 12}}
            entry.summarize_run(args, jobs, completed, {"reference": "SYNTHETIC_NOT_PROVIDED"})
            root = Path(temp) / "r24"
            status = logic.read_json(root / "run_status.json")
            self.assertEqual(status["status"], "COMPLETE_R24_A_B")
            self.assertEqual(status["completed"], 120)
            self.assertEqual(status["concentration_not_verified"], 30)
            self.assertFalse((root / "baseline_reference_summary.csv").exists())
            self.assertFalse((root / "baseline_comparison.csv").exists())
            with open(root / "exclude_ADM/label_noise/per_seed.csv", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 15)
            with open(root / "exclude_ADM/clip_bias/per_seed.csv", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 5)
            with open(root / "all_generators_summary.csv", newline="") as handle:
                values = list(csv.DictReader(handle))
            macros = [row for row in values if row["scope"] == "MACRO_SIX_GENERATORS"]
            self.assertEqual({r["condition"] for r in macros}, {"label_noise_10pct", "label_noise_20pct", "label_noise_30pct", "clip_concentrated"})
            for row in macros:
                for metric in ("acc", "ap", "auc"):
                    self.assertAlmostEqual(float(row[metric + "_mean"]), statistics.mean(SYNTHETIC_BASES[metric]) + statistics.mean(SYNTHETIC_OFFSETS[metric]), places=14)
                    self.assertAlmostEqual(float(row[metric + "_std"]), statistics.pstdev(SYNTHETIC_OFFSETS[metric]), places=14)
            with open(root / "macro_per_seed.csv", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 20)
            del completed[next(iter(completed))]
            entry.summarize_run(args, jobs, completed, {"reference": "SYNTHETIC_NOT_PROVIDED"})
            self.assertEqual(logic.read_json(root / "run_status.json")["status"], "PARTIAL_119_OF_120")
            with open(root / "all_generators_summary.csv", newline="") as handle:
                values = list(csv.DictReader(handle))
            affected = [r for r in values if r["condition"] == "label_noise_10pct" and r["generator"] == "MACRO"]
            self.assertEqual(affected, [])
            self.assertEqual(len([r for r in values if r["scope"] == "MACRO_SIX_GENERATORS"]), 3)


class PaperMacroStatisticsTests(unittest.TestCase):
    def test_three_metric_macro_stds_are_seed_variation_not_between_generators(self):
        rows = synthetic_raw_rows()
        result = logic.paper_macro_by_seed(rows, "label_noise_10pct")
        self.assertIsNotNone(result)
        self.assertEqual([r["seed"] for r in result["per_seed"]], list(logic.SEEDS))
        for metric in ("acc", "ap", "auc"):
            with self.subTest(metric=metric):
                actual = result["summary"]
                expected_std = statistics.pstdev(SYNTHETIC_OFFSETS[metric])
                old_wrong_std = statistics.pstdev(SYNTHETIC_BASES[metric])
                self.assertAlmostEqual(actual[metric + "_std"], expected_std, places=14)
                self.assertGreater(old_wrong_std, 5 * actual[metric + "_std"])
                for i, row in enumerate(result["per_seed"]):
                    self.assertAlmostEqual(row[metric], statistics.mean(SYNTHETIC_BASES[metric]) + SYNTHETIC_OFFSETS[metric][i], places=14)
        self.assertEqual(len({round(result["summary"][m + "_std"], 10) for m in ("acc", "ap", "auc")}), 3)

    def test_opposite_seed_fluctuations_cancel_before_std_not_average_rms_or_flat_std(self):
        rows = synthetic_raw_rows()
        for row in rows:
            g, s = logic.GENERATORS.index(row["generator"]), logic.SEEDS.index(row["seed"])
            for metric in ("acc", "ap", "auc"):
                row[metric] = SYNTHETIC_BASES[metric][g] + (-1 if g % 2 else 1) * SYNTHETIC_OFFSETS[metric][s]
        summary = logic.paper_macro_by_seed(rows, "label_noise_10pct")["summary"]
        for metric in ("acc", "ap", "auc"):
            self.assertAlmostEqual(summary[metric + "_std"], 0, places=14)
            self.assertGreater(statistics.mean(statistics.pstdev(r[metric] for r in rows if r["generator"] == g)
                                              for g in logic.GENERATORS), 0)
            self.assertGreater(statistics.pstdev(r[metric] for r in rows), .01)

    def test_input_order_and_query_counts_do_not_change_equal_weight_macro(self):
        rows = synthetic_raw_rows()
        expected = logic.paper_macro_by_seed(rows, "label_noise_10pct")
        for row in rows:
            row["num_real_query"] = 10 ** logic.GENERATORS.index(row["generator"])
            row["num_fake_query"] = 7 * row["num_real_query"]
        self.assertEqual(logic.paper_macro_by_seed(list(reversed(rows)), "label_noise_10pct"), expected)

    def test_missing_generator_seed_duplicate_unknown_ids_and_nonfinite_never_publish_macro(self):
        rows = synthetic_raw_rows()
        cases = [[], [r for r in rows if r["generator"] != "ADM"], rows[:-1],
                 rows + [copy.deepcopy(rows[0])], rows[:-1] + [copy.deepcopy(rows[0])],
                 [{**r, "generator": "unknown"} if i == 0 else r for i, r in enumerate(rows)],
                 [{**r, "seed": 105} if i == 0 else r for i, r in enumerate(rows)]]
        for metric in ("acc", "ap", "auc"):
            for invalid in (math.nan, math.inf, -math.inf):
                bad = copy.deepcopy(rows)
                bad[0][metric] = invalid
                cases.append(bad)
        for number, case in enumerate(cases):
            with self.subTest(case=number):
                self.assertIsNone(logic.paper_macro_by_seed(case, "label_noise_10pct"))

    def fixture(self, directory):
        args = types.SimpleNamespace(output_dir=Path(directory), generators=list(logic.GENERATORS), seeds=list(logic.SEEDS),
                                     experiment="all", noise_rates=list(logic.NOISE_RATES), use_fp16=True)
        completed = {}
        for row in synthetic_raw_rows():
            key = f"{row['generator']}/{row['seed']}/label_noise_10pct"
            selection = logic.label_noise_selection(21, 22, row["seed"], row["generator"], .1)
            completed[key] = {"generator": row["generator"], "seed": row["seed"], "experiment": "label_noise",
                              "condition": row["condition"], "identity": {"SYNTHETIC": True,
                              "checkpoint": {"ckpt_sha256": "SYNTHETIC", "stats_sha256": "SYNTHETIC"}, "cache_sha256": ""},
                              "selection": selection, "support_manifest": [],
                              "metrics": {**{m: row[m] for m in ("acc", "ap", "auc")}, "alpha_real": .5, "alpha_fake": .5,
                                          "num_real_support": 10, "num_fake_support": 10, "num_real_query": 11, "num_fake_query": 12}}
        jobs = {"SYNTHETIC_JOB_" + str(i): None for i in range(120)}
        return args, jobs, completed

    def read_csv(self, path):
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_one_condition_can_publish_paper_macro_before_120_tasks_and_keep_raw_precision(self):
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_macro_") as temp:
            args, jobs, completed = self.fixture(temp)
            entry.summarize_run(args, jobs, completed, {"reference": "SYNTHETIC_NOT_PROVIDED"})
            root = Path(temp) / "r24"
            self.assertEqual(logic.read_json(root / "run_status.json")["status"], "PARTIAL_30_OF_120")
            summaries = self.read_csv(root / "all_generators_summary.csv")
            macros = [r for r in summaries if r["scope"] == "MACRO_SIX_GENERATORS"]
            self.assertEqual(len(macros), 1)
            self.assertEqual(macros[0]["std_definition"], "population_std_of_seedwise_six_generator_macros_ddof0")
            witnesses = self.read_csv(root / "macro_per_seed.csv")
            self.assertEqual([int(r["seed"]) for r in witnesses], list(logic.SEEDS))
            for metric in ("acc", "ap", "auc"):
                self.assertEqual(float(macros[0][metric + "_mean"]), statistics.mean(float(r[metric]) for r in witnesses))
                self.assertEqual(float(macros[0][metric + "_std"]), statistics.pstdev(float(r[metric]) for r in witnesses))
                self.assertNotEqual(float(macros[0][metric + "_mean"]), round(float(macros[0][metric + "_mean"]), 4))
            generator = next(r for r in summaries if r["generator"] == "ADM")
            raw = self.read_csv(root / "exclude_ADM/label_noise/per_seed.csv")
            for metric in ("acc", "ap", "auc"):
                self.assertEqual(float(generator[metric + "_std"]), statistics.pstdev(float(r[metric]) for r in raw))
            report = (root / "R24_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("先跨六生成器等权平均", report)
            self.assertIn("五个独立训练 checkpoint", report)
            for metric in ("acc", "ap", "auc"):
                self.assertIn(f"{100 * float(macros[0][metric + '_mean']):.2f} ± {100 * float(macros[0][metric + '_std']):.2f}", report)

    def test_adm_only_single_seed_missing_seed_and_duplicate_still_write_progress_without_macros(self):
        for scope in ("ADM-only", "single-seed", "missing-seed", "duplicate", "unknown-generator"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_partial_") as temp:
                args, jobs, completed = self.fixture(temp)
                if scope == "ADM-only":
                    args.generators = ["ADM"]
                    completed = {k: r for k, r in completed.items() if r["generator"] == "ADM"}
                elif scope == "single-seed":
                    args.seeds = [42]
                    completed = {k: r for k, r in completed.items() if r["seed"] == 42}
                elif scope == "missing-seed":
                    completed.pop(next(iter(completed)))
                elif scope == "duplicate":
                    completed["SYNTHETIC_DUPLICATE"] = copy.deepcopy(next(iter(completed.values())))
                else:
                    extra = copy.deepcopy(next(iter(completed.values())))
                    extra["generator"] = "unknown"
                    completed["SYNTHETIC_UNKNOWN_GENERATOR"] = extra
                entry.summarize_run(args, jobs, completed, {"reference": "SYNTHETIC_NOT_PROVIDED"})
                root = Path(temp) / "r24"
                self.assertEqual(logic.read_json(root / "run_status.json")["completed"], len(completed))
                self.assertEqual(self.read_csv(root / "macro_per_seed.csv"), [])
                self.assertTrue(all(r["scope"] == "GENERATOR" for r in self.read_csv(root / "all_generators_summary.csv")))
                self.assertTrue((root / "exclude_ADM/label_noise/per_seed.csv").exists())

    def test_empty_refresh_removes_stale_macro_tables(self):
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_empty_") as temp:
            args, jobs, completed = self.fixture(temp)
            entry.summarize_run(args, jobs, completed, {})
            entry.summarize_run(args, jobs, {}, {})
            root = Path(temp) / "r24"
            self.assertEqual(self.read_csv(root / "all_generators_summary.csv"), [])
            self.assertEqual(self.read_csv(root / "macro_per_seed.csv"), [])
            self.assertEqual(logic.read_json(root / "run_status.json")["status"], "PARTIAL_0_OF_120")

    def test_external_baseline_remains_existing_file_and_generator_mean_delta(self):
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_baseline_") as temp:
            args, jobs, completed = self.fixture(temp)
            args.baseline_csv = str(Path(temp) / "SYNTHETIC_existing_formal.csv")
            args.baseline_reference = "SYNTHETIC_EXTERNAL_FILE_NOT_PAPER_DATA"
            raw = synthetic_raw_rows()
            for row in raw:
                for metric in ("acc", "ap", "auc"):
                    row[metric] += .01
            logic.write_csv(args.baseline_csv, raw)
            clean, baseline = entry.existing_baseline(args)
            self.assertEqual(baseline["status"], "EXTERNAL_EXISTING_FORMAL_CSV_NOT_RERUN")
            baseline["summary"] = clean
            entry.summarize_run(args, jobs, completed, baseline)
            comparisons = self.read_csv(Path(temp) / "r24/baseline_comparison.csv")
            self.assertEqual(len(comparisons), 6)
            for row in comparisons:
                for metric in ("acc", "ap", "auc"):
                    self.assertAlmostEqual(float(row[metric + "_delta"]), -.01, places=14)


def numpy_evaluation_namespace(source):
    """Execute actual eval/loss functions with small NumPy tensor test doubles."""
    import numpy as np

    class Tensor(np.ndarray):
        def __new__(cls, value):
            return np.asarray(value, dtype=np.float32).view(cls)

        def __getitem__(self, key): return Tensor(super().__getitem__(key))
        def float(self): return Tensor(self)
        def to(self, *args, **kwargs): return self
        def cpu(self): return self
        def detach(self): return self.copy()
        def permute(self, *dims): return Tensor(np.asarray(self).transpose(dims))
        def unsqueeze(self, dim): return Tensor(np.expand_dims(np.asarray(self), dim))
        def squeeze(self, dim=None): return Tensor(np.squeeze(np.asarray(self), axis=dim))
        def pow(self, exponent): return Tensor(np.asarray(self) ** exponent)
        def sum(self, dim=None): return Tensor(np.asarray(self).sum(axis=dim))
        def mean(self, dim=None): return Tensor(np.asarray(self).mean(axis=dim))
        def clamp_min(self, lower): return Tensor(np.maximum(self, lower))
        def clamp(self, low, high): return Tensor(np.clip(self, low, high))
        def softmax(self, dim):
            value = np.asarray(self)
            result = np.exp(value - value.max(axis=dim, keepdims=True))
            return Tensor(result / result.sum(axis=dim, keepdims=True))
        def argmax(self, dim): return np.asarray(self).argmax(axis=dim).view(Tensor)

    torch = types.SimpleNamespace(Tensor=Tensor, device=object, no_grad=lambda: (lambda function: function),
                                  stack=lambda values, dim=0: Tensor(np.stack(values, axis=dim)),
                                  cat=lambda values, dim=0: Tensor(np.concatenate(values, axis=dim)),
                                  sqrt=lambda value: Tensor(np.sqrt(value)), sigmoid=lambda value: Tensor(1 / (1 + np.exp(-np.asarray(value)))))
    namespace = {"np": np, "torch": torch, "math": math, "autocast": lambda **kw: nullcontext(),
                 "F": types.SimpleNamespace(normalize=lambda value, p, dim: Tensor(np.asarray(value) / np.maximum(np.linalg.norm(np.asarray(value), ord=p, axis=dim, keepdims=True), 1e-12))),
                 "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
                 "VALID_MODEL_MODES": {"dual", "rgb-only", "freq-only"}, "VALID_BRANCH_MODES": {"dual", "rgb-only", "freq-only"},
                 "sample_support_query_indices": __import__("util.ddfsd_sampling", fromlist=["sample_support_query_indices"]).sample_support_query_indices}
    names = {"_validate_model_mode", "ordinary_euclidean_distance", "compute_prototypes", "compute_support_sigmas", "compute_alpha", "fuse_distances", "compute_query_logits"}
    loss_tree = ast.parse((ROOT / "model/ddfsd_losses.py").read_text())
    nodes = [n for n in loss_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), "SYNTHETIC_actual_losses", "exec"), namespace)
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    # binary_metrics itself is unchanged; here use simple transparent summaries
    # so parity compares every score/prediction without a sklearn installation.
    exec(compile(ast.Module(nodes, type_ignores=[]), "SYNTHETIC_actual_eval", "exec"), namespace)
    namespace["binary_metrics"] = lambda labels, scores, preds: {
        "acc": sum(a == b for a, b in zip(labels, preds)) / len(labels),
        "SYNTHETIC_score_fingerprint": sum(p * (i + 1) for i, p in enumerate(scores)) / len(scores),
        "SYNTHETIC_labels": labels, "SYNTHETIC_scores": scores, "SYNTHETIC_preds": preds}
    namespace["Tensor"] = Tensor
    return namespace


@unittest.skipUnless(HAS_NUMPY, "NOT_RUN_ENV_MISSING: numpy (no installation attempted)")
class NumpyRegressionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        source = (ROOT / "util/ddfsd_eval.py").read_text()
        self.current = numpy_evaluation_namespace(source)
        self.original = numpy_evaluation_namespace((ROOT / "tests/fixtures/r24_original_ddfsd_eval.py.txt").read_text())
        tensor = self.current["Tensor"]

        class Dataset:
            def __init__(self, cls):
                self.paths = [f"SYNTHETIC_class{cls}_{i}.png" for i in range(28 + cls)]
                self.images = []
                for i in range(len(self.paths)):
                    value = np.zeros(512, dtype=np.float32)
                    angle = (.1 + i * .01) if cls == 0 else (1.3 + i * .01)
                    value[:2] = [math.cos(angle), math.sin(angle)]
                    self.images.append(tensor(value))
            def __len__(self): return len(self.paths)
            def __getitem__(self, index): return self.images[index], 0

        self.datasets = [Dataset(0), Dataset(1)]
        def loader(dataset, indices, batch_size, **kwargs):
            for offset in range(0, len(indices), batch_size):
                yield tensor(np.stack([dataset[i][0] for i in indices[offset:offset + batch_size]])), None

        for namespace in (self.current, self.original):
            namespace["load_ddfsd_class_dataset"] = lambda root, name, split, **kwargs: self.datasets[0 if name == "real" else 1]
            namespace["make_subset_loader"] = loader
        self.calls = []
        calls = self.calls
        class Model:
            def eval(self): return self
            def __call__(self, images):
                calls.append(np.asarray(images).copy())
                return {"z_rgb": images, "z_freq": tensor(np.asarray(images)[:, ::-1].copy())}
        self.model = Model()
        self.kwargs = dict(model=self.model, data_root="SYNTHETIC", fake_class="ADM", support_shot=10, seed=42,
                           batch_size=7, num_workers=0, device=types.SimpleNamespace(type="cpu"), use_fp16=False,
                           tau=.2, tau_r=.1, strict_images=True)

    def test_public_entrypoint_matches_immutable_original_including_indices_and_every_prediction(self):
        for return_indices in (False, True):
            original = self.original["evaluate_binary_few_shot"](**self.kwargs, return_indices=return_indices)
            current = self.current["evaluate_binary_few_shot"](**self.kwargs, return_indices=return_indices)
            self.assertEqual(current, original)

    def test_zero_noise_matches_original_and_support_tensor_is_1_10_2_512(self):
        original = self.original["evaluate_binary_few_shot"](**self.kwargs)
        selection = logic.label_noise_selection(28, 29, 42, "ADM", 0)
        shapes = []
        compute = self.current["compute_prototypes"]
        self.current["compute_prototypes"] = lambda rgb, freq, **kw: (shapes.append((rgb.shape, freq.shape)) or compute(rgb, freq, **kw))
        current = self.current["_evaluate_binary_few_shot_impl"](**self.kwargs, selection=selection, datasets=self.datasets, return_alpha=True)
        self.assertEqual({k: current[k] for k in original}, original)
        self.assertEqual(shapes, [((1, 10, 2, 512), (1, 10, 2, 512))])
        self.assertEqual(current["num_real_query"] + current["num_fake_query"], 28 + 29 - 20)

    def test_noise_changes_model_input_membership_and_keeps_all_query_labels(self):
        import numpy as np
        selection = logic.label_noise_selection(28, 29, 42, "ADM", .3)
        result = self.current["_evaluate_binary_few_shot_impl"](**self.kwargs, selection=selection, datasets=self.datasets, return_alpha=True)
        expected = np.stack([self.datasets[source][i][0] for group in selection["groups"] for source, i in group])
        np.testing.assert_array_equal(self.calls[0], expected)
        self.assertEqual(result["SYNTHETIC_labels"], [0] * 18 + [1] * 19)
        self.assertEqual(result["num_real_support"], 10)
        self.assertEqual(result["num_fake_support"], 10)

    def test_batch_size_does_not_change_query_set_or_scores(self):
        selection = logic.label_noise_selection(28, 29, 42, "ADM", .2)
        first = self.current["_evaluate_binary_few_shot_impl"](**self.kwargs, selection=selection, datasets=self.datasets)
        second = self.current["_evaluate_binary_few_shot_impl"](**{**self.kwargs, "batch_size": 13}, selection=selection, datasets=self.datasets)
        self.assertEqual(first, second)

    def test_numpy_cache_roundtrip_and_strict_corruption_checks(self):
        import numpy as np
        records = [{"index": i, "relative_path": f"real/val/SYNTHETIC_{i}.png", "sha256": str(i)} for i in range(12)]
        identity = {"model_id": "SYNTHETIC", "resolved_revision": "one"}
        features = np.tile(np.array([[1, 0]], dtype=np.float32), (12, 1))
        metadata = {"schema": 1, "class_name": "real", "n_images": 12, "feature_dim": 2,
                    "inventory_sha256": logic.digest(records), "clip_identity": identity}
        with tempfile.TemporaryDirectory(prefix="r24_SYNTHETIC_") as temp:
            root = Path(temp)
            runtime.save_cache(root / "real", features, metadata, records)
            values, stored, cache_id = runtime.load_cache(root, "real", records, identity)
            np.testing.assert_array_equal(values, features)
            self.assertEqual(stored, metadata)
            del values
            for changed, model in ((list(reversed(records)), identity), (records, {**identity, "resolved_revision": "two"})):
                with self.assertRaises(ValueError):
                    runtime.load_cache(root, "real", changed, model)
            (root / "real/features.npy").unlink()
            with self.assertRaises(ValueError):
                runtime.load_cache(root, "real", records, identity)
            with self.assertRaisesRegex(FileNotFoundError, "Missing completed CLIP cache"):
                runtime.load_cache(root, "ADM", records, identity)


if __name__ == "__main__":
    unittest.main()
