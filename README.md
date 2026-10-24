# DDFSD: Dual-Domain Prototypical Learning for Few-Shot AI-Generated Image Detection

<p align="center">
  <a href="https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector/stargazers"><img src="https://img.shields.io/github/stars/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector?style=flat-square" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.3.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch >= 2.3.0">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square" alt="Apache-2.0 License"></a>
</p>

---

## Overview

R2-1 support reliability diagnostics: [AutoDL one-command evaluation and local validation](docs/R21_RELIABILITY_USAGE.md).
This adds analysis of the existing final dual checkpoints; GPU experiment results are generated only when run on AutoDL.

R2-4 support-quality experiments (label noise and CLIP-concentrated support): [AutoDL preparation, commands and validation](docs/R2-4_support_quality_experiments.md). These use frozen final checkpoints and preserve the original main evaluation interface.

This repository provides the implementation of DDFSD, a few-shot detector for identifying AI-generated images from an unseen target generator. DDFSD learns dual-domain prototype spaces from RGB content and signed high-frequency Haar-DWT responses. Class-specific adaptive weights fuse the two domains according to support-set dispersion, allowing the more reliable domain to contribute more strongly to each class decision.

During episodic training, an asymmetric prototype margin strengthens real-fake separation while preserving structure between fake-generator classes. At test time, the encoders remain frozen: real and fake prototypes are built directly from the target support set, and the remaining target images are classified without gradient updates on the target generator.

<p align="center">
  <img src="./figs/ddfsd_framework.png" width="95%" alt="DDFSD framework">
</p>
<p align="center"><em>Fig. 1: Overall framework of DDFSD.</em></p>

---

## Repository Structure

```text
Dual-Domain-Few-Shot-AIGI-Detector/
|-- datasets/
|   `-- ddfsd_datasets.py          # GenImage loading, transforms, and data iterators
|-- figs/
|   `-- ddfsd_framework.png        # DDFSD framework figure
|-- model/
|   |-- ddfsd.py                   # RGB/frequency encoders and projection heads
|   `-- ddfsd_losses.py            # Prototype, fusion, and margin objectives
|-- scripts/
|   |-- train_main.sh              # Train one leave-one-generator-out task
|   |-- eval_main.sh               # Evaluate one trained task
|   `-- run_main_experiment.sh     # Run all leave-one-out tasks
|-- tests/
|   `-- test_ddfsd_core.py         # Core unit tests
|-- util/                          # Evaluation, frequency, sampling, and logging helpers
|-- train_ddfsd.py                 # Training entry point
|-- test_ddfsd.py                  # Formal evaluation entry point
|-- requirements.txt
|-- LICENSE
`-- README.md
```

---

## Dataset

The main protocol uses **GenImage** with the following seven project-level categories:

- `real`
- `ADM`
- `BigGAN`
- `glide`
- `Midjourney`
- `SD`
- `VQDM`

`SD` is the merged Stable Diffusion category used by this project. `DATA_ROOT` must point directly to the `GenImage` directory that contains all seven category directories:

```text
GenImage/
|-- real/
|   |-- train/
|   `-- val/
|-- ADM/
|   |-- train/
|   `-- val/
|-- BigGAN/
|   |-- train/
|   `-- val/
|-- glide/
|   |-- train/
|   `-- val/
|-- Midjourney/
|   |-- train/
|   `-- val/
|-- SD/
|   |-- train/
|   `-- val/
`-- VQDM/
    |-- train/
    `-- val/
```

GenImage is not included in this repository. The repository also does not currently provide a complete utility for automatically converting the original GenImage release into this seven-category layout; users must prepare the data accordingly.

---

## Compared Methods

The accompanying study compares DDFSD with eight representative AI-generated image detection methods covering different forensic domains and learning paradigms.

### Spatial-domain detectors

- **CNNSpot** learns transferable generation artifacts directly from RGB images using a convolutional classifier and generalization-oriented augmentation strategies.
- **NPR** models neighboring-pixel relationships to expose local structural traces introduced by upsampling operations in generative networks.

### Frequency-aware detector

- **FreqNet** performs frequency-aware modeling at both the image and intermediate-feature levels, explicitly exploiting frequency-domain cues including amplitude- and phase-related evidence.

### Spatial-frequency and foundation-model-based detectors

- **FatFormer** combines local spatial forgery traces with wavelet-frequency evidence in a pretrained vision-language framework.
- **UnivFD** uses frozen CLIP representations with a lightweight classifier to reduce overfitting to artifacts from specific generators.
- **VIB-Net** uses a variational information bottleneck to retain authenticity-relevant evidence while suppressing semantic information unrelated to the detection task.

### Few-shot detectors

- **FSD** learns a prototypical metric space in a single RGB domain through episodic training. It builds prototypes from a small support set for the target generator and performs inference without test-time gradient updates. DDFSD extends this paradigm with dual-domain prototypes, support-set reliability estimation, and an asymmetric prototype-margin constraint.
- **LIDA** extracts generation fingerprints from low bit planes and combines few-shot adaptation with similarity retrieval and detection.

---

## Environment

| Item | Recommended setting |
| --- | --- |
| Python | Python 3.10 |
| PyTorch | `torch>=2.3.0`, as specified in `requirements.txt` |
| GPU | CUDA-capable GPU with a compatible PyTorch build |
| Execution | Single GPU; the training wrapper uses `torchrun --nproc_per_node 1` |

Core dependencies are:

```text
torch>=2.3.0
torchvision
timm
numpy
Pillow
scikit-learn
tensorboard
```

The shell wrappers require Bash. No specific GPU model or fixed VRAM capacity is required by the repository.

---

## Installation

```bash
git clone https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector.git
cd Dual-Domain-Few-Shot-AIGI-Detector

python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the default PyTorch package does not match your CUDA runtime, install the appropriate CUDA-enabled PyTorch build for your system before installing the remaining requirements.

---

## Checkpoints

Pretrained checkpoints can be downloaded from:

- [Baidu Netdisk](https://pan.baidu.com/s/1YmL59_hJIIguExwbGzEbQA?pwd=vuuy)

After extracting the downloaded ZIP archive, the package has the following structure:

```text
DDFSD_pretrained/
|-- exclude_ADM/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- exclude_BigGAN/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- exclude_glide/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- exclude_Midjourney/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- exclude_SD/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- exclude_VQDM/
|   |-- ddfsd_step[15000].pth
|   `-- freq_stats.pt
|-- MANIFEST.csv
|-- SHA256SUMS.txt
`-- README.txt
```

Each `exclude_<CLASS>` directory represents one leave-one-generator-out task. Its `ddfsd_step[15000].pth` file is the main-experiment checkpoint, and its `freq_stats.pt` file contains the frequency-domain mean and standard deviation computed from that task's training split. These two files must always be used as a matched pair from the same directory; never combine a checkpoint from one held-out class with frequency statistics from another. Formal evaluation must not recompute frequency statistics from validation or test data.

`MANIFEST.csv` records artifact provenance and verification information, `SHA256SUMS.txt` provides integrity checksums, and `README.txt` contains concise instructions for using the package.

---

## Data Preparation

Set the dataset root before running an experiment:

```bash
DATA_ROOT=/path/to/GenImage
```

`DATA_ROOT` must be the directory that directly contains `real`, `ADM`, `BigGAN`, `glide`, `Midjourney`, `SD`, and `VQDM`.

For a training task, the training script computes the frequency-domain mean and standard deviation from the training split of `real` and the five non-excluded fake classes, then saves them as `freq_stats.pt`. Formal evaluation must load the statistics produced by the corresponding training task; it must not estimate them again from validation or test images.

---

## Running DDFSD

### 1. Train One Leave-One-Generator-Out Task

```bash
DATA_ROOT=/path/to/GenImage \
EXCLUDE_CLASS=ADM \
bash scripts/train_main.sh
```

`EXCLUDE_CLASS` supports `ADM`, `BigGAN`, `glide`, `Midjourney`, `SD`, and `VQDM`. Training refuses to reuse an output directory that already contains checkpoints.

### 2. Evaluate One Task

```bash
DATA_ROOT=/path/to/GenImage \
EXCLUDE_CLASS=ADM \
bash scripts/eval_main.sh
```

By default, evaluation loads the step-15000 checkpoint and its matching `freq_stats.pt`. It performs 10-shot evaluation with seeds `42, 101, 102, 103, 104`, uses the complete remaining validation set as queries, and reports ACC, AP, and AUC. The published main protocol uses both branches with support-dependent adaptive fusion.

Set `RUN_ROOT`, `RUN_DIR`, `CKPT_PATH`, `FREQ_STATS_PATH`, or `OUTPUT_DIR` explicitly when the files are stored outside the default run layout.

#### Evaluate with Downloaded Checkpoints

```bash
DATA_ROOT=/path/to/GenImage \
EXCLUDE_CLASS=ADM \
CKPT_PATH='/path/to/DDFSD_pretrained/exclude_ADM/ddfsd_step[15000].pth' \
FREQ_STATS_PATH='/path/to/DDFSD_pretrained/exclude_ADM/freq_stats.pt' \
OUTPUT_DIR='/path/to/results/exclude_ADM' \
bash scripts/eval_main.sh
```

Replace `ADM` with the desired held-out generator and use the checkpoint and frequency statistics from the same `exclude_<CLASS>` directory. The quoted checkpoint path is required because the filename contains square brackets.

### 3. Run All Six Leave-One-Out Tasks

```bash
MODE=all \
DATA_ROOT=/path/to/GenImage \
bash scripts/run_main_experiment.sh
```

The driver supports:

- `MODE=train` for training only
- `MODE=eval` for evaluation only
- `MODE=all` for training followed by evaluation

The six tasks run in the order `ADM`, `BigGAN`, `glide`, `Midjourney`, `SD`, and `VQDM`. The driver stops if any task fails.

---

## Main Configuration

| Item | Main setting |
| --- | --- |
| Data | Full GenImage |
| Protocol | Six-class leave-one-generator-out |
| Training | 15,000 steps; batch size 16; single GPU |
| Episode | 3-way: real + 2 fake classes |
| Episode samples | 5 support + 5 query images per class during training |
| RGB encoder | ResNet-50 |
| Frequency encoder | ResNet-18 |
| Embedding dimension | 512 |
| Frequency input | Signed Haar-DWT `LH`, `HL`, and `HH` bands from the Y channel |
| Scheduler | StepLR; step size 5,000; gamma 0.5 |
| Learning rates | Backbone `3e-5`; projection head `1e-4` for both domains |
| Prototype margins | `m_rf=1.2`; `m_ff=0.6`; `lambda_ff=0.5` |
| Separation weight | Target `lambda_sep=0.03`; linear warmup from step 2,500 to 7,500 |
| Branch dropout | Dual/RGB-only/frequency-only: `0.90 / 0.05 / 0.05` |
| Formal evaluation | Step 15,000; 10-shot; 5 seeds; full validation query set |
| Metrics | ACC / AP / AUC |

This table describes the runtime configuration, not experimental results.

---

## Output Structure

With the default experiment name, outputs are written outside the repository in the following layout:

```text
<run_root>/
`-- main_full_steps15000/
    `-- exclude_<class>/
        |-- ckpt/
        |   `-- ddfsd_step[15000].pth
        |-- tb/
        |-- logs/
        |   |-- train_driver.log
        |   |-- eval_driver.log
        |   `-- <timestamp>_log.txt
        |-- train.log
        |-- freq_stats.pt
        `-- formal_eval_step15000/
            |-- ddfsd_eval_per_seed.csv
            |-- ddfsd_eval_summary.csv
            |-- formal_val_invalid_images.csv
            |-- config.json
            |-- logs/
            |   `-- <timestamp>_log.txt
            `-- eval.log
```

The driver logs are created by `run_main_experiment.sh`; timestamped logger files are created by the Python entry points.

---

## Evaluation

Formal evaluation writes per-seed metrics to `ddfsd_eval_per_seed.csv`, aggregate mean and standard-deviation metrics to `ddfsd_eval_summary.csv`, the resolved protocol and paths to `config.json`, and console output to `eval.log`. It also audits the exact validation images before inference and records the audit in `formal_val_invalid_images.csv`.

TensorBoard events are stored under each task's `tb/` directory. Launch TensorBoard with a generic run root:

```bash
tensorboard --logdir /path/to/run_root/main_full_steps15000 --host 0.0.0.0 --port 6006
```

---

## Acknowledgements

We thank the authors of FSD (Few-Shot Learner Generalizes Across AI-Generated Image Detection) for releasing their work and code, which provided the foundation for this project.
