# R2-1 支持集可靠性分析：本机准备 / AutoDL 执行

本次实现只交付代码、合成测试和运行说明。**GPU_VALIDATION_PENDING_AUTODL；EXPERIMENT_RESULTS=NOT_RUN_LOCALLY**。
不下载数据/模型，不训练，不改变模型、损失、原正式测试定义，也不自动提交或上传 AutoDL 结果。
唯一代码仓库：<https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector>。

## 1. 在 AutoDL 拉取同一提交

下面全部命令在 **AutoDL 现有环境**中运行。本机 Codex 不执行 GPU 实验，也不远程操作 AutoDL。

已有克隆：

```bash
cd /path/to/Dual-Domain-Few-Shot-AIGI-Detector
git remote -v
git pull --ff-only origin main
git rev-parse HEAD
git status --short
```

没有克隆：

```bash
git clone https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector.git
cd Dual-Domain-Few-Shot-AIGI-Detector
git rev-parse HEAD
```

使用本机交付信息中的实际 commit 核对 `HEAD`。若交付的是实验分支/PR，先拉取并切换该分支；未合并 PR 不代表 main 已更新。
执行程序要求已跟踪文件与 HEAD 一致，以便身份中 Git SHA 和源码 SHA 对应；运行结果必须放在仓库外。

## 2. 只配置三个真实目录

所有 `/path/to/...` 都是占位符，需要换成当前 AutoDL 的真实目录，代码不硬编码服务器或本机路径。

```bash
export DATA_ROOT='/path/to/GenImage'
export CKPT_ROOT='/path/to/DDFSD_pretrained'
export OUTPUT_DIR='/root/autodl-tmp/R21_Reliability_Analysis'

ls "$DATA_ROOT"
ls "$CKPT_ROOT"
find "$DATA_ROOT" -maxdepth 2 -type d -name val
find "$CKPT_ROOT" -maxdepth 4 -type f \( -name 'ddfsd_step[[]15000[]].pth' -o -name freq_stats.pt \)
```

`DATA_ROOT` 直接包含 `real/ ADM/ BigGAN/ glide/ Midjourney/ SD/ VQDM/`，每类有 `val/`。
SD 使用现有合并类；所有图像路径、排序、Resize(256)/CenterCrop(224)/ToTensor、严格解码均复用原数据集。
`CKPT_ROOT` 直接包含六个任务目录，默认布局如下：

```text
DDFSD_pretrained/
  exclude_ADM/ddfsd_step[15000].pth      + freq_stats.pt
  exclude_BigGAN/ddfsd_step[15000].pth   + freq_stats.pt
  exclude_glide/ddfsd_step[15000].pth    + freq_stats.pt
  exclude_Midjourney/ddfsd_step[15000].pth + freq_stats.pt
  exclude_SD/ddfsd_step[15000].pth       + freq_stats.pt
  exclude_VQDM/ddfsd_step[15000].pth     + freq_stats.pt
```

如果解压后多嵌套一层，把 `CKPT_ROOT` 指向**直接含六个 exclude_* 的那层**。
如果是训练输出布局 `exclude_ADM/ckpt/ddfsd_step[15000].pth`，stats 仍在 `exclude_ADM/freq_stats.pt`，设置：

```bash
export CHECKPOINT_TEMPLATE='exclude_{generator}/ckpt/ddfsd_step[15000].pth'
```

Python 等价参数：`--checkpoint-template 'exclude_{generator}/ckpt/ddfsd_step[15000].pth'`。
另有 `--stats-template` / `STATS_TEMPLATE`，默认 `exclude_{generator}/freq_stats.pt`。
模板是精确路径，不是 glob；`[15000]` 为字面文件名。二者必须解析到同一 `exclude_<generator>` 任务树，不能跨任务引用或自动寻找相似历史权重。
程序检查 checkpoint 元数据中的 step=15000、exclude_class 与目录一致、配置不冲突、严格 state_dict 加载和有限值；stats mean/std 必须为 [3] 且有限，std 非负。
model_mode 兼容正式入口支持的历史双分支权重：分别检查顶层、config 和 args（支持字典或 Namespace），所有显式 model_mode 都必须为 dual；即使顶层为 dual，嵌套单分支声明或 config/args 冲突仍拒绝。显式 None、空串等无效值也拒绝，只有字段缺失才采用历史兼容路径。
无论是否有模式声明，都实例化 `DDFSDDualDomainNet(pretrained=False, model_mode='dual')`，按该模型的完整 state_dict 核对 RGB/Frequency 的 backbone/projector 四组键，再以 `strict=True` 加载全部参数和 buffer；缺分支、缺个别键、维度不匹配或多余键均拒绝，不凭默认字符串判断架构正确。
`logs/preflight_validation.json` 与 `run_manifest.json` 的 checkpoints 项记录 `model_mode_confirmation`：有显式声明为 `explicit_dual`，完全缺失声明但结构确认通过为 `legacy_dual_state_dict`；`model_mode_evidence` 保留声明来源、四组完整键数与 strict_load=true。此依据只有全部加载及有限值验证成功后才写为通过。
先加载 checkpoint，再应用选定 stats，顺序与正式入口相同。历史记录的旧 stats 路径仅保留作审计，不据此加载其它文件。
历史兼容只允许缺失 model_mode；缺少 step/exclude_class 等原有必要身份或未通过其他检查仍失败，不改写 checkpoint、不靠文件名推测其身份。
目录配对和 SHA 能防止续跑混用；需要验证文件来源时应使用已知可信 checksum manifest，程序不能仅凭数值证明 stats 的训练来源。

## 3. 使用原有镜像检查依赖

```bash
nvidia-smi
python -c "import torch, torchvision, timm, numpy, PIL, sklearn; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import matplotlib; print(matplotlib.__version__)"
```

已有正式评估环境只增加绘图依赖；缺少时在 **AutoDL** 执行：

```bash
python -m pip install -r requirements-r21.txt
```

原环境其余依赖见 `requirements.txt`。若缺小依赖，按报错安装缺失项，例如 `python -m pip install numpy Pillow scikit-learn matplotlib`。
若 torch/torchvision/CUDA 版本不兼容，先恢复该镜像已有的兼容环境；不要在本机安装 PyTorch/CUDA 试跑，也不要让本流程下载 pretrained 参数。
模型始终 `pretrained=False`，全部参数由已存在的最终 checkpoint 加载。
设备与 `test_ddfsd.py` 相同：CUDA 可用时用当前 CUDA 设备，否则 CPU；FP16 仅 CUDA 上生效。实际设备、驱动相关运行库版本与精度写入身份。
如需选卡，在启动前设置 `CUDA_VISIBLE_DEVICES`，续跑保持一致。

## 4. 一条命令启动全部 30 组

从仓库根目录执行；将路径替换为真实值：

```bash
DATA_ROOT='/path/to/GenImage' \
CKPT_ROOT='/path/to/DDFSD_pretrained' \
OUTPUT_DIR='/root/autodl-tmp/R21_Reliability_Analysis' \
bash scripts/run_r21_reliability.sh
```

已经 export 三个目录时只需 `bash scripts/run_r21_reliability.sh`。
无须逐项确认。支持 `PYTHON`、`SEEDS`、`SUPPORT_SHOT`、`BATCH_SIZE`、`NUM_WORKERS`、`USE_FP16`、`CHECKPOINT_TEMPLATE`、`STATS_TEMPLATE` 环境变量。
脚本从任意工作目录都能调用，Shell 参数也会传给 Python，例如 `bash scripts/run_r21_reliability.sh --validate-only`。

完全独立的 Python 等价入口（默认目录布局）：

```bash
python -u tools/run_r21_reliability.py \
  --data-root '/path/to/GenImage' \
  --ckpt-root '/path/to/DDFSD_pretrained' \
  --output-dir '/root/autodl-tmp/R21_Reliability_Analysis' \
  --seeds '42,101,102,103,104' \
  --support-shot 10 \
  --all-queries --resume --package
```

默认固定六个生成器，五 seeds=[42,101,102,103,104]，10-shot，tau=.2，tau_r=.1，clip=[.25,.75]，dual，FP16=true，batch=128，workers=8。
每类 Support 调用原 `sample_support_query_indices`，同 seed；剩余全部 val 作为 Query。不提供 max_query 截断选项。
`--tau`、`--tau-r`、`--support-shot`、`--seeds`、`--use-fp16` 可覆盖用于明确标记的自定义协议；不会冒充默认完整实验。
改动 seed 顺序也会改变身份。`--batch-size` / `--num-workers` 不改变 Query 规模，但改变后须使用新输出目录。
本次不实施 Fixed/Inverse Alpha、R2-6 或重新训练。

预检会一次检查七类全部 val 与六组 checkpoint/stats，严格解码每张图，并读取文件内容计算 SHA-256。
这可能耗时较长，但任何一项缺失/损坏都会列入 `logs/preflight_validation.json` / `logs/formal_val_invalid_images.csv` 并在正式推理前停止，不会丢类或替换损坏图像。
Query 推理按生成器顺序运行，日志显示当前任务与处理进度；每任务只编码一次，保存 RGB/Frequency 距离，三模式分数和预测离线可复算。
不使用跨任务 embedding 缓存，避免复用错误；同生成器模型可在五 seeds 间复用。

### 自动 parity：额外一遍正式验证

每个批次先严格核查保存的融合距离所重建 logits 与现有 `compute_query_logits(..., branch_mode='dual')` 一致。
完成一个生成器的五个诊断任务后，自动调用未经修改的 `test_ddfsd.py`，同模型、stats、预处理、batch、精度、Support seed、全部 Query，一次跑该生成器五 seeds。
这是独立正式入口核查，会额外进行一遍 Dual 推理和正式图像审计；不会分别为 RGB-only、Frequency-only 再推理三遍。
比较每 seed ACC、Real/Fake ACC、Balanced ACC、AP、AUC，绝对容差 1e-8；不同运行环境的差异会保留并阻止完整通过，不通过放宽容差来迎合历史结果。
参考日志：`formal/<generator>/reference.log`；原入口完整 CSV/配置及校验标记也保留在该目录。
六类五 seeds 仍是 **30 组实验**，额外 reference 是对这 30 组同输入结果的验证。

## 5. 预检、续跑与离线分析

完整预检但不做模型 forward：

```bash
python tools/run_r21_reliability.py --validate-only
```

以上短命令依赖已经 export 的 DATA_ROOT / CKPT_ROOT / OUTPUT_DIR；也可以显式传三个 `--*-root/dir` 参数。
`--validate-only` 仍需已有深度学习环境来检查六权重能否严格加载；它不是本机无依赖测试入口。
验证通过会写资源/运行 manifest，但不会生成伪实验结果。

正常中断后，用**完全相同的原命令**续跑。`--resume` 默认开启；无须手动枚举未完成任务。
任务身份包含 generator、seed、Git commit 与源码 hash、checkpoint/stats SHA、内容哈希数据清单、路径、协议、batch、workers、精度、设备与依赖版本。
只有身份完全匹配、所有文件 checksum 正确且逐图/Support/指标复算通过才复用；损坏或中断的同身份任务重跑。
身份不同会停止并保留旧结果，必须选新 OUTPUT_DIR；**拉取修复代码后 commit 改变，也用新目录**。
如果之前仅在 checkpoint 预检处失败（PARTIAL_0_OF_30），且尚无 `run_manifest.json`，可在拉取修复后沿用原输出目录；已生成 manifest 时仍遵守新目录规则，不手改旧身份文件。
数据集和权重在一次运行期间应保持只读，不在预检与推理期间替换文件。

写文件采用同目录临时文件+原子替换，任务通过完整语义复算之后才原子写 `per_task/<key>/DONE.json`。
根目录 `run_status.json` 在各任务后更新。进程互斥锁 `.r21.lock` 防止两个主入口同时写同目录。
SIGKILL/机器掉电可能残留锁：先 `cat "$OUTPUT_DIR/.r21.lock"` 并用 `ps -p <pid>` 核对同机进程；确认已结束后才 `rm "$OUTPUT_DIR/.r21.lock"` 并原命令续跑。
不要删除有效任务目录或自动接受身份不一致结果。

无需模型或原数据重新推理，只从已有原始输出复算：

```bash
python tools/run_r21_reliability.py --output-dir "$OUTPUT_DIR" --analyze-only --package
# 独立分析入口，需 numpy + matplotlib，不导入 torch：
python tools/analyze_r21_reliability.py --output-dir "$OUTPUT_DIR" --package
```

分析可在部分任务完成后运行；报告/图表必须显示部分状态，退出码 2，不能据此宣称完整。
没有任何通过验证的任务时不生成结果报告。
分析入口应在主运行停止后使用，避免并发改写汇总。主入口已带互斥锁。
出错时保留 `logs/last_traceback.txt`、`logs/last_failure.json`、运行 manifest 和正式日志供本机修复；不让 AutoDL 操作者现场改实验逻辑。

### 可选可信校验清单

`--checksum-manifest /path/to/checksums.json` 接受以下 JSON 结构（必须填**全部六类 checkpoint/stats、七类数据 hash**，不能用示例字符串）：

```json
{
  "checkpoints": {
    "ADM": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"},
    "BigGAN": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"},
    "glide": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"},
    "Midjourney": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"},
    "SD": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"},
    "VQDM": {"ckpt_sha256": "<sha256>", "stats_sha256": "<sha256>"}
  },
  "datasets": {
    "real": "<inventory sha256>", "ADM": "<inventory sha256>",
    "BigGAN": "<inventory sha256>", "glide": "<inventory sha256>",
    "Midjourney": "<inventory sha256>", "SD": "<inventory sha256>", "VQDM": "<inventory sha256>"
  }
}
```

数据 hash 是官方排序后的 `[{index, relative_path, bytes, sha256}, ...]`，按 `r21_common.canonical_hash` 的排序键/紧凑 UTF-8 JSON 计算。
预检输出和 run_manifest 都保留此值。没有外部可信清单时仍全量计算并记录 SHA，供后续核查。

## 6. 输出、定义与验收

```text
OUTPUT_DIR/
  run_manifest.json                 # 所有身份、环境、配置、原命令
  run_status.json
  logs/                             # 全量资源检查、严格图像审计、异常
  manifests/<class>.json.gz          # 七类有序内容清单
  per_query/<generator>_seed<seed>.npz # 30 份全量 Query 无损数组
  per_task/<key>/                    # Support 特征/原型、抽样清单、指标、DONE
  formal/<generator>/                # 原 test_ddfsd.py 证据与校验
  episode_class_metrics.csv          # 默认 60 类别行
  task_metrics.csv                   # 默认 30 任务，三模式指标各列
  generator_summary.csv
  distribution_summary.csv           # sigma/alpha/分离度 min/Q25/median/Q75/max
  correlation_results.csv
  mismatched_cases.csv
  parity_validation.csv
  completion_validation.json
  figures/alpha_margin_scatter.png
  figures/generator_comparison.png
  figures/alpha_clipping.png
  R21_REPORT.md                      # 中文报告及英文回信证据摘要
```

NPZ 使用 `allow_pickle=False` 可加载：

- `d_rgb` / `d_freq` / `d_dual`：float32 [N,2]，列 0=Real、1=Fake，分别对应六个要求的 d_rgb_real/.../d_fused_fake。`d_dual` 即融合距离。
- `score_rgb/freq/dual`：softmax(-distance/tau) 的 fake 概率；`pred_rgb/freq/dual`：该 softmax 的 argmax（平局选 0）。
- `labels` (Real=0/Fake=1)、`source_index`、`relative_path`：逐图身份；`generator`、`seed`、`identity_sha256`：任务身份。
- `per_task/<key>/support.npz`：support_rgb/freq [shot,2,D]、proto_rgb/freq [2,D]；support.json 记录原函数 sigma、alpha、裁剪前 raw 和两域原型分离度。
- `sampling.json.gz`：两个类别所有 Support/Query 下标与相对路径，按正式采样顺序保存；大清单不塞入主 CSV。

对每个类别，仅在该真实标签的 Query 上计算 margin：Real 为 d_fake−d_real，Fake 为 d_real−d_fake。
X=alpha_rgb_i−.5；Y=mean_margin_rgb_i−mean_margin_freq_i。正 margin 表示相对支持真类；严格负表示错类；零是距离平局，实际 argmax 仍按正式定义。
不能平均两类 alpha 后和全局 ACC 相关，也不能将每张 Query 当成独立 alpha 样本。
类别行同时保留该类三模式 Recall、delta_recall；三模式 ACC/AP/AUC 使用全量 Query。
推理指标由原 `binary_metrics` 产生；离线纯 Python 实现按平均秩 AUC、同分数阈值 AP 复算并核对，不修改原评估。

统计：Real 30 / Fake 30 / Overall 60 行 Spearman（平均秩）、非平局方向一致率与覆盖率、每生成器对照；Overall 明确为描述性。
X=0 或 Y=0 精确平局单列，常数相关为 undefined/NaN；不人为置零。
生成器 cluster bootstrap 默认 10,000 次、seed=2101，整簇重采样并保留所有 seed/类别，给出百分位 95% CI 和有效次数；不把 Query 重叠当独立显著性证据，不报告独立性 p 值。
单生成器只有一个 cluster，因此不提供其 generator-cluster CI；六簇总数也限制区间稳定性。
`--bootstrap-repeats` / `--bootstrap-seed` 仅影响汇总，可重新分析，不影响已保存推理任务。

反例公开规则：全部 X*Y<0 都保留；每域额外筛选 sigma≤全体类别行 Q25 且 proto_sep≤Q25，且 mean_margin≤0 或 Recall≤.5。
分位数线性插值，精确阈值写入报告/验证 JSON；无满足记录则如实为空。
论文参考 mACC=91.10%、mAP=92.21%、mAUC=95.61% 仅用于显示实际差值，不作为硬编码输出、完成条件或调整结果的目标。

**唯一默认完整通过状态：`COMPLETE_AND_MATCHING (30/30)`**，且 completion_validation 中 class_rows=60、parity_matching=true。
它要求每组无 Support/Query 交集，官方索引与路径完全匹配，所有 Query 数量/维度/标签/有限值/距离融合/Support 原型及 sigma/alpha/三模式指标均复算通过，所有 CSV 与原始记录一致，并与正式入口逐 seed 匹配。
中断/缺失为 `PARTIAL_n_OF_30`；全任务完成但 reference 缺失/不符为 `PARITY_PENDING_OR_MISMATCH`。
参数改动后的完整状态为 `CUSTOM_COMPLETE_AND_MATCHING`，不能称本次要求的默认完整结果。
预检 `PASS`、单任务 `DONE`、代码测试 `PASS` 均不等于真实 GPU 全实验完成。

## 7. 下载小归档用于复核

Shell 默认附带 `--package`；Python 可加 `--package`，或用上面的 analyze-only 命令生成：

```text
OUTPUT_DIR 的同级/<输出目录名>_review.tar.gz
```

归档只包含小体积 CSV、PNG、报告、配置、checksum、验证、Support 摘要与正式参考日志；排除所有 Query/Support NPZ、数据集、权重、逐图/抽样大清单。
`review_checksums.json` 可验证归档内容。此归档便于 ChatGPT 检查汇总，但完整逐图重算仍需原输出目录的 NPZ/manifest，必须保留在 AutoDL。
程序不自动上传归档，不在 AutoDL git commit/push。下载后由用户决定是否提供给复核者。

## 8. 本机可执行验证（无 GPU / 数据 / checkpoint）

```bash
python -m unittest discover -s tests -p test_r21_reliability.py -v
python -m compileall -q tools tests/test_r21_reliability.py
git diff --check
bash -n scripts/run_r21_reliability.sh
```

标准 Python 无需 numpy/torch 可运行基础测试；存在 numpy 时附加执行小型合成 NPZ、真实 loss 函数 AST 的 NumPy 适配器、30 任务／60 行复算/归档/缺失/损坏/NaN 测试。
适配器只验证公式与形状，不是 PyTorch、CUDA 或正式推理验证。测试夹具标记 SYNTHETIC，仅在临时目录生成并清理。
本机没有的检查标记 NOT_RUN_ENV_MISSING，不为了测试安装 PyTorch/CUDA/timm 或搜索真实数据/权重。
真实 GPU、原环境兼容性、原正式入口数值 parity、三张真实图表的渲染与论文参考差值，全部需要 AutoDL 实际运行验收。

### 历史 checkpoint 模式兼容修复：本机检查记录

- 标准 Python 3.10、无 numpy/torch：22 项测试通过，4 项 NumPy 合成测试标记 NOT_RUN_ENV_MISSING。
- 本机已有 NumPy 运行时：26 项测试全部通过，包含新增 8 项 checkpoint 兼容测试以及原有 R2-1 合成复算测试。
- 新增测试覆盖缺少模式字段的历史双分支、各层显式 dual、显式单分支/无效值、顶层与 config/args 冲突、整支/部分参数缺失、严格加载形状/键错误、无效统计和非有限参数，以及报告确认依据。
- Python compileall、Bash 语法、Shell 实际参数传递及 git diff --check 均通过。
- checkpoint 测试使用标准库模型/张量替身，验证完整键集检查、strict=True 调用及 stats 应用顺序；不等同真实 PyTorch 加载。未安装 PyTorch/CUDA，未修改或加载正式 checkpoint，未执行 GPU 实验。
- 真实模型严格加载与正式实验仍为 GPU_VALIDATION_PENDING_AUTODL；拉取 main 后可在 AutoDL 现有环境运行原入口，无须改 checkpoint 或现场修改逻辑。
