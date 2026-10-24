# R2-4 支持集质量敏感性实验

本次交付为代码和合成验证，真实 GPU 实验 **PENDING_AUTODL**。不下载 GenImage，不在本机安装 PyTorch/CUDA、不下载 CLIP、不重新训练，也不生成论文实验数值。
目标仓库是 <https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector>，独立实验分支 `exp/r2-4-support-quality`；不合并公开 main。

## 1. 固定协议与两个条件

六个 held-out ID：ADM、BigGAN、**glide**、Midjourney、SD、VQDM。SD 保持项目已合并类别。
每生成器使用匹配的 step15000 Dual checkpoint 和该任务 freq_stats.pt，冻结模型、eval/no_grad，10 Real + 10 Fake Support。
seeds 固定默认 42,101,102,103,104；每 seed 每条件只建立一次原型和 class-specific adaptive alpha，评测该条件的全部剩余 val。
tau=.2、tau_r=.1、alpha clip=[.25,.75]，原始 DDFSD Resize256/CenterCrop224、RGB 和 Y/Haar 高频、频域标准化、fake=1、softmax(-distance/tau)、binary_metrics 全部复用。
FP16 默认 true，只在 CUDA autocast 生效；batch 默认128、workers8。每类至少11张 val，无 Query 截断参数。

**A Label Noise**：先原生 sample_support_query_indices，分别抽10张 Real/Fake。三档 p=.1/.2/.3，对每来源交换 k=round(10p)=1/2/3 张的原型归属。
每个观测类仍10张，误标总量2/4/6，原20张来源集合不变。像素、Query 索引/顺序、Query 真值完全不变。
独立 SHA-256 派生的 generator+seed 随机日程对两类支持索引各作排列；各档取长度 k 前缀保证嵌套，不使用 Python hash()，也不消耗正式抽样的 RNG。
交换后 Real 原型输入=(10-k)个原 Real+k个 Fake，Fake 原型反之，确实改变模型输入，不只改日志。

**B CLIP-Concentrated**：固定冻结的公开 CLIP image encoder，默认 openai/clip-vit-base-patch32，仅用官方 CLIP processor 生成选择特征。
每个 generator+seed 从 real/val 稳定随机选一个真实锚点；同一锚点对全部 Real 与当前 Fake 各选 cosine Top10。真实来源标签完全正确，不基于 DDFSD 预测选图。
按相似度降序、官方图片索引升序打破平局。Real 锚点强制属于支持集：极端全同向量边界平局时保留锚点、替换最后一个非锚点，其余候选仍按索引排序。此规则固定，不根据性能重选。
两类各自 Query 为去掉本条件 Support 后的所有 val，按官方路径索引升序。CLIP 组和 Random 组移走的图片不同，Query 可不同；不构建共同 Query 池。
CLIP tensor 不送给 DDFSD，DDFSD 仍重新读取原图、使用其自己的正式预处理与两域编码。

默认全部运行：90 个 Label Noise generator-seed-rate 条件 + 30 个 CLIP bias generator-seed 条件，共120次完整 Query 评测。
这不是几千个独立 episode。每生成器/条件先按5 seeds求 mean±population std（statistics.pstdev，ddof=0），再宏平均六个生成器均值。
宏 std 明确定义为六个生成器均值的 population std，与生成器内部 seed std 分列。CSV 保存原精度，仅报告显示百分比两位小数。

原论文 Clean Baseline 不重新跑。`--noise-rates 0` 仅用于显式 clean parity 验证，条件名 `clean_parity_only`，不作为新增干净基线证据。
报告可引用已有正式 per-seed CSV，绝不预填原论文数字或预测本实验表现。

## 2. AutoDL Cursor 拉取分支

已有 **zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector** 克隆，在工作区没有未提交更改时：

```bash
cd /path/to/Dual-Domain-Few-Shot-AIGI-Detector
git remote -v
git status -sb
git fetch origin
git switch --track origin/exp/r2-4-support-quality
```

若分支已经存在，则用：

```bash
git switch exp/r2-4-support-quality
git pull --ff-only origin exp/r2-4-support-quality
git rev-parse HEAD
```

独立新克隆（避免修改其它仓库的 remote 或已有工作）：

```bash
git clone --branch exp/r2-4-support-quality https://github.com/zenghao0718/Dual-Domain-Few-Shot-AIGI-Detector.git Dual-Domain-Few-Shot-AIGI-Detector-r24
cd Dual-Domain-Few-Shot-AIGI-Detector-r24
```

新入口禁止写 bytecode，测试也使用 -B，避免运行生成文件干扰版本身份。执行要求已跟踪代码与 commit 一致；不要 reset 或覆盖用户已有修改。

## 3. 环境和外部目录

以下均在 AutoDL 已有 DDFSD 环境执行，所有路径是需要替换的占位符：

```bash
nvidia-smi
python -c "import torch, torchvision, timm, numpy, PIL, sklearn; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements-r24.txt

export DATA_ROOT='/path/to/GenImage'
export CKPT_ROOT='/path/to/final_training_tasks'
export CLIP_MODEL_PATH='/path/to/clip-vit-base-patch32'
export CLIP_CACHE_ROOT='/path/to/r24_clip_cache'
export OUTPUT_DIR='/path/to/R24_support_quality'

ls "$DATA_ROOT"
ls "$CKPT_ROOT"
find "$CKPT_ROOT" -maxdepth 4 -type f \( -name 'ddfsd_step[[]15000[]].pth' -o -name freq_stats.pt \)
```

DATA_ROOT 下直接有 real/val、ADM/val、BigGAN/val、glide/val、Midjourney/val、SD/val、VQDM/val。
默认 CKPT_ROOT 下每类：`exclude_ADM/ckpt/ddfsd_step[15000].pth` 和 `exclude_ADM/freq_stats.pt`，对应原 scripts/eval_main.sh。
若解压文件是 `exclude_ADM/ddfsd_step[15000].pth`，设置：

```bash
export CHECKPOINT_TEMPLATE='exclude_{generator}/ddfsd_step[15000].pth'
```

支持参数 `--checkpoint-template`、`--stats-template`，按精确字符串解析，不能自动挑相似历史 checkpoint。
检查顶层及 config/args 的 step/目标类别/dual 声明；历史 model_mode 缺失可兼容，但必须四组 RGB/Frequency backbone/projector 完整并 strict=True 加载双分支全部参数和 buffer。
显式非 dual、模式/类别冲突、step 非15000、缺参数、无效stats或非有限模型一律停止。stats 必须位于同任务目录，允许 checkpoint 在子目录 ckpt/；不能引用其它任务 stats。
checkpoint/stats/data/cache/output/model 文件全部放仓库外。输出与数据、权重根目录、缓存根目录不得重叠，不自动 Git 提交/上传结果。

## 4. 事先准备 CLIP 权重，之后离线编码

已有本地目录时直接用 `--clip-model-path "$CLIP_MODEL_PATH"`。没有时，在允许联网的环境事先准备一次（本机 Codex 不执行）：

```bash
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/clip-vit-base-patch32', local_dir=os.environ['CLIP_MODEL_PATH'], allow_patterns=['*.json','*.safetensors','*.bin'])"
```

可在下载调用中显式传完整 `revision='<真实Hub commit SHA>'` 固定来源，不在文档编造 revision。
也可把预先准备的完整目录复制到 AutoDL。模型目录须含配置、processor 和实际权重，不只 config。
缓存入口始终 local_files_only=True，不隐式联网，不让随机初始化的缺失权重成为选择器；会检查加载信息中 missing/mismatched/error，并保持 eval/无梯度。
采用 Transformers 4.x 的 CLIPVisionModelWithProjection.image_embeds，float32编码后 L2 归一化、float32存储，eager attention，不使用 CLIP 文本分支进行分类。
如果现有 torch 与 Transformers 的权重格式检查不兼容（例如旧 torch 读 bin 被库拒绝），保留链式错误；准备兼容的 safetensors 本地目录，或在已有兼容 CLIP 环境先生成缓存，再回原 DDFSD 环境评估，不修改 DDFSD 模型环境来绕过检查。
缓存记录实际 Hub snapshot revision；纯本地目录无法恢复 Hub commit 时明确标记 local-content-sha256，并保存配置/权重的内容 SHA，不伪造 revision。

先准备全部7类，Real只编码一次：

```bash
python -B -u tools/build_r24_clip_cache.py \
  --data-root "$DATA_ROOT" --cache-root "$CLIP_CACHE_ROOT" \
  --clip-model-path "$CLIP_MODEL_PATH" \
  --classes all --batch-size 128 --num-workers 8 --device auto --resume
```

也支持 `--classes real,ADM` 的单任务准备；之后同命令改成 all 会校验并跳过已完成类，其余继续编码，不重算六次Real。
支持 `--clip-model-id`（默认标准模型）、`--clip-revision`、`--skip-existing`（resume别名）。若只使用本机已有 Hugging Face 缓存可省略 model-path。
模型/预处理、权重SHA、请求/解析revision、precision、batch、设备、依赖版本、Git SHA、官方排序的相对路径和每张内容 SHA 一起绑定缓存；不同图片内容/顺序/数量、模型/processor不一致会明确拒绝，不静默混用。
中断未有 DONE 的类别会重算；有效 DONE 的类别在完整校验后跳过。损坏 DONE 不默默当成功，需检查或新建缓存目录。
更新代码 commit 或改变缓存编码精度/设备/batch后，使用新缓存目录；不要编辑metadata假装身份一致。

## 5. ADM 功能检查（小范围，不截断 Query）

先选单个生成器、一个 seed 做全 Query smoke。此处明确是部分范围，不宣称六类五seed完成：

```bash
python -B -u test_ddfsd_r24.py \
  --experiment all --exclude-class ADM --seeds 42 \
  --data-root "$DATA_ROOT" --ckpt-root "$CKPT_ROOT" \
  --clip-cache-root "$CLIP_CACHE_ROOT" \
  --output-dir '/path/to/R24_smoke_ADM' --validate-only

python -B -u test_ddfsd_r24.py \
  --experiment all --exclude-class ADM --seeds 42 \
  --data-root "$DATA_ROOT" --ckpt-root "$CKPT_ROOT" \
  --clip-cache-root "$CLIP_CACHE_ROOT" \
  --output-dir '/path/to/R24_smoke_ADM'
```

validate-only 会严格审计所有所需原图、校验缓存并加载所需 checkpoint/stats，不执行 DDFSD forward。
若用直接根目录 checkpoint 布局，Python 命令同时加 `--checkpoint-template 'exclude_{generator}/ddfsd_step[15000].pth'`；Shell 会自动读取 CHECKPOINT_TEMPLATE 环境变量。
单个 checkpoint 也可用（不能与 exclude-class all 混用）：

```bash
python -B test_ddfsd_r24.py --experiment label_noise --exclude-class ADM \
  --data-root "$DATA_ROOT" \
  --ckpt-path '/path/to/exclude_ADM/ckpt/ddfsd_step[15000].pth' \
  --freq-stats-path '/path/to/exclude_ADM/freq_stats.pt' \
  --output-dir '/path/to/R24_ADM_noise'
```

noise-only 不需要 CLIP/cache依赖。额外显式 `--noise-rates 0 --seeds 42` 是可选 clean一致性检查，不会默认触发重跑干净基线。

## 6. 全六类 × 五 seeds × 两项实验

确认上面路径/缓存完整后，默认一次运行所有条件：

```bash
DATA_ROOT="$DATA_ROOT" CKPT_ROOT="$CKPT_ROOT" \
CLIP_CACHE_ROOT="$CLIP_CACHE_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
bash scripts/eval_r24.sh
```

Python 独立等价入口：

```bash
python -B -u test_ddfsd_r24.py \
  --experiment all --exclude-class all \
  --data-root "$DATA_ROOT" --ckpt-root "$CKPT_ROOT" \
  --clip-cache-root "$CLIP_CACHE_ROOT" --output-dir "$OUTPUT_DIR" \
  --seeds '42,101,102,103,104' --noise-rates '0.1,0.2,0.3' --resume
```

Shell 同时读取 EXCLUDE_CLASS、EXPERIMENT、SEEDS、NOISE_RATES、BATCH_SIZE、NUM_WORKERS、USE_FP16、CHECKPOINT_TEMPLATE、STATS_TEMPLATE、PYTHON。
单任务可通过环境 CKPT_PATH/FREQ_STATS_PATH 提供精确路径；全六类前应 unset 这些单任务变量，使用 CKPT_ROOT。
可分别设置 `--experiment label_noise` 或 `--experiment clip_bias`，输出到各自新目录。默认不会要求操作者确认120次。
所有条件各自的 Support/Query 完整记录，Query 数必须为各类 val 数减10，绝不按 CLIP 相似度或性能筛 Query。

## 7. 原论文基线引用

有已有正式 per-seed CSV 时，添加：

```bash
bash scripts/eval_r24.sh \
  --baseline-reference '原论文正式5-seed评测结果' \
  --baseline-csv '/path/to/existing_formal_per_seed_all_generators.csv'
```

CSV 允许 generator 或 exclude_class 列，须含 seed、acc、ap、auc（0..1）。每个所选生成器必须恰有42,101,102,103,104五行；若提供model_mode/branch_mode/support_shot/ckpt_step也校验正式协议。
单 ADM 可直接引用原 `ddfsd_eval_per_seed.csv`；六类运行可将已有六文件各保留一次表头合并后引用。不会为此运行任何 clean模型。
引用CSV的绝对路径和SHA保留，`baseline_reference_summary.csv` 标记为旧证据汇总，`baseline_comparison.csv` 给出本条件均值减已有 clean均值的差，不把旧结果写入本次 per_seed.csv。
没有提供数值CSV时，记录 NOT_PROVIDED，只输出真实条件指标，不伪造基线或“下降百分比”。

## 8. 输出、完成与续跑

```text
OUTPUT_DIR/r24/
  run_config.json                    # 协议、Git、全部资源/缓存身份、环境
  run_status.json
  logs/image_audit.json
  logs/last_failure.json              # 若有错误则保留完整 traceback 和命令
  dataset_manifests/<class>.json.gz   # 官方有序全路径/大小/内容SHA清单
  validation.json                    # 仅 validate-only
  per_task/<generator>/<experiment>/<seed_condition>/
    result.json.gz                   # 真指标、完整Support/Query索引、诊断、支持来源
    DONE.json                        # 身份、结果SHA；语义校验后最后原子写
  exclude_ADM/label_noise/
    per_seed.csv                     # 15行：5 seeds × 3 rates
    summary.csv                      # 3行，每档5seed mean/pstdev
    support_manifest.csv             # 每seed/条件20行，保留真实及观测标签
    config.json
  exclude_ADM/clip_bias/
    per_seed.csv                     # 5行
    summary.csv
    support_manifest.csv
    clip_concentration.csv
    config.json
  ...其余五个生成器...
  all_generators_summary.csv
  baseline_reference_summary.csv     # 仅实际提供已有CSV时
  baseline_comparison.csv            # 同上，差值而非预计结果
  R24_REPORT.md                       # 据真实运行结果生成的中文记录
```

source_index 为原排序列表索引，observed_label 控制原型归属，true_label/source_class 保留来源。标签扰动 result 中保留 original_support_indices、swapped_indices、query_indices、交换后 groups；B 还保留 anchor_index/seed、Top10相似度、每类支持集平均成对cosine和同锚点均值。
clip_concentration 比较同seed官方 Random 10+10 的 CLIP内部相似度，不额外跑 DDFSD clean。任一类集中组内部平均不严格高于Random则 CONCENTRATION_NOT_VERIFIED，保留全部指标和该seed，不重选锚点。
报告不声称 CLIP Top10 有人工“猫/狗”等类别标签。

模型参数不更新。预检一次检查所选所有数据/缓存及六组模型，再按生成器复用冻结模型跑各seed条件。
任务原子落盘；原命令重启时重新校验身份、SHA、确定性选择、完整Query与指标后复用已完成任务。中断未完成的条件重跑。
身份包括代码、权重/stats、数据内容清单、缓存、协议、设备环境、精度和batch。改变这些字段，或拉取新commit，须选新输出目录；不绕过身份检查混用旧结果。
`.r24.lock` 防止并发写；SIGKILL残留时先确认旧进程已结束，再删除对应输出/缓存根目录的锁，重跑原命令。
范围完整状态为 COMPLETE_REQUESTED_SCOPE，只有默认六类五seeds全部四条件完成才是 COMPLETE_R24_A_B；部分条件仅生成 PARTIAL 范围宏平均，不冒充六类完整宏平均。
失败记录 FAILED_OR_INTERRUPTED_SEE_LAST_FAILURE；查 traceback，在本机修复并提交后由AutoDL拉新commit、新目录重跑，不现场修改实验逻辑。
不自动上传数据/缓存/输出，不在 AutoDL commit/push 结果。需要复核时用户自行下载小 CSV、配置、报告；完整索引gzip保留在服务器。

## 9. 本机验证与边界

```bash
python -B -m unittest discover -s tests -v
python -B test_ddfsd_r24.py --help
python -B tools/build_r24_clip_cache.py --help
bash -n scripts/eval_r24.sh
git diff --check
```

新增Python文件与重构评估文件均使用 py_compile 编译到临时目录，避免改变仓库原先跟踪的历史 pyc。
纯标准库测试覆盖三档交换/嵌套/来源守恒、全部Query、稳定RNG、CLIP Top10/同锚点/平局/锚点保留、未确认集中结果保留、缓存身份/数量/维度/非有限值和元数据错误。
已有NumPy运行时还执行模型替身回归：从基线 03f0b4d 保存的不可变原评估快照，与重构公开入口逐分数、预测、索引和输出字段比较；0%额外确认 [1,10,2,512]；非零交换检查实际送入模型的20张顺序/来源以及不变的Query标签。缓存NPY实际落盘/读取/损坏也测试。
所有测试路径、向量及临时报告明确 SYNTHETIC，自动清理，不是正式实验结果。原核心 torch 测试在本机缺 torch 时明确 NOT_RUN_ENV_MISSING；安装了原环境时原测试函数照常运行。
未做：真实 checkpoint/GPU数值parity、真实CLIP加载/提取、全量六类评估、真实集中程度和性能下降；必须由 AutoDL 实际验收，不预判结果支持论文。

本机最终检查：R2-4 专项为20项测试，现成NumPy运行时全部通过；标准Python为15项通过、5项NumPy环境跳过。与既有R2-1一起执行完整 discover 共47项：现成NumPy运行时46项通过、1个原Torch核心模块跳过；标准Python37项通过、10个环境跳过（9项NumPy测试、1个原Torch核心模块）。原核心模块的6个测试函数未执行。
7个新增/修改Python文件 py_compile 到临时目录通过；两个帮助命令、Bash语法、Shell真实参数传递、git diff --check通过。
原 test_ddfsd.py、scripts/eval_main.sh、train_ddfsd.py、模型/损失、数据集和正式采样文件 diff为空；仅 util/ddfsd_eval.py 提取私有共享主体，其公开默认行为由不可变原版本合成回归验证。

API依据：[Transformers CLIP官方文档](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/clip#transformers.CLIPVisionModelWithProjection)、[Hub预下载/固定revision](https://huggingface.co/docs/huggingface_hub/guides/download)。
