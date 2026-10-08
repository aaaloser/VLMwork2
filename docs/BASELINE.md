# QIG InternVL2-8B 实验 baseline

整理日期：2026-10-08。此 baseline 保留 QIG 官方历史，并记录本地 InternVL2-8B、COCO 校准和 MMMU 复现所需的兼容修改。

## 分支、标签与远程

- `main`：保留官方原始提交，整理时为 `06fa88134671647827dea08dc37c50c81e4eeac7`。
- `upstream/main`：官方远程跟踪分支；地址为 `https://github.com/ucas-xiang/QIG.git`。
- `baseline/qig-internvl2-8b`：已复现代码及兼容补丁。
- `qig-internvl2-8b-repro-baseline`：固定 baseline 标签。后续对照应记录该标签解析出的提交 SHA。
- `exp/initial`：从 baseline 创建的实验起点，当前 checkout 在此分支。

个人远程 `origin` 尚未配置。创建自己的空远程仓库后，在本仓库执行：

```bash
git remote add origin <你的仓库地址>
git push -u origin baseline/qig-internvl2-8b
git push origin qig-internvl2-8b-repro-baseline
git push -u origin exp/initial
```

开始一个独立实验时，从固定标签建立分支：

```bash
git switch -c exp/your-method qig-internvl2-8b-repro-baseline
```

兼容修复留在 baseline，算法创新放在 `exp/*`。实验记录至少包含提交 SHA、配置文件、scale SHA256、数据 SHA256、环境、GPU 和评估任务。

## baseline 内的修改

1. `qmllm/models/internvl2/internvl2.py`：带 labels 的 reweight 前向关闭 cache，获得完整序列 logits。
2. `qmllm/methods/qig/quantize/pre_quant.py`：冻结参数、清空参数梯度、仅对输入 embedding 反传，关闭额外输出。
3. `main.py`：按当前 lmms-eval 接口传入 `model=args.model` 和 `lm=lm`。
4. `tests/`：三个相关回归测试。
5. `.gitignore`：忽略本地日志、scale、结果、Python 缓存、编辑器和构建产物；文件仍保留在磁盘。
6. `patches/` 与 `scripts/`：追踪原先位于模型目录/缓存中的生成 logits 优化。

## 第三方依赖

两个第三方仓库现在是 submodule，由父仓库 gitlink 固定到具体提交：

| 目录 | 来源 | 固定 SHA |
| --- | --- | --- |
| `3rdparty/LLaVA-NeXT` | `https://github.com/LSY-noya/LLaVA-NeXT.git` | `1a7e8b2b4880f7548c1883669302a8b36bd79df6` |
| `3rdparty/lmms-eval` | `https://github.com/LSY-noya/lmms-eval.git` | `9c030a5932b6d340ff58611339415a74f5ed4b6f` |

新机器克隆时使用 `git clone --recurse-submodules <你的仓库地址>`；已有 clone 使用 `git submodule update --init --recursive`。日常更新不要使用 `--remote`，它会偏离这里固定的依赖版本。第三方改动应先在其仓库内提交，再更新父仓库 gitlink；如果提交只存在本机，应先把它推送到自己可访问的第三方 fork。

现有 editable 安装路径未改变。模型、环境和数据仍位于 `/data/liyifan24/VLM/` 下。

## 模型代码补丁

生成时只投影最后一个 token 可以降低长 prompt 的显存峰值；校准 loss 仍要求完整序列，因此必须与 wrapper 的 `labels -> use_cache=False` 修复一起使用。

```bash
bash scripts/apply_internlm2_patch.sh /data/liyifan24/VLM/models/InternVL2-8B
```

脚本校验上下文后应用补丁；已经应用时直接报告，其他代码版本不匹配时退出。2026-10-08 检查时模型源文件和当前 Hugging Face 缓存都已含该补丁。新机器应在启动加载模型前补丁模型目录源文件，之后 Transformers 可按本地源文件同步动态模块缓存。若进程已加载旧模块，需要重新启动进程。脚本不删除 Hugging Face 缓存。

## 环境记录

`environments/qig-observed-20261008.json` 记录当前 Python 和全部 Python 包版本，是观测快照，不是验证过可直接安装的依赖锁文件。关键版本：Python 3.11.16、torch 包版本 2.8.0、Transformers 4.49.0、FlashAttention 2.8.3.post1，glibc 2.31。

FlashAttention 在本机使用 CUDA 12.4 源码编译（对应 RTX 3090），PyTorch CUDA 运行时为 12.8。编译工具版本与运行时版本不同。glibc 2.31 无法加载先前尝试的官方预编译 wheel；新环境需要验证实际二进制兼容性，不能仅按版本号安装。

本机重建 FlashAttention 的已采用方式：

```bash
CUDA_HOME=/usr/local/cuda-12.4 \
PATH="$CONDA_PREFIX/bin:/usr/local/cuda-12.4/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
TORCH_CUDA_ARCH_LIST=8.6 FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=6 NVCC_THREADS=4 \
python -m pip install --no-build-isolation --no-cache-dir \
    --no-binary=flash-attn flash-attn==2.8.3.post1
```

在其他机器上按实际 GPU/CUDA 调整，以上命令要求已激活目标环境。

## 配置与运行

`configs/baseline/` 提供 n=32 的 W4A8 和 W3A16 搜索/评估配置，路径按当前机器填写。复制配置后更改 n_samples、位宽或方法时，应使用新的 scale 路径。已有 scale 文件会让 `--run_process` 跳过搜索，不能以运行成功作为重新校准的证据。

```bash
cd /data/liyifan24/VLM/code/QIG
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /data/liyifan24/VLM/envs/qig/bin/python main_quant.py \
    --config configs/baseline/internvl2_w4a8_n32_quant.yaml
CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /data/liyifan24/VLM/envs/qig/bin/python main.py \
    --config configs/baseline/internvl2_w4a8_n32_eval.yaml
```

W3A16 使用对应的两个 `internvl2_w3a16_n32_*` 配置。新评估配置使用 `mmmu_val`（900 个带标签样本）；之前 `--tasks mmmu` 还会生成 10,500 个 test 样本的 submission，总响应数为 11,400。若需要 test 提交，将复制的评估配置改成 `tasks: mmmu`。

历史 2026-09-23 记录：W4A8 n=32 reweight+distort 的 MMMU validation 为 44.667%，W3A16 n=32 reweight 为 45.778%。这些是本地复现结果，不是原论文数字。历史命令保留在 `reproduce.md`。

`experiments/local-artifacts-20261008.json` 记录本地数据、scale 和全部现有结果 JSON 的路径/校验和。结果 JSON 没有位宽和 scale 信息，且部分 scale 后来重跑过，因此当前文件校验和不能证明历史结果使用了哪一份 scale。正式新实验应在启动前记录这些关联。

## 本地验证

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    /data/liyifan24/VLM/envs/qig/bin/python -m unittest -v \
    tests.test_main_evaluator_arguments \
    tests.test_internvl2_reweight_use_cache \
    tests.test_qig_reweight_activation_gradients
```

模型、数据、scale、日志、结果和环境没有复制入 Git。整理工作不会自动触发完整量化或评估。
