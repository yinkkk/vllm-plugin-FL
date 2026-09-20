# MiniCPM5-2B 双平台 Baseline 执行计划

> **For agentic workers:** 执行时使用 `executing-plans` 技能逐项推进；只有用户明确要求并行代理时才委派。使用下方复选框记录状态。

**Goal:** 在天数 BI-V150 和沐曦曦云 C500（64 GB）上分别跑通 MiniCPM5-2B，建立可复现的吞吐与精度基线。

**Architecture:** 使用赛事或厂商提供的 Linux GPU 容器承载驱动兼容的 PyTorch、Triton/FlagTree；通过 vllm-plugin-FL 加载模型并调用 FlagGems。两种硬件分别运行相同测试场景、记录环境和结果，不混用厂商依赖或配置。

**Tech Stack:** vllm-plugin-FL `flagos-2026-s2`、FlagGems `v5.3.5`、vLLM、厂商 PyTorch、厂商 Triton/FlagTree、ModelScope、Bash。

## 全局约束与当前状态

- 推理框架必须来自 `https://github.com/flagos-ai/vllm-plugin-FL` 的 `flagos-2026-s2` 分支，实际执行记录完整 commit。
- FlagGems 必须基于 `https://github.com/flagos-ai/FlagGems` 的 `v5.3.5` tag，实际执行记录完整 commit。
- 权重使用 `OpenBMB/MiniCPM5-2B`，以比赛指定 revision 为准；不替换为 Base、GPTQ 或其他量化版本。
- 优化目标为 `total tokens/s`，精度必须满足赛事标准。当前尚未拿到官方评测脚本、数据集、阈值和镜像信息，因此本计划不宣称官方验收通过。
- 首轮使用 BF16、真实权重、单卡 TP=1；不使用 dummy 权重、量化 KV cache 或模型裁剪。
- 所有命令在对应 GPU 服务器的 Linux 容器内执行，不在 Windows PowerShell 执行。
- 当前只完成源码分析和计划编写，未连接两台服务器，未获得任何实测吞吐或精度分数。
- 2026-09-20 本地仓库为 `main`，origin 为 `yinkkk/vllm-plugin-FL`；本地远程跟踪的比赛分支为 `13eb9be`。该短哈希是检查快照，不代替比赛要求的最终提交版本。
- 比赛分支 README 和测试依赖指向 vLLM 0.24.0，但沐曦 Dockerfile 仍固定 0.20.2 和另一版 FlagGems，且提及 C550 验证。不能直接认定该 Dockerfile 满足 C500 比赛要求。
- 仓库 `examples/minicpm/README.md` 是旧多模态模型示例，不是 MiniCPM5-2B 的部署规范。

## 交付物与目录

本次只新增本文档，不修改模型、算子或本地 Git 分支。执行计划时，在两台服务器各建立独立运行目录；每次执行生成新的 RUN_DIR，避免覆盖结果。

```text
baseline-work/
  vllm-plugin-FL/                 比赛框架源码
  FlagGems/                       指定版本算子源码
  models/MiniCPM5-2B/             权重，也可指向赛事已有挂载
  runs/<vendor>-<timestamp>/
    environment.txt              系统、Python 包、设备信息
    source-versions.txt          两个源码仓库提交
    model-config.json            本次权重的配置快照
    model-files.sha256           权重与 tokenizer 文件校验
    serve.log                    功能验证服务日志
    smoke-response.json          功能验证响应
    throughput-*.log              完整性能测试日志
    throughput-*.json             性能测试结果
    accuracy/                    官方精度评测输出
    acceptance.md                人工填写的最终验收记录
```

## 任务 1：确认赛事条件和两台机器环境

**输入：** 两台服务器登录方式、赛事镜像与规则。

**输出：** 每台服务器的环境记录，明确是否可以进入模型运行阶段。

- [ ] 收集两台机器的 SSH 地址、端口、账号和可用 GPU 编号；密钥与密码不写入本仓库。
- [ ] 记录镜像名称及 digest、宿主驱动版本、容器 GPU 挂载方式和模型挂载路径。使用平台提供的设备透传启动方式，不套用 NVIDIA 专用 Docker 命令。
- [ ] 确认评测使用单卡还是多卡、输入输出长度分布、请求数、采样设置、计时范围、精度任务与阈值，以及是否限制算子回退和调参范围。
- [ ] 两个平台分别执行以下初始化；每个后续终端都需要设置同一组 BASELINE_ROOT、VENDOR、RUN_DIR、MODEL_PATH 环境变量。

```bash
set -euo pipefail
mkdir -p baseline-work
cd baseline-work
export BASELINE_ROOT="$PWD"

# 沐曦设置 metax；天数将下一行改成 iluvatar。
export VENDOR=metax
export RUN_DIR="$BASELINE_ROOT/runs/${VENDOR}-$(date +%Y%m%d-%H%M%S)"
export MODEL_PATH="$BASELINE_ROOT/models/MiniCPM5-2B"
mkdir -p "$RUN_DIR/accuracy"

{
  date -Is
  uname -a
  python --version
  python -m pip list --format=freeze
  python -c 'import torch; print("torch:", torch.__version__); print("device_available:", torch.cuda.is_available()); print("device_count:", torch.cuda.device_count()); print("device:", torch.cuda.get_device_name(0)); print("memory:", torch.cuda.get_device_properties(0).total_memory)'
} 2>&1 | tee "$RUN_DIR/environment.txt"
```

**通过条件：** 容器能够识别指定 GPU；设备没有被其他测试占用；厂商运行时与宿主驱动匹配。`torch.cuda` 是适配环境暴露的接口，不代表显卡是 NVIDIA。若设备不可见，先修复容器/驱动环境，不继续安装 Python 包。

## 任务 2：固定框架与算子版本

**输入：** 已验证的赛事/厂商容器。

**输出：** 可导入的 FL 平台、两个源码提交和依赖快照。

- [ ] 首先检查预装版本；不要盲目升级 torch、triton、transformers 或覆盖厂商编译产物。

```bash
python -m pip show torch triton flagtree vllm flag-gems vllm-plugin-fl
```

- [ ] 在新的工作目录拉取源码；如果目录已经存在，先检查分支、提交及未提交修改，不重复 clone 或覆盖。

```bash
cd "$BASELINE_ROOT"
git clone --branch flagos-2026-s2 --single-branch \
  https://github.com/flagos-ai/vllm-plugin-FL.git
git clone --branch v5.3.5 --depth 1 \
  https://github.com/flagos-ai/FlagGems.git

{
  git -C vllm-plugin-FL branch --show-current
  git -C vllm-plugin-FL rev-parse HEAD
  git -C vllm-plugin-FL status --short
  git -C FlagGems describe --tags --exact-match
  git -C FlagGems rev-parse HEAD
  git -C FlagGems status --short
} | tee "$RUN_DIR/source-versions.txt"
```

- [ ] 若赛事指定镜像已有匹配的 vLLM，保留该安装并记录版本。如果没有，先确认赛事兼容要求；以下是比赛分支 README 的 vLLM 0.24.0 empty 安装路径，仅用于依赖齐全的厂商环境。

```bash
# 条件执行：不要在已就绪镜像中无条件重装。
cd "$BASELINE_ROOT"
git clone --branch v0.24.0 --depth 1 https://github.com/vllm-project/vllm.git
VLLM_TARGET_DEVICE=empty python -m pip install \
  --no-build-isolation --no-deps ./vllm
```

- [ ] 安装可编辑的指定源码，保留厂商依赖；缺少构建工具时，按镜像配套说明补齐，再执行安装。

```bash
cd "$BASELINE_ROOT"
export GEMS_VENDOR="$VENDOR"
python -m pip install --no-build-isolation --no-deps -e ./FlagGems
python -m pip install --no-build-isolation --no-deps -e ./vllm-plugin-FL

export VLLM_PLUGINS=fl
export USE_FLAGGEMS=1

python -m pip check
python -c 'import flag_gems, vllm_fl; from vllm.platforms import current_platform; print("FlagGems:", flag_gems.__file__); print("FL:", vllm_fl.__file__); print("platform:", type(current_platform)); print("vendor:", getattr(current_platform, "vendor_name", None))'
python -m pip list --format=freeze > "$RUN_DIR/installed-packages.txt"
```

**通过条件：** 导入位置对应本次源码；平台为 FL，vendor 对应 `metax` 或 `iluvatar`；依赖检查没有未解释的冲突。`--no-deps` 不会自动补全运行依赖，不能把安装成功当作环境验收。

**特别说明：** 不要设置 `VLLM_VENDOR=metax` 或 `VLLM_VENDOR=iluvatar`，检查时该分支 setup.py 的扩展构建选项只接受 cuda。是否需要构建原生扩展应依据厂商镜像和实际错误处理，不把 README 的 CUDA 构建指令直接套到两种国产卡。

## 任务 3：固定权重并完成功能验证

**输入：** 可用推理环境、赛事模型目录或下载权限。

**输出：** 模型文件标识、服务日志和实际生成结果。

- [ ] 若已有赛事挂载，令 MODEL_PATH 指向该目录；没有挂载才执行下载。赛事指定 revision 时，应使用该 revision 下载，两台机器必须一致。

```bash
modelscope download --model OpenBMB/MiniCPM5-2B \
  --local_dir "$MODEL_PATH"
```

- [ ] 核对配置与文件校验值；大模型文件校验只需在建立基线时执行一次。

```bash
cp "$MODEL_PATH/config.json" "$RUN_DIR/model-config.json"
python -c 'import json, os; c=json.load(open(os.path.join(os.environ["MODEL_PATH"], "config.json"))); print({k:c.get(k) for k in ["architectures", "torch_dtype", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"]})'
(
  cd "$MODEL_PATH"
  find -L . -type f \( -name '*.json' -o -name '*.safetensors' -o -name '*.model' -o -name '*.jinja' \) \
    -print0 | sort -z | xargs -0 sha256sum
) > "$RUN_DIR/model-files.sha256"
```

预期架构为 `LlamaForCausalLM`，官方当前配置为 BF16、hidden_size=2048、42 层、16 个 attention heads、2 个 KV heads。若赛事权重配置不同，先核对 revision，不修改配置强行匹配。

- [ ] 使用平台的设备可见性配置隔离一张空闲 GPU，再启动保守的功能验证服务。以下内存比例要求设备有足够空闲显存。

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name MiniCPM5-2B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager 2>&1 | tee "$RUN_DIR/serve.log"
```

- [ ] 服务就绪后，在另一个已设置相同 RUN_DIR 的终端请求并保存响应。

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniCPM5-2B","messages":[{"role":"user","content":"用一句话解释什么是矩阵乘法。"}],"temperature":0,"max_tokens":256}' \
  | tee "$RUN_DIR/smoke-response.json"
```

**通过条件：** 服务完成加载，使用正确 FL 平台，无 kernel/设备异常，请求成功且返回非空生成内容。模型可能生成思考内容，短回复截断不等于精度失败；必要时增加功能验证输出长度。此步骤不作为精度评测。

## 任务 4：建立可复现的离线吞吐基线

**输入：** 功能验证通过的两套环境。

**输出：** 每个平台三个长度场景的重复测量和 JSON 结果。

- [ ] 用 Ctrl+C 停止自己启动的服务，确认其 GPU 进程已退出。不要终止其他用户进程。吞吐测试自行加载模型，不与服务同时运行。
- [ ] 确认当前 vLLM CLI 支持参数，并将实际帮助保存到结果目录。

```bash
vllm bench throughput --help > "$RUN_DIR/throughput-help.txt"
```

- [ ] 执行以下开发基线。两个平台使用相同 BF16、单卡、种子、请求数、长度、缓存与执行模式。8192 上下文足够覆盖这里最大的 6144+1024；正式场景超出时必须重新配置。

```bash
set -euo pipefail
for input_len in 1024 4096 6144; do
  for run in 0 1 2 3; do
    name="throughput-in${input_len}-out1024-eager-run${run}"
    vllm bench throughput \
      --model "$MODEL_PATH" --backend vllm \
      --dataset-name random \
      --input-len "$input_len" --output-len 1024 --num-prompts 300 \
      --dtype bfloat16 --tensor-parallel-size 1 \
      --max-model-len 8192 \
      --max-num-seqs 64 --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.85 \
      --no-enable-prefix-caching \
      --seed 42 --enforce-eager \
      --output-json "$RUN_DIR/${name}.json" \
      2>&1 | tee "$RUN_DIR/${name}.log"
  done
done
```

这里关闭 prefix caching 是为了明确开发基线条件，不代表比赛要求关闭。正式测试按赛事规则设置。若厂商镜像 CLI 与 0.24.0 不同，先根据保存的帮助核对，不静默删除参数；记录所有变化并保持两平台条件一致。

- [ ] run0 作为首次运行/编译缓存预热记录；run1–3 分别保留，并计算中位数与最小/最大值。每次命令都创建新进程，run0 不能证明后续进程已完成所有进程内预热；最终计时范围、预热规则服从官方脚本。
- [ ] 检查日志是否有异常、请求未完成、输出长度异常、显存不足或频繁 KV cache 抢占。不能只摘取最高数字。

```bash
python - <<'PY'
import json
import os
import statistics
from pathlib import Path

root = Path(os.environ['RUN_DIR'])
for length in (1024, 4096, 6144):
    values = []
    for run in (1, 2, 3):
        path = root / f'throughput-in{length}-out1024-eager-run{run}.json'
        data = json.loads(path.read_text())
        assert data['num_requests'] == 300, path
        assert data['elapsed_time'] > 0, path
        assert data['tokens_per_second'] > 0, path
        values.append(data['tokens_per_second'])
    print(f'{length}/1024: median={statistics.median(values):.2f}, '
          f'min={min(values):.2f}, max={max(values):.2f} total tokens/s')
PY
```

**指标解释：** 此 CLI 的 `total tokens/s` 为输入与输出 token 总数除以推理计时，JSON 字段为 `tokens_per_second`；`output tokens/s` 仅统计生成 token。该离线指标不等于包含 HTTP 开销的服务吞吐。不同长度分布的总 token 吞吐不直接比较，不自行把三个场景的均值作为比赛成绩。

**通过条件：** 两台机器各完成三个场景、每场景三个有效测量；保存运行日志和 JSON；无运行失败。若无法承载初始 batch，可降低 max-num-seqs 排障，但应明确标为另一配置并重新建立可比基线。

## 任务 5：精度基线与官方验收

**输入：** 官方精度评测工具、数据、生成设置和阈值。

**输出：** 每台机器的原始精度结果及验收结论。

- [ ] 向赛事平台取得正式精度与性能评测命令；记录脚本 commit、数据 revision、模型 revision、精度阈值和成绩汇总方式。
- [ ] 先在未做性能优化的当前版本执行官方精度脚本，保存完整命令、日志和原始输出到 `$RUN_DIR/accuracy/`。
- [ ] 核对 tokenizer、chat template、thinking 模式、采样参数、最大输出长度、停止条件与官方要求一致。功能验证的 temperature=0 不自动成为正式精度设置。
- [ ] 两个平台分别通过官方精度门槛；若需要 logits/算子数值对比，采用官方给出的误差容限，不自行指定“等价”阈值。
- [ ] 用官方性能脚本复测；若官方走在线服务，不用本计划的离线吞吐数字代替。

**缺少官方材料时的状态：** 可以完成任务 1–4，交付“开发基线已跑通”；任务 5 保持未完成，不能标记“比赛 baseline 已验收”。仓库的 LM Eval 脚本只能作为辅助诊断，不能推断就是赛事评测任务。不要为补齐记录而虚构官方命令。

## 任务 6：冻结基线与优化交接

- [ ] 为每个平台填写下表并存为对应 RUN_DIR 下的 `acceptance.md`，未执行项明确写“未执行”，不要估计分数。

| 字段 | 天数 BI-V150 | 沐曦 C500 64 GB |
|---|---|---|
| 运行目录、日期 | 未执行 | 未执行 |
| GPU 数量与可见设备 | 未执行 | 未执行 |
| 镜像 digest / 驱动 | 未执行 | 未执行 |
| torch / Triton或FlagTree / vLLM | 未执行 | 未执行 |
| FL commit / FlagGems commit | 未执行 | 未执行 |
| 模型 revision / 文件校验 | 未执行 | 未执行 |
| BF16 功能验证 | 未执行 | 未执行 |
| 1k/1k 中位吞吐及范围 | 未执行 | 未执行 |
| 4k/1k 中位吞吐及范围 | 未执行 | 未执行 |
| 6k/1k 中位吞吐及范围 | 未执行 | 未执行 |
| 官方精度分数 / 阈值 | 未执行 | 未执行 |
| 官方 total tokens/s | 未执行 | 未执行 |
| 已知回退、限制和异常 | 未执行 | 未执行 |

- [ ] 固定基线源码和结果，不在同一结果文件上覆盖新实验。
- [ ] 后续优化按“batch 参数 → 图执行 → KV cache/调度 → 热点 kernel”推进，每次只改变明确的一组因素。
- [ ] 优化版本对比同平台、同评测条件的基线；加速比 = 优化后 total tokens/s ÷ 基线 total tokens/s。
- [ ] 调整执行模式或算子实现后重新跑精度；精度未过关的性能结果不作为有效优化成绩。

基线阶段不进行大范围自动调优。仓库 `benchmark_throughput_autotune.py` 会探索算子启用组合，示例包含 dummy 权重，首轮也可能禁用 FlagGems；使用前必须核对赛事规则并改成真实权重。

## 两个平台的排障入口

| 现象 | 首先核查 | 处理边界 |
|---|---|---|
| torch 无法发现设备 | 宿主驱动、容器设备透传、厂商 torch | 不靠安装通用 CUDA wheel 解决 |
| 导入 vLLM 报 API 不匹配 | vLLM 与插件 commit、厂商依赖 | 不混用 main 与比赛分支的安装文档 |
| BF16 不支持或 kernel 编译失败 | 硬件能力、厂商编译器和对应 kernel | 不静默改 FP16；需记录并满足比赛精度要求 |
| 天数 Triton 编译异常 | iluvatar 后端、FlagTree/Triton 版本和已有适配补丁 | 不直接升级到通用 Triton |
| 沐曦 attention 导入失败 | MACA 配套算子与镜像兼容性 | 分支默认 attention 选择 vendor:metax |
| 模型能生成但很慢 | 实际 dispatch、回退、batch、设备占用、重算 | 先查日志，再做 profiler；不先重写算子 |
| 图执行失败 | 先确认 eager 基线，再检查编译/graph 支持 | eager 成功不代表 graph 已支持 |
| 提升吞吐但精度下降 | 权重、dtype、融合、采样、停止条件 | 回到已验证基线定位，不放宽阈值 |

比赛分支中，`vllm_fl/dispatch/config/metax.yaml` 默认将 attention 交给厂商实现；`iluvatar.yaml` 优先 FlagOS。FlagGems 处于开启状态不等于所有算子都实际走 FlagGems，日志和调度配置需要一起检查。调试日志仅用于定位，正式计时恢复正常日志级别。

## 参考依据

- [比赛分支与安装说明](https://github.com/flagos-ai/vllm-plugin-FL/tree/flagos-2026-s2)
- [FlagGems v5.3.5](https://github.com/flagos-ai/FlagGems/tree/v5.3.5)
- [赛事指定模型来源](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B)
- [官方模型配置镜像](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json)
- [模型官方 vLLM 部署说明](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm.md)
- [vLLM 0.24.0 吞吐测试实现](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/benchmarks/throughput.py)
- 比赛分支内部入口：`benchmarks/flagos_eval/run_benchmark.sh`、`benchmarks/benchmark_throughput_autotune.py`、`vllm_fl/dispatch/README.md`、`vllm_fl/dispatch/config/{metax,iluvatar}.yaml`。

执行顺序：每台机器依次完成任务 1–6。两平台可分别执行，但必须保留独立环境与结果；任何一台未通过官方精度验收，都不能以另一台的结果替代。
