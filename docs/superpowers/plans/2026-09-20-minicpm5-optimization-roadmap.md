# MiniCPM5-2B 双平台推理优化路线

> 本文是阶段性路线，不是已经验证的加速方案。后续执行使用 `executing-plans` 按阶段推进；只有用户明确要求时才启动并行代理。先完成 [baseline 执行计划](2026-09-20-minicpm5-dual-vendor-baseline.md)，再依据实测结果收敛具体修改。

**目标：** 在天数 BI-V150 和沐曦曦云 C500（64 GB）上，在满足比赛精度要求的前提下提高 MiniCPM5-2B 的 total tokens/s。

**实现范围：** 推理参数与调度、FL 算子接入和后端选择、FlagGems 的实际热点算子。先利用现有实现，再针对模型形状优化，不重写整套推理框架。

**技术栈：** vllm-plugin-FL `flagos-2026-s2`、FlagGems `v5.3.5`、赛事匹配的 vLLM / 厂商 PyTorch / Triton 或 FlagTree。

## 1. 当前代码基础与约束

2026-09-20 本地检查结果：

| 仓库 | 当前状态 | 后续用途 |
|---|---|---|
| `C:\Users\Yqj\projects\vllm-plugin-FL` | `dev-minicpm5-baseline`，基于比赛分支 `13eb9be`；文档提交 `615927e` | 保存推理接入、参数配置和实验文档 |
| `C:\Users\Yqj\projects\FlagGems` | 当前 `master`，提交 `52674ef8f`；origin 指向官方仓库 | 开发前从比赛 tag 建个人分支，不能直接以 master 作为比赛基线 |
| FlagGems 比赛版本 | 本地 `v5.3.5` 解析为 `a7620cc191a0b42e040194622c5758b22a7a25dc` | 固定算子库基础，所有优化提交可追溯 |

- FL 后续只推送到个人 fork，不创建 PR、不向官方仓库提交。
- FlagGems 尚未配置个人 fork；本文不修改它的分支或 remote。需要发布算子修改时，再配置个人 fork，官方 remote 仅用于读取。
- 一次有效实验必须同时记录 FL commit、FlagGems commit、未提交 diff、环境和配置，不能只记录一个仓库的版本。
- 先确认赛事是否允许修改算子源码、选择厂商后端、调整 graph/调度和使用缓存；尚未明确的选项列为条件性实验。
- BF16、真实权重、官方生成语义是基准。量化、KV cache 降精度、剪枝和改变输出长度不作为当前优化路线。
- 本文中的优先级是启动顺序，profile 后应重新排序；不承诺固定倍数提升，不根据显存容量推断硬件算力或带宽。

## 2. 总体策略

可以从三个方向切入：

| 路径 | 优点 | 局限 | 本项目安排 |
|---|---|---|---|
| 参数和执行模式优化 | 修改小、验证快，可确定 batch 和 graph 是否限制吞吐 | 收益可能有限 | 第一阶段优先 |
| 热点算子和模型形状优化 | 能结合硬件特性，适合形成可解释的技术成果 | 需要 profile、数值和端到端验证 | 主攻方向 |
| 大规模模型/引擎重构 | 可能暴露更深层融合机会 | 适配、精度、维护风险高 | 前两条确认不足后再评估 |

推荐顺序：**固定基线 → 测量瓶颈 → 调参数/执行模式 → 优化一到两个热点算子 → 组合消融 → 官方验收。**

两台机器共享模型语义、测试集和代码组织，但分别选择参数与 kernel 配置。只有明确测得相同瓶颈时才复用优化结论。

## 3. 模型特征与优化假设

以下维度来自已核对的官方模型配置，实际运行再次确认赛事权重 revision：42 层、hidden size 2048、FFN intermediate size 6144、16 个 Q heads、2 个 KV heads、head dimension 128、vocab size 130560。

| 模型特征 | 值得验证的假设 | 首先采集的证据 |
|---|---|---|
| 42 层、小隐藏维度 | decode 的小 kernel 调度开销可能明显 | 每步 kernel 数、CPU/GPU 空隙、graph 前后时间 |
| GQA，Q:KV=8:1 | attention 的 KV 读取复用和访存布局可能影响吞吐 | attention 实际后端、上下文长度、读写耗时 |
| FFN 宽度 6144 | 激活和矩阵乘可针对常见 shape 调优 | 实际 M/N/K、stride、dtype 和调用频次 |
| 词表 130560 | LM head 或采样可能成为 decode 热点 | 输出投影与采样占比，而非仅看 Transformer 层 |
| 2B 单卡场景 | 多卡通信未必带来净收益 | 单卡饱和度与赛事卡数要求 |

形状示例用于定位，不代替运行时采样：TP=1 时，Q/K/V 输出维度理论上为 2048/256/256，若合并投影则总宽度为 2560；gate/up 合并时输出宽度为 12288。是否合并、输入矩阵布局、实际 batch 大小必须从执行路径确认。

Prefill 和 decode 分开分析：前者常见较大的 token batch，后者的计算形状随活跃序列数变化。不能仅用单请求 decode 或纯 prefill 的结果代表比赛混合负载。

## 4. 阶段与验收

### P0：冻结可复现基线

- [ ] 按已有 baseline 文档在两种卡上完成环境、功能、吞吐和精度验证。
- [ ] FlagGems 开发前从 `v5.3.5` 建个人分支；保留当前 master，不以切换版本为由丢弃已有修改。
- [ ] 固定权重、软件版本、设备数、精度、数据和生成配置。
- [ ] 记录默认 dispatch、回退情况及 graph 支持状态。

**产出：** 两个平台各一份 baseline 数据包和对应两个仓库的版本标识。

**通过条件：** 吞吐可重复、真实模型功能正常；官方精度材料未齐时只能标为开发基线，正式验收仍待执行。

### P1：识别端到端瓶颈

- [ ] 分别采样短输入、长输入和 decode 较重的代表场景；使用实际评测负载核对结论。
- [ ] 优先使用厂商可用 profiler；若 PyTorch profiler 在对应环境支持 GPU 活动采集则使用它。先确认能看到真实设备事件，不将 CPU 时间冒充 kernel 时间。
- [ ] 记录 top kernels、调用次数、总耗时、实际 shape、CPU 调度间隙和显存/抢占现象。
- [ ] 区分首次编译、模型加载、稳态推理时间。profile 运行和正式计时运行分开，正式计时关闭重型 tracing。
- [ ] 形成每个平台的前三项瓶颈及对应代码入口。

**产出：** 两张热点表：耗时占比、假设、可改位置、预期影响、验证实验。

**通过条件：** 每个候选修改有测量证据；若尚无设备 profile，只能保留为假设。

### P2：先优化参数与执行模式

建议用小范围顺序搜索，不一次跑所有参数的笛卡尔积：

| 因素 | 起始实验范围 | 观察项 |
|---|---|---|
| `max-num-seqs` | 32、64、128、256；有收益且有余量才扩展 | 总吞吐、活跃序列、排队和抢占 |
| `max-num-batched-tokens` | 2048、4096、8192、16384，结合实际长度 | prefill/decode 竞争、显存和吞吐 |
| `gpu-memory-utilization` | 0.85、0.90；0.95 仅在空闲独占且有安全余量时尝试 | KV cache 容量、graph 占用和 OOM |
| graph / 编译模式 | eager 对照已支持的 graph 配置 | kernel 启动开销、捕获成功率、显存变化 |
| chunked prefill | 后端支持且规则允许时做开关对照 | 长输入时 decode 是否被阻塞 |
| prefix caching | 仅在规则允许且真实存在前缀复用时比较 | 命中率、收益与计时规则 |

- [ ] 固定其他因素，先找合理 batch 范围，再测执行模式及交互影响。
- [ ] 删除 `--enforce-eager` 只表示允许默认执行模式，不等于实际启用了 graph；从日志或 profile 验证。
- [ ] 保持 max-model-len 能覆盖官方最大输入加输出，不通过截短样本制造提升。
- [ ] 在两个平台分别保留最佳合法配置，并记录为什么其他候选失败。

**产出：** 不改 kernel 的配置优化版本及相对 P0 的收益。

**通过条件：** 无精度回归、无请求失败、吞吐提升超出重复测量波动；若无显著提升，保留基线并进入 P3，不硬选“最优”噪声点。

### P3：针对一到两个热点优化 FlagGems

根据 P1 的耗时占比排序，不把下表顺序当作已经确认的热点：

| 候选 | 可尝试的修改 | 必须验证 |
|---|---|---|
| RMSNorm / residual + RMSNorm | block、warp 数、向量化访存、减少重复加载 | FP32 累加语义、epsilon、原地更新和 residual 返回语义 |
| SiLU + Mul | 针对 6144 宽度的 tile、向量化、减少访存 | 非连续输入、尾块、BF16 误差；已有融合不重复实现 |
| RoPE / KV 写入 | 改善布局访问；证据充分再考虑融合 | position、cache slot、边界、原地行为与 graph 兼容 |
| GEMM | 对真实 M/N/K/stride 调 block、warp、stage 或已有厂商实现 | prefill/decode 各自收益；不假设 Triton 总比厂商库快 |
| Attention | GQA 复用、分块、适用时的分割归约参数 | 因果 mask、变长、paged cache、长上下文数值与额外归约成本 |
| LM head / sampler | 热点明确时减少冗余转换、同步或调度 | 完整 logits/采样语义，不裁词表、不改采样策略 |

- [ ] 先从实际调用函数反查 kernel，确认修改会被 FL 命中。
- [ ] 复用 FlagGems 对应 `tests/` 和 `benchmark/`，补充本模型的真实 shape、stride、边界输入；关注两种平台的实现差异。
- [ ] 微基准完成预热、设备同步和重复计时，隔离编译与数据构造时间；记录误差和耗时分布。
- [ ] 单算子正确后接回完整模型，验证真实调用、端到端收益和精度。
- [ ] 仅在频繁 shape 上有证据时增加特化；对其他 shape 保留正确通用路径，避免大量难维护分支。

**产出：** 一到两个有源码 diff、数值测试、微基准和端到端结果的优化，而不是一组未经验证的 kernel 改动。

**通过条件：** 正确性通过、实际路径命中、端到端净收益成立。单算子加速但端到端无收益时，不默认纳入最终配置。

### P4：组合优化与消融

- [ ] 对比 baseline、仅参数优化、仅算子 A、仅算子 B、参数+A+B。
- [ ] graph 和融合可能互相影响，组合版本重新测试，不能将各项加速百分比直接相加。
- [ ] 对短、中、长输入及官方负载检查退化；按照官方评分汇总方式决定保留组合。
- [ ] 每种硬件保存独立配置，共享公共代码；厂商特化限制在相应后端。

**产出：** 两个平台最终候选配置、消融表、退化场景说明。

### P5：官方验收与复现交付

- [ ] 执行完整官方精度与吞吐评测，保留所有原始输出。
- [ ] 在干净的赛事兼容环境验证安装和运行，避免依赖本地未提交文件或旧编译缓存。
- [ ] 冻结两个仓库的 commit、镜像、权重标识和启动命令。
- [ ] 整理“瓶颈证据 → 修改 → 正确性 → 微基准 → 端到端收益”的说明。
- [ ] 仅向个人 fork 保存代码；不创建 PR。

**通过条件：** 两台机器分别满足官方精度阈值和运行约束，结果可以按照记录复现。

## 5. 代码落点

以下 FlagGems 路径已按本地 `v5.3.5` 检查；未切换工作区前，直接打开文件显示的仍可能是 master 版本，可使用 `git show v5.3.5:<路径>` 查看比赛基线。

| 层级 | 入口 | 职责 |
|---|---|---|
| FL 调度 | `vllm_fl/dispatch/config/metax.yaml`、`iluvatar.yaml` | 平台算子顺序与回退 |
| FL 归一化接入 | `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` | 调用 `gems_rms_forward` |
| FL 激活接入 | `vllm_fl/dispatch/backends/flaggems/impl/activation.py` | 拆分 gate/up，调用 `gems_silu_and_mul` |
| FL attention | `vllm_fl/dispatch/backends/flaggems/impl/attention.py` | attention 与 KV cache 接入；实际后端可能不同 |
| FlagGems 模块 | `src/flag_gems/modules/normalization.py`、`activation.py` | 模块接口、C 扩展或 Python 路径选择 |
| FlagGems Norm | `src/flag_gems/ops/rms_norm.py`、`src/flag_gems/fused/fused_add_rms_norm.py` | 归一化 kernel 候选 |
| FlagGems 激活/RoPE | `src/flag_gems/fused/silu_and_mul.py`、`rotary_embedding.py` | 已有融合与位置编码 |
| FlagGems KV 写入 | `src/flag_gems/fused/reshape_and_cache_flash.py` | KV cache 写入候选 |
| FlagGems GEMM | `src/flag_gems/ops/mm.py`、`addmm.py` | 通用矩阵乘实现 |
| 厂商 GEMM | `src/flag_gems/runtime/backend/_metax/ops/mm.py`、`_iluvatar/ops/mm.py` | 厂商覆盖实现 |
| 厂商调参 | `src/flag_gems/runtime/backend/_metax/tune_configs.yaml`、`_iluvatar/tune_configs.yaml` | 对应后端的已有调参配置 |
| 测试/微基准 | 如 `tests/test_fused_add_rms_norm.py`、`benchmark/test_fused_add_rms_norm.py`、`benchmark/test_mm.py`、`benchmark/test_silu_and_mul.py` | 优先扩展已有覆盖 |

Norm 模块可能走 C 扩展路径，厂商后端也可能覆盖通用 kernel。仅修改通用 Python 文件不保证运行会命中，必须沿实际调用链确认。

## 6. 两种硬件的侧重点

| 平台 | 已确认的默认软件差异 | 优先验证 |
|---|---|---|
| 天数 BI-V150 | FL 配置的 attention 优先 FlagOS，存在厂商回退路径 | 实际 attention 路径、Triton/FlagTree 兼容、graph、decode 小 kernel 与 GEMM |
| 沐曦 C500 | FL 配置的 attention 固定首选 `vendor:metax`，其他部分算子优先 FlagGems | MACA attention 是否热点、FlagGems GEMM/Norm/激活、并发与 KV cache 利用 |

沐曦若未走 FlagGems attention，优化该 kernel 不会自动带来收益；切换后端必须满足规则并重新验证。64 GB 是容量信息，不能据此承诺某个并发数或吞吐水平。

## 7. 实验记录与取舍

每个实验至少记录：实验 ID、平台、两个仓库 commit、模型 revision、配置 diff、假设、实际 kernel、精度结果、重复测量结果、相对基线收益、是否保留及原因。

| 实验组 | 目的 | 对照 |
|---|---|---|
| B0 | 原始 BF16 基线 | 每个平台独立保存 |
| C1 | 仅改 batch/调度 | B0 |
| C2 | 合法 graph/执行模式 | 固定 batch 的 eager |
| K1/K2 | 单个热点算子优化 | 同环境未改算子版本 |
| F1 | 最佳合法组合 | B0，以及 C1/C2/K1/K2 消融 |

同一场景至少三个有效重复测量，报告中位数和范围；小幅收益接近噪声时交替运行基线/候选复核。以官方计时、预热和汇总规则为准，不用 profiler 开销、请求长度变化或共享服务器负载差异解释为优化收益。

用于估算投入回报：若某算子占总时间比例为 p，该算子加速 s 倍，理想整体加速上限约为 `1 / ((1-p) + p/s)`，实际还受调度和重叠影响。占比很低的算子即使微基准漂亮，也应低于真正热点的优先级。

## 8. 第一轮执行范围

第一轮只承诺交付以下内容，不承诺未测量的吞吐提升：

1. 两套基线和环境记录。
2. 两张热点表与一组参数/执行模式对照。
3. 每个平台选定一个值得修改的算子，或明确证据显示暂不需要修改 kernel。
4. 至少一个候选的正确性、端到端收益和保留/放弃结论。

暂不优先投入多卡 TP、MoE/MLA/稀疏注意力、量化和大规模自动搜索：当前模型与首轮单卡目标不能直接证明这些投入必要。若官方规定多卡或 profile 出现新瓶颈，再调整路线。

## 参考

- [baseline 执行计划](2026-09-20-minicpm5-dual-vendor-baseline.md)
- [比赛推理分支](https://github.com/flagos-ai/vllm-plugin-FL/tree/flagos-2026-s2)
- [FlagGems v5.3.5](https://github.com/flagos-ai/FlagGems/tree/v5.3.5)
- [模型配置](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json)

本文只规划后续优化；未改变任何运行配置、算子源码、FlagGems 分支或 Git remote，也未在服务器运行实验。
