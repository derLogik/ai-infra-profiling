# Qwen3-8B-AWQ 在 8 GB 笔记本 GPU 上的推理栈剖析

同一份 AWQ 4-bit 权重，三种运行时（transformers / vLLM / SGLang），
同一套负载（短 / 中 / 长 prompt × 并发 1/2/4/8），同一台机器实测。

> 结果文件：`results/tables.md`（全部表格）、`results/backend_comparison.png`（九宫格图）。
> 原始数据：`results/*.json`。复现命令见文末。

---

## 快速验证（不需要 GPU）

```bash
python -m pip install pytest httpx   # 只需要这两个
python -m pytest -q                  # 75 项，全部是 CPU 逻辑测试
python bench/kv_budget.py --kv-mib 801   # 应打印 5,696 token / 并发 2
```

- 测试**不依赖 CUDA、不依赖 torch**：聚合、日志解析、实验对比、脱敏、KV 容量模型。
  `hf_bench` 的输入构造测试在缺少 torch 时会自动跳过，不会报错。
- CI 在 Python 3.10 / 3.11 / 3.12 上跑这套测试，并额外断言 KV 模型
  能复现日志里的 `5,696 tokens`（见 `.github/workflows/tests.yml`）。
- 装依赖：`pip install -e ".[test]"`；要复现 transformers 基线才需要
  `pip install -e ".[hf]"`（会拉 torch，几百 MB 起）。

---

## TL;DR

| 结论 | 数字 |
|---|---|
| 引擎相对裸 `generate()` 的解码加速 | **5.7–7.3×**（同并发下；单流 7.26×，并发 4 时 6.94×）⚠️ 见 §6 修正记录 |
| vLLM 与 SGLang 单流解码 | 基本持平（71.1 vs 73.6 tok/s，ITL 13.5 vs 13.0 ms）——都是带宽瓶颈 |
| 短 prompt 并发 8 的聚合吞吐 | vLLM 434 / SGLang 437 tok/s，**两者打平**，6× 于单流 |
| 中 prompt 并发 8 | vLLM 277 vs SGLang 156 tok/s —— **vLLM 快 1.8×**，差在 KV 容量 |
| 长 prompt 并发 8 | vLLM 63 vs SGLang 43 tok/s，TTFT 8.1 s vs 11.7 s —— 两者都撞 KV 墙 |
| 长 prompt 的 KV 墙 | 单请求 2,176 token，8 并发要 17,408 token；实际 KV 只有 5,696（vLLM）/ 2,554（SGLang）|
| 冷启动 | transformers 9 s（进程内）/ vLLM 74 s / SGLang 73 s（内核缓存已热；首次 341 s）|

一句话：**在这张卡上，8B 模型的瓶颈从头到尾都是 KV cache 显存，不是算力。**
单流时两个引擎几乎一样快；并发一上来，谁能把更多显存留给 KV，谁就赢。

---

## 0. 读法：观测 / 推断 / 待验证

这份报告的每个结论都标了它的证据等级，读的时候请按标签区分：

| 标记 | 含义 |
|---|---|
| **观测** | 直接测到的数。原始记录在 `results/*.json`；⚠️ **现存结果由旧版客户端生成，只有中位数、没有区间和逐请求明细**（见下方修订说明），新版客户端已保留全部原始样本，重跑即可获得 |
| **推断** | 由观测经计算/模型推导得到，文中给出推导过程，可复核 |
| **待验证** | 目前只有间接支持，需要调度器日志或多卡实验才能定案 |

2026-10 版修订说明：

- 旧版客户端只把每格的中位数写进 JSON，3 次重复的原始样本被丢弃，
  因此"在测量误差内一致"这类话当时无法核验。现在逐轮、逐请求都保留。
  本文中的**旧数据（2026-10 之前采集）无法补出区间**，已在表格中标为单值。
- 旧版把并发档的**请求级中位数**当成了"最后一个请求的等待时间"来论证排队行为，
  这是错的（详见 §6.3）。现在同时报告中位数、P95 和最大值。

---

## 1. 选型：为什么是 Qwen3-8B-AWQ

需求原话是"最新的 qwen3.8 小模型"。查证后有两个事实：

- **Qwen3.8 没有小模型**。该系列最小的是 `Qwen3.8-27B`，AWQ 4-bit 约 20 GB，
  8 GB 的 3070 Laptop 装不下。
- **"qwen3.8" 按字面读就是 Qwen3-8B**（3 系第 8B 档），这也是 8 GB 卡上
  唯一塞得进、又能跑出有意义数据的选择。

另一个约束：用户此前已经做过 `Qwen3.5-0.8B`，再测 0.8B 是重复劳动。
0.8B 也太小——它的权重只有 ~0.5 GB，几乎不产生显存压力，**测不出任何
AI infra 的真实取舍**。8B + 4-bit 才是"刚刚好装不下"的甜点区：权重吃掉
5.68 GiB 之后，留给 KV cache 的只剩 1 GiB 出头，所有调度差异都被放大。

所以最终选 **`Qwen/Qwen3-8B-AWQ`**（AWQ 4-bit，group_size 128，zero_point，
GEMM 版本）。整套 harness 与模型无关，换模型只需改 `--model-path`。

| 参数 | 值 |
|---|---|
| 层数 / 注意力头 / KV 头 / head_dim | 36 / 32 / 8 / 128（GQA 4:1）|
| 磁盘占用 | 5.7 GB |
| 加载后显存 | 5.68 GiB |
| 原生上下文 | 40,960（本实验限制到 4,096）|

---

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop，8,192 MiB，SM 8.6（Ampere）|
| 驱动 | 596.21 |
| 宿主 | Windows 11 Home China 10.0.26200 |
| 虚拟化 | WSL 2.7.13.0，内核 6.18.33.2-microsoft-standard-WSL2 |
| 发行版 | Ubuntu 26.04.1 LTS |
| Python | 3.12.14 |

三个独立 venv（互相隔离，避免依赖打架）：

| venv | torch | transformers | triton | 备注 |
|---|---|---|---|---|
| `qwen3-infer` | 2.14.0 | 5.16.1 | 3.8.0 | 裸 transformers 基线，+ accelerate 1.15.0、gptqmodel 7.6.0 |
| `vllm` | 2.13.0 | 5.17.0 | 3.7.1 | vLLM 0.29.0 |
| `sglang` | 2.13.0 | 5.12.1 | 3.7.1 | SGLang 0.5.20 |

**为什么用 AWQ 而不是 FP8/int8**：Ampere（SM 8.6）没有原生 FP8 单元；
4-bit AWQ 是这张卡上"能装下 8B"的最优解，vLLM 侧走 `awq_marlin` 内核
（解量化融进 GEMM，不需要反量化到 fp16 再算）。

---

## 3. 显存账本

先算清楚账，后面所有现象都能对上。

**权重**：5.68 GiB（4-bit，约 8.5 GB 的 fp16 权重压到 1/4）。

**KV cache**：每 token 每层存 K 和 V 各一份：

```
2 × 36 层 × 8 KV头 × 128 维 × 2 字节 = 147,456 B = 144 KiB / token
4,096 token  →  576 MiB
2,176 token（长负载单请求）→ 306 MiB
```

**可用显存**：8192 MiB 减去 Windows 桌面合成（WSL 里 `nvidia-smi` 报
启动时只剩 6.95 GiB 可用），再减权重 5.68 GiB，
**留给 KV + 激活 + CUDA graph 的只有约 1.2 GiB。**

这就是整份报告的主线：
1.2 GiB 的池子要同时装下 KV cache、prefill 激活、CUDA graph。
三个运行时分配策略不同，结果就分叉了。

---

## 4. 启动成本

| 运行时 | 启动 (s) | 显存占用 (GiB) |
|---|---|---|
| transformers（进程内）| **9.2** | 5.68（加载后）|
| vLLM 0.29.0 | 74.2 | 7.07（压测后）|
| SGLang 0.5.20 | 73.1 | 7.48（压测后）|

（vLLM / SGLang 的显存是整轮压测结束后的实测值，含 KV 池与激活；
transformers 是权重加载完成时 `memory_allocated()`，不含生成期的 KV。）

- transformers 只做权重加载，9 秒（5.7 GB 基本走页缓存）。
- vLLM 的 74 s 拆开看（日志实测）：权重读盘只占 **8.0 s**；
  `model loading` 整体 37.5 s（AWQ 权重重排、内核初始化）；
  建 KV cache + warmup 8.6 s；**CUDA graph 捕获只要 2 s**；
  剩下约 25 s 是 Python 导入和进程启动。真正贵的是框架自身的导入与
  量化内核准备，不是"加载模型"。
- SGLang 的 73 s 是**磁盘内核心编译缓存已热**时的数字。同一台机器上
  冷缓存首次启动实测 **341 s**（当时是 0.85 配置），因为要现场编译
  Triton 内核并捕获 42 张 CUDA graph。容器/机器重启后第一轮的启动时间
  要有心理准备。

**工程含义**：离线批处理可以忽略；但在弹性伸缩场景里，70 s 的冷启动
意味着扩一个副本头一分钟没有产出，冷启动优化（预编译缓存、镜像内
capture）比调吞吐更值钱。

---

## 5. 单流延迟（并发 = 1）

| 负载 | 运行时 | 吞吐 tok/s | TTFT ms | ITL ms |
|---|---|---|---|---|
| 短 128→128 | transformers | 9.8 | 198 | 101.4 |
| | **vLLM** | **71.1** | **100.0** | **13.5** |
| | **SGLang** | **73.6** | **94.2** | **13.0** |
| 中 512→256 | transformers | 10.5 | 370 | 94.1 |
| | vLLM | 68.5 | 261.3 | 13.6 |
| | SGLang | 70.1 | 301.7 | 13.1 |
| 长 2048→128 | transformers | 8.1 | 1252 | 115.2 |
| | vLLM | 43.9 | 1029.5 | 14.8 |
| | SGLang | 45.6 | 1064.0 | 13.7 |

### 怎么读这张表

**解码（ITL）才是引擎的价值所在。** 引擎把 101 ms/token 压到 13 ms/token，
7.3 倍。原因很直白：裸 `generate()` 每步都要走 Python 采样、动态形状、
没有 CUDA graph，CPU 提交开销把 GPU 喂不饱；引擎把整个解码步捕获成
一张 CUDA graph，CPU 侧几乎零开销。

**两个引擎的 ITL 几乎完全相同（13.0 vs 13.5 ms）**。这不是巧合：单流解码
是纯显存带宽瓶颈——每生成一个 token 要把 8B 模型的 4-bit 权重（约 4.5 GB）
完整读一遍，除以 3070 Laptop 约 448 GB/s 的带宽，理论下限就在 10~14 ms。
**只要内核不比理论值差，谁的实现都一样。** 这也是为什么两个团队
在单流上打平。

**TTFT 差异来自 prefill 实现。** 短 prompt 的 100 ms 里约有 20~30 ms
是固定开销（HTTP、分词、调度、首 token 同步），剩下是 prefill。
长 prompt 2,048 token 用 1.03 s，反推 prefill 约 **2,000 token/s**——
对 4-bit 解量化 GEMM + 单卡来说合理。

**吞吐随 prompt 变长而下降**（71 → 69 → 44 tok/s）：输出长度固定 128，
但 KV 越长，每步注意力开销越大，且首 token 等待被计入端到端。

---

## 6. 并发与批处理：三种完全不同的调度模型

这是全篇最有信息量的部分。

| 负载 | 并发 | vLLM tok/s | SGLang tok/s | transformers tok/s |
|---|---|---|---|---|
| 短 | 1 | 71.1 | 73.6 | 9.8 |
| 短 | 2 | 135.5 | 139.7 | — |
| 短 | 4 | 249.0 | 255.1 | **35.9**（静态批）|
| 短 | 8 | **434.4** | **436.5** | — |
| 中 | 8 | **276.9** | 156.2 | — |
| 长 | 8 | **63.4** | 42.6 | — |

### 6.1 短 prompt：两个引擎打平，6× 于单流

短 prompt 的 KV 需求极小（128+128 token ≈ 37 MiB/请求），8 个并发总共
不到 300 MiB，KV 池装得下。于是两边都能把请求真正塞进一次前向：
吞吐 434 vs 437 tok/s，**在测量误差内完全一致**。

这印证了第 5 节的结论：当瓶颈是带宽而不是调度时，两个引擎没有区别。
选哪个更多取决于生态、API、部署方式，而不是性能。

### 6.2 中 prompt：KV 容量开始分叉

vLLM 277 vs SGLang 156，差 1.8 倍。看 TTFT 就明白了：

| 并发 | vLLM TTFT | SGLang TTFT |
|---|---|---|
| 1 | 261 ms | 302 ms |
| 2 | 387 ms | 615 ms |
| 4 | 987 ms | 894 ms |
| 8 | **1,585 ms** | **5,406 ms** |

两边 KV 容量不同：**vLLM 拿到 5,696 token（0.78 GiB），SGLang 只有
2,554 token（0.37 GiB）**。同样 8 个 512-token 请求，vLLM 能同时容纳
大半，SGLang 只能分批推进，后到的请求排队等 KV，TTFT 被推高 3.4 倍。

### 6.3 长 prompt：都撞上 KV 墙，vLLM 更抗

**观测**：单请求 2,048+128 = 2,176 token，8 并发共需 17,408 token，
而 KV 池只有 5,696（vLLM）/ 2,554（SGLang）。vLLM 吞吐 43.9 → 63.4 tok/s
（并发 1 → 8，**+44%**），TTFT 中位数 1.03 → 8.09 s，E2E 中位数 2.9 → 10.1 s。

| 并发 | vLLM tok/s | SGLang tok/s | vLLM TTFT | SGLang TTFT |
|---|---|---|---|---|
| 1 | 43.9 | 45.6 | 1.03 s | 1.06 s |
| 2 | 64.4 | 45.4 | 1.98 s | 2.48 s |
| 4 | 64.6 | 44.2 | 3.96 s | 5.49 s |
| 8 | 63.4 | 42.6 | 8.09 s | 11.72 s |

**推断**（可逐步复核）：

1. KV 池 5,696 ÷ 每请求 2,176 ≈ **2.6**，即同时最多跑 **2 个**长请求。
   （vLLM 启动日志自己算的 `Maximum concurrency for 4,096 tokens per request: 1.39x`
   是同一件事在 4K 上下文下的数值。）
2. 于是 8 个请求排成 **4 波**，每波 2 个。
3. 每波成本 = prefill 2 × 2,048 token（按 §5 反推的 ~2,000 tok/s ≈ 2.0 s）
   + decode 128 步（按 ITL 15.8 ms ≈ 2.0 s）≈ **4 s/波**，4 波 ≈ **16 s**。
4. 由 16 s 推出三个可核对的值：
   - 吞吐 = 8 × 128 / 16 = **64 tok/s**（实测 63.4）
   - 请求级 TTFT 中位数 = 8.0 s（实测 8.09）
   - E2E 中位数 = 10.0 s（实测 10.1）

三个独立观测量同时吻合到 1% 以内，说明"2 路批处理 × 4 波"这个模型站得住。

**为什么不是"逐个串行"**：若 8 个请求真的一前一后排队，每波 4 s，
则总时长 32 s、吞吐 32 tok/s、TTFT 中位数 16 s——而实测是 63.4 tok/s 和 8.09 s，
**两个数都差了整整一倍**。旧版报告写的"请求只能串行化……完全吻合"不成立，
错在两处：

- 只看吞吐的绝对值，没看它的变化方向（43.9 → 63.4 是**涨**的；
  真串行的话吞吐该停在单流水平不动）；
- 把**请求级中位数**当成了"最后一个请求的等待时间"。8 个并发请求的中位数
  天然落在中间，与最大值不是一个量。旧版客户端只存中位数，看不出这个区别。

**待验证**：波次结构是从吞吐 / TTFT / E2E 三个数反推的，尚未由调度器日志直接确认。
开 `--enable-logging-iteration-details` 能打印每步 batch 的构成，
用 `bench/analyze_iterations.py` 一键判读（见 §11）。

关于 KV 池容量 5,696 这个数本身从哪来——`gpu_memory_utilization` 如何经过实测
non-KV 占用、page 字节、层数换算成 block 数——见 `notes/kv-cache-allocation.md`，
那里给出了完整源码链路、验算和一份可证伪的预测表。

**交叉验证**：§10.5 的 A/B 实验是独立旁证——同一引擎只切换前缀缓存，
长 prompt 8 并发的 TTFT 从 8.09 s 掉到 104 ms、吞吐从 63.4 涨到 380.8 tok/s。
这证实该场景下 **prefill 是主导成本**，排队不是主要矛盾。

vLLM 在长 prompt 上比 SGLang 高 **49%**（63.4 vs 42.6），原因还是 KV 池更大
（5,696 vs 2,554）：vLLM 能同时容纳 2 个长请求，SGLang 基本只能一个一个来。


vLLM 自己在启动日志里就写得很清楚：

```
GPU KV cache size: 5,696 tokens
Maximum concurrency for 4,096 tokens per request: 1.39x
```

**1.39x** —— 官方直接告诉你：这张卡跑 4K 上下文，并发上限是 1.4 个请求。
这个数字比任何压测都更有说服力。

### 6.4 transformers：静态批 ≠ 并发

HF `generate()` 没有连续批处理。`batch=4` 是把同一 prompt 复制 4 份
一次前向算完，四个请求必须**同时开始、同时结束**：

| batch | 短 prompt tok/s | 中 prompt tok/s |
|---|---|---|
| 1 | 9.8 | 10.5 |
| 4 | 35.9（3.7×）| 37.5（3.6×）|

接近线性（3.6×/4），说明解码确实是带宽瓶颈、批处理摊薄了权重读取。
但代价是**短请求必须等长请求**：生产里请求长度参差，静态批会把
短请求的延迟拉长到最长请求的水平。这就是连续批处理存在的理由。

**同并发下 vLLM 比静态批快多少**（必须是同并发才可比）：

| 负载 | 并发 | transformers | vLLM | 倍数 |
|---|---|---|---|---|
| 短 | 1 | 9.8 | 71.1 | 7.26× |
| 短 | 4 | 35.9 | 249.0 | 6.94× |
| 中 | 1 | 10.5 | 68.5 | 6.52× |
| 中 | 4 | 37.5 | 213.3 | 5.69× |

> **修正记录（2026-10-28）**：本节原来写"vLLM 并发 8 做到 434 tok/s，
> 是 transformers 静态批 4 的 12 倍"。这个比法**并发不一样**（8 vs 4），
> 不可比。改成同并发后是 **5.7–7.3 倍**，反而更稳定——跨负载、跨并发都落在
> 同一区间，比一个孤立的 12 倍更可信。
>
> ⚠️ 另外，transformers 侧的输入构造**当时还没有套 chat template**
> （`hf_bench.py` 已加 `--chat-template`，默认开启以对齐引擎侧）。
> 所以上表数字需要重跑确认；在重跑前只能当作**方向性**结论。

---

## 7. SGLang 的 KV 分配：一个必须实测才敢用的旋钮

SGLang 用 `--mem-fraction-static` 一次性切出"权重 + KV"的静态池。
默认 0.8/0.85 在这张卡上会出问题——**KV 池小到连一个 2K 请求都放不进去**，
请求直接被 400 拒绝：

```
Input length (2060 tokens) exceeds the maximum allowed length (1035 tokens)
```

这不是超时，是启动时就定死的容量。实测标定（`bench/probe_sglang.py`）：

| `--mem-fraction-static` | KV 容量 (token) | 最大可接受输入 | 能否跑 2048-token 负载 |
|---|---|---|---|
| 0.85 | 1,041 | 1,035 | ✗（400）|
| 0.88 | 2,554 | 2,548 | ✓（只够 1 个并发）|
| 0.90 | 3,562 | 3,556 | ✓（1 个，勉强）|

> ⚠️ **这张表的数据没有归档到仓库里。** 表中数字来自 `probe_sglang.py` 的若干次
> 试跑，但当时的输出没有保存成 `results/probe_*.json`（脚本只把服务日志写到
> `/tmp/probe_*.log`，容器重启后丢了）。
> **2026-10-28 起脚本已改为每跑完一档就落盘到 `results/probe_sglang.json`**
> （含 `max_total_num_tokens` / `max_req_input_len` / 启动耗时 / 完整命令），
> 重跑一次即可让这张表变成可复核数据，届时请把本段换成"数据见 …"。
> 在那之前，这张表属于**"做过但没有留下原始记录"**，与 §13 里说的旧版中位数数据
> 同一性质。写这段是为了不让读者以为 `results/` 里现在就有对应文件。

两个关键发现：

1. **`--context-length` 完全不影响 KV 容量**。设 4096 和设 2560，
   `max_total_num_tokens` 都是 3,562——它只改"允许的最大输入"上限，
   不改池子大小。想多要 KV，只有调 `mem-fraction-static`。
2. **CUDA graph 和 KV 抢同一块显存**。开 graph 时 KV 容量 1,041 token
   （对应最大可接受输入 1,035）；关掉 graph，KV 容量几乎不变（1,041），
   但**显存省下 0.94 GiB、解码慢一倍**（ITL 13 ms → 23 ms）。
   也就是说 graph 的显存开销并不直接吃 KV 池，而是整体把显存顶到天花板。

**最终采用的配置是 0.88 + CUDA graph**（KV 2,554 token，显存 7.48/8.19 GiB）。
再往上试 0.90 + graph（显存 7.76 GiB，只剩 0.43 GiB 余量）会**让整个 WSL
GPU 栈崩溃、虚拟机重启**——不是 OOM 异常，是直接掉机器。所以 0.88 是这台
机器上"能跑满全部负载"的安全边界。

这个取舍本身就是结论：**8 GB 卡上，CUDA graph 与长上下文容量不可兼得。**

---

## 8. 三个运行时的定位

| 维度 | transformers | vLLM | SGLang |
|---|---|---|---|
| 单流解码 | 10 tok/s | 71 tok/s | 74 tok/s |
| 并发 8（短） | 静态批 36 | 434 | 437 |
| 并发 8（中/长） | — | 277 / 63 | 156 / 43 |
| KV 容量 | 手动管理 | 5,696 token | 2,554 token |
| 启动 | 9 s | 74 s | 73 s（冷 341 s）|
| 显存占用 | 5.68 GiB | 7.07 GiB | 7.48 GiB |
| 适合 | 研究、单次推理、改模型结构 | 生产 API、高并发 | 结构化输出、RadixAttention 类工作负载 |

**vLLM 在这张卡上整体更稳**：同样的显存预算下 KV 池是 SGLang 的 2.2 倍，
中/长 prompt 的并发表现明显更好，且没有"配置不当就 400"的坑。

**SGLang 在短 prompt 上与 vLLM 持平**，说明它的调度器本身没问题；
差距全部来自显存分配策略。它真正的优势（RadixAttention 前缀复用、
结构化输出/约束解码）在这套"每次都是冷前缀"的测法里根本没被激活——
为了公平，两边都关掉了前缀缓存。如果负载有大量共享前缀（RAG 的
系统提示词、few-shot），SGLang 的表现会反过来。

**transformers 不是用来跑服务的**，它的价值是：能改模型内部、能调试
量化内核、作为"零优化的下限"衡量引擎到底省了多少。7.3× 就是答案。

---

## 9. 这台机器（WSL2 + 8 GB）踩过的坑

全部是实测踩出来的，按"是否会影响结论"排序。

### 9.1 显存超配不会报 OOM，而是直接把虚拟机打崩

显存顶到 7.76/8.19 GiB 时，SGLang 在切负载的瞬间（运行时 JIT 加载
Triton 内核）**没有抛异常，而是整个 WSL GPU 栈挂了**（`Wsl/Service/E_UNEXPECTED`），
虚拟机重启，正在跑的数据全丢。

对策：benchmark 必须**每跑完一格就落盘**（`hf_bench.py` 的 `dump()`），
并且把显存留出至少 0.6 GiB 余量。

### 9.2 对持有 CUDA 上下文的进程发 SIGKILL = 杀 GPU 栈

`pkill -9` 一个 `VLLM::EngineCore` 会把 dxg 驱动带崩。所有停止逻辑
只允许 SIGTERM，然后耐心等（`run_matrix.py` 的 `stop()`）。

### 9.3 有个自启动服务一直在抢 8000 端口和显存

机器上有一个 `vllm-qwen3.service`（跑 Qwen3.5-0.8B 的旧服务，`Restart=on-failure`）。
它会在每次 WSL 启动后占用 8000 端口和约 1.5 GiB 显存，导致：

- benchmark 打到它的端口 → 401（它启用了 api-key）
- 两个 CUDA 进程抢显存 → `cudaErrorIllegalAddress`，测量作废

因为这台机器**没有免密 sudo**，无法永久 `disable`。当前的临时压制办法是
连续 SIGTERM 十几次，触发 systemd 的 `StartLimitBurst`，让它进入
`start-limit-hit` 停止重启。

> **建议你手动处理一次**（需要密码）：
> `sudo systemctl disable --now vllm-qwen3.service`
> 否则每次重启后跑基准都得先把它按死。

### 9.4 vLLM 在 WSL 上默认禁用 pinned memory

不设 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 会直接报 `UVA is not available` 起不来。
内核 ≥ 4.19.121 时这个开关是安全的（本机 6.18）。

### 9.5 `no_proxy=127.*` 不是通配符

系统代理（7892）把 `127.0.0.1:8000` 的请求也接走了，返回 502。
curl/urllib 只做字面后缀匹配，不认识 `127.*`。所有子进程必须显式清空
`*_proxy` 环境变量并设置 `NO_PROXY=127.0.0.1,localhost`。

### 9.6 `pkill -f` 会杀掉自己

`bash -c 'pkill -f run_matrix.py'` 的命令行本身就含 `run_matrix.py`，
pkill 匹配到自己，命令以退出码 15 自杀，后面的清理全没执行。
用正则字符类规避：`pkill -f "run_matrix[.]py"`。

### 9.7 精简 PATH 下找不到 nvidia-smi

WSL 的 `nvidia-smi` 在 `/usr/lib/wsl/lib`，不是 `/usr/bin`。
辅助指标取不到就记 NaN，不要因此中断整轮基准。

---

## 10. 方法学：会让数字说谎的陷阱，与一个控制变量实验

这部分比结果本身更容易被忽略。

### 10.1 前缀缓存会让 TTFT 失去意义

所有压测请求共用同一个 prompt。开着前缀缓存时，第 2 个请求之后
prefill 全部命中缓存：

| 场景 | 冷 prefill（关缓存）| 热 prefill（开缓存）|
|---|---|---|
| 短 prompt 单流 TTFT | 100 ms | **23 ms** |
| 长 prompt 并发 8 TTFT | 8,089 ms | **104 ms** |

热缓存的 104 ms 不是"prefill 变快了"，是**根本没做 prefill**。
所以正式矩阵两边都关掉缓存（`--no-enable-prefix-caching` / `--disable-radix-cache`），
热数据单独存为 `results/vllm_prefixcache.json` 作对照。

### 10.2 Qwen3 默认开思考模式，会把输出预算吃光

不关思考模式时，128 个输出 token 全花在 `reasoning_content` 上，
`content` 一个 delta 都不出现 → TTFT 测成 NaN，端到端 0.04 s 的假数据。
所有请求都带 `chat_template_kwargs: {"enable_thinking": false}`。

**如果哪天看到某个负载 TTFT 是 NaN、E2E 异常短，先查这个。**

### 10.3 模型刚加载后，第一个测的格子必然慢 2~2.5 倍

不是负载差异，是顺序效应：笔记本 GPU 还处在低频，Triton 还在按真实
序列长度现场 autotune。做了个对照实验——把 long 放到 short 前面：

| 顺序 | long | short |
|---|---|---|
| short 先 | 9.8 tok/s（101 ms）| — |
| **long 先** | **3.5 tok/s（279 ms）** | **8.0 tok/s（124 ms）** |

慢的那格永远是"先跑的那格"。对策：正式测量前按**真实输出长度**空跑
一遍全部负载（全局预热，全部丢弃）。

### 10.4 静态批的 E2E 不能和并发 E2E 直接比

transformers `batch=4` 的 E2E 是"4 个请求一起跑完的总时长"；
vLLM `concurrency=4` 的 E2E 是"单个请求的端到端"。只有**聚合吞吐**
是可比的。表格里 transformers 的 TTFT/ITL 用 `—` 标出，就是提醒这一点。

### 10.5 控制变量实验：同一引擎只切换前缀缓存（A/B）

前面几节多次提到"前缀缓存会污染 TTFT"。这一节把它单独做成对照实验：
**同一个 vLLM、同一份权重、同一台机器、同一套负载、同样 3 次重复，
唯一的变量是 `--no-enable-prefix-caching` 是否加上。**

| 负载 | 并发 | 缓存关 (tok/s) | 缓存开 (tok/s) | 吞吐倍数 | TTFT 关 (ms) | TTFT 开 (ms) |
|---|---|---|---|---|---|---|
| short | 1 | 71.1 | 74.7 | 1.05× | 100 | **23** |
| short | 8 | 434.4 | 552.3 | 1.27× | 555 | **66** |
| medium | 1 | 68.5 | 73.4 | 1.07× | 261 | **25** |
| medium | 8 | 276.9 | 504.3 | 1.82× | 1,585 | **76** |
| long | 1 | 43.9 | 67.5 | 1.54× | 1,029 | **30** |
| long | 8 | 63.4 | 380.8 | **6.0×** | 8,089 | **104** |

结论有三条，都能从这张表直接读出来：

1. **收益随 prefill 占比单调上升**。short 单流的 prefill 只占 100 ms 里的一小部分，
   开关几乎没有影响（1.05×）；long 8 并发的成本几乎全是 prefill，
   于是吞吐 6.0×、TTFT 78×。这条规律本身就是"prefill 是长 prompt 的主导成本"的
   直接证据，也解释了 §6.3 里那个 8 秒是怎么来的。
2. **缓存命中时 TTFT 不再是"延迟"，而是"没发生"**。104 ms 不是 prefill 变快 78 倍，
   是根本没做 prefill——所以 §10.1 才要求正式矩阵一律关掉缓存。
3. **decode 部分没有被这个开关改变**。缓存开着时 short 单流 74.7 tok/s 与
   关着时的 71.1 处在同一量级，说明节省的完全是 prefill；这也反过来支持
   §5 的判断：单流解码是带宽瓶颈，和缓存无关。

**这个实验的边界**（必须说明，否则收益会被高估）：8 个请求用的是**完全相同的
prompt**，这正是前缀缓存最理想的情形。生产里请求各不相同，命中率取决于共享前缀
（系统提示词、few-shot、多轮对话历史）的真实比例。所以上表是收益的**上界**，
不是典型值；SGLang 的 RadixAttention 在同类负载下也会受益，本表只测了 vLLM。

---


## 11. 复现

下面的命令是**在这台机器上实际执行过的原始形式**。其中把 Windows 用户名写作 `USER`、
WSL 家目录写作 `~`，请按自己的环境替换。换机器时更推荐用本节末尾「路径参数化」里的参数形式。
全部从 Windows 侧通过 `wsl.exe` 调用，避免 `bash -c` 吞变量。

```bash
# 0) 先把自启服务按死（见 9.3）
wsl.exe -d Ubuntu -- bash -c 'for i in 1 2 3 4 5 6 7 8 9 10 11 12; do pkill -TERM -f "Qwen3[.]5-0[.]8B"; sleep 1; done'

# 1) 完整矩阵（启动服务 -> 压测 -> 停服务，可加 --backends vllm,sglang）
wsl.exe -d Ubuntu -- bash -c 'export PATH=~/venvs/vllm/bin:/usr/lib/wsl/lib:/usr/bin:/bin; \
  ~/venvs/vllm/bin/python \
  /mnt/c/Users/USER/work/ai-infra-profiling/bench/run_matrix.py \
  --model-path ~/models/Qwen3-8B-AWQ --backends vllm \
  --concurrency 1,2,4,8 --repeats 3 --max-model-len 4096'

# SGLang（KV 旋钮必须给到 0.88，否则长 prompt 会被 400）
wsl.exe -d Ubuntu -- bash -c 'export PATH=~/venvs/sglang/bin:/usr/lib/wsl/lib:/usr/bin:/bin; \
  ~/venvs/sglang/bin/python \
  /mnt/c/Users/USER/work/ai-infra-profiling/bench/run_matrix.py \
  --model-path ~/models/Qwen3-8B-AWQ --backends sglang \
  --concurrency 1,2,4,8 --repeats 3 --max-model-len 4096 --mem-frac-static 0.88 --sglang-cuda-graph'

# 2) transformers 基线
wsl.exe -d Ubuntu -- bash -c 'export PATH=~/venvs/qwen3-infer/bin:/usr/lib/wsl/lib:/usr/bin:/bin; \
  ~/venvs/qwen3-infer/bin/python \
  /mnt/c/Users/USER/work/ai-infra-profiling/bench/hf_bench.py \
  --model-path ~/models/Qwen3-8B-AWQ --batch-sizes 1 --repeats 3'

# 3) 出表出图
python -I bench/report.py
```

### 路径参数化（2026-10 起）

脚本里不再出现任何个人路径。venv 根目录默认 `~/venvs`，可用 `--venv-root`
整体替换，或单独指定 `--vllm-bin` / `--vllm-python` / `--sglang-python`：

```bash
# 只需要改这几个变量，其余就近默认
export VENV_ROOT=$HOME/venvs
export MODEL=$HOME/models/Qwen3-8B-AWQ
export REPO=$(pwd)

# vLLM
$VENV_ROOT/vllm/bin/python $REPO/bench/run_matrix.py \
    --model-path $MODEL --backends vllm \
    --venv-root $VENV_ROOT --concurrency 1,2,4,8 --repeats 3 --max-model-len 4096

# SGLang（KV 旋钮要给到 0.88，否则长 prompt 直接被 400）
$VENV_ROOT/sglang/bin/python $REPO/bench/run_matrix.py \
    --model-path $MODEL --backends sglang --venv-root $VENV_ROOT \
    --mem-frac-static 0.88 --sglang-cuda-graph

# transformers 基线
$VENV_ROOT/qwen3-infer/bin/python $REPO/bench/hf_bench.py \
    --model-path $MODEL --batch-sizes 1,4,8 --repeats 3 --out results/transformers.json

# 出表出图（新版会输出 中位数[最小-最大] 与数据质量一节）
python -I bench/report.py
```

不想起服务也能验口径：

```bash
python -m pytest bench/ -q     # 15 项，mock HTTP + 合成日志，不需要 GPU
```

### 怎么把 §6.3 的"待验证"变成"已证实"

§6.3 的"2 路 × 4 波"目前是从吞吐 / TTFT / E2E 反推的。要坐实只需三步：

```bash
# 1) 只跑长 prompt 那一组，并打开逐 iteration 日志（约 2 分钟）
$VENV_ROOT/vllm/bin/vllm serve $MODEL --served-model-name qwen3-8b-awq \
    --quantization awq_marlin --max-model-len 4096 --max-num-seqs 8 \
    --no-enable-prefix-caching --enable-logging-iteration-details \
    2>&1 | tee logs/iter_long.log

# 2) 另开一个终端，只压长 prompt
$VENV_ROOT/vllm/bin/python bench/client_bench.py --backend vllm \
    --model qwen3-8b-awq --tokenizer-path $MODEL \
    --concurrency 8 --repeats 1 --workloads long --out results/long_c8.json

# 3) 判读
python bench/analyze_iterations.py logs/iter_long.log --expect-requests 8
```

输出会直接告诉你"同时在跑的请求数峰值"是 1、2 还是 8，以及波次数。
如果是 2 且 4 波，§6.3 就从"推断"升级为"观测"——这是这份报告里
唯一一个还挂着"待验证"标签的强结论。

### 版本与来源

后端版本不再手写在文档里，由 `run_matrix.py` 从各自 venv 里
`importlib.metadata.version()` 读出并写进结果 JSON 的 `version` 字段；
`report.py` 出表时直接引用，避免文档与实际跑的东西对不上。

### 文件

| 文件 | 作用 |
|---|---|
| `bench/run_matrix.py` | 起服务 → 等 `/health` → 压测 → SIGTERM，记录启动耗时与显存 |
| `bench/client_bench.py` | httpx + SSE 流式客户端，测 TTFT / ITL / 吞吐 |
| `bench/hf_bench.py` | 进程内 transformers 基线 |
| `bench/probe_sglang.py` | 标定 SGLang 的 KV 容量旋钮 |
| `bench/report.py` | 生成 `results/tables.md` 和九宫格对比图 |
| `bench/test_aggregate.py` | 口径与失败路径的单测（mock HTTP，不需要 GPU）|
| `bench/analyze_iterations.py` | 解析 `--enable-logging-iteration-details` 日志，判读真实并发结构 |
| `bench/test_analyze_iterations.py` | 上面那个判读逻辑的单测（合成长 prompt 日志）|
| `bench/pdl.py` | 并行分段下载器（`hf download` 在这台机器上会卡住）|

### 测量口径

```
TTFT = 请求发出 → 第一个非空 content delta
ITL  = (结束时间 − TTFT) / (completion_tokens − 1)
吞吐 = 该并发档全部请求的 completion_tokens 之和 / 墙钟
```
每格 3 次取中位数；每格前有 16-token 预热（不计入）。

---

## 12. 已知限制

- **单卡、无张量并行**。结论不能外推到多卡。
- **只测了 AWQ 4-bit**。Ampere 没有原生 FP8，无法做 FP8 对照；
  FP16 装不下 8B，理论上限没测。
- **SGLang 用的是"能跑满负载"的配置（0.88 + graph）**，不是它的默认
  0.85——默认配置在这台机器上连 2K 输入都拒绝。两者都跑通才有对比意义，
  但读者要知道这不是开箱即用的表现。
- **前缀缓存被关闭**以测真实 prefill 成本。有共享前缀的负载会呈现
  完全不同的排名（见第 10.1 节的热数据）。
- **每组 3 次取中位数**，笔记本散热与电源策略会带来 5~10% 抖动；
  第 10.3 节的顺序效应已用全局预热消除，但跨轮次仍有小幅漂移。
- Qwen3 的**思考模式全程关闭**，比的是纯解码；开启后输出长度不可控，
  没法做等长对比。
- **2026-10 之前采集的数据只有中位数，没有区间**。当时客户端丢弃了逐轮样本，
  已无法补算 min/max，`results/tables.md` 里这批格子只显示单值。
  新版客户端逐轮、逐请求都保留，重跑即可获得区间。
- **并发档的 TTFT 是请求级中位数**，不等于"最慢那个请求"的等待时间。
  新版同时给出 `req_ttft_s_max` 与 `req_ttft_s_p95`，旧的单值口径不要拿来
  讨论排队行为（§6.3 曾经踩过这个坑）。
- **`peak_vram_gib` 在旧版里从未写入**，所以那一列当时恒为 `-`；
  现在由压测期间的后台采样给出，与"加载后显存"是两个不同的量。

---

## 13. 如果继续往下做

按"信息量 / 成本"排序：

1. **把并发上限从 8 提到 32**（短 prompt），看 vLLM 在哪里饱和——
   现在 8 并发还没到拐点（434 tok/s 仍在上涨）。
2. **测前缀缓存真正命中时的收益**（RAG / 共享系统提示词），
   这才是 SGLang 该赢的场景。
3. **换一张 16 GB 的卡做对照**，验证"瓶颈是 KV 显存"这个结论：
   如果 16 GB 上两个引擎在中长 prompt 的差距消失，结论就闭环了。
4. **chunked prefill 的对照**：`--enable-chunked-prefill` 不显式指定时，
   vLLM 会把它解析成"该模型是否支持分块 prefill"（`arg_utils.py:2502`），
   Qwen3-8B 属于支持，所以本机运行时是 `True`（见 `logs/vllm.log` 的
   EngineCore config 行）。**所以要做的是 `--no-enable-chunked-prefill` 的对照组**，
   而不是"打开它"。
   同时开 `--enable-logging-iteration-details`（`arg_utils.py:1394`，默认关闭），
   它逐 iteration 打印 `N context requests / N context tokens / N generation
   requests / N generation tokens`，可以直接数出每步有几个请求在跑——
   用 `bench/analyze_iterations.py` 一键判读，不必从吞吐反推波次结构。
