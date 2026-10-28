# KV cache 是怎么算出来的（源码核对笔记）

> 目的：把报告里"KV 池 5,696 token"从一个日志数字，升级成**能从代码推导、能预判、
> 能被追问**的结论。
>
> **版本声明**：本文引用的代码来自本地 `work/vllm_src`，其版本是
> `v0.25.0-21-gfabec87f`（2026-07-10）。压测实际运行的是 vLLM **0.29.0**。
> 两者相差约三个月，**公式和链路是稳定的，但行号可能有偏移**。
> 引用行号时请说明"以 v0.25.0 源码为准"，或按代码行为描述而不是行号。

---

## 一、完整链路：从 `--gpu-memory-utilization` 到 token 数

分成五步，每一步都能在源码里指到位置。

### 1. 你请求多少显存

```
requested_memory = 显存总量 × gpu_memory_utilization
```

`gpu_worker.py:533-539` 的 debug 日志会打印 `Requested memory: %f (util), %s GiB`。

### 2. 实测"非 KV"部分占多少

`gpu_worker.py:470-502` 用一个 dummy batch 真跑一次 forward（`profile_run()`），
量出三块开销并相加：

```
non_kv_cache_memory = non_torch_increase        # CUDA context、通信缓冲等
                    + torch_peak_increase       # 激活峰值（profile_run 前后 torch 峰值之差）
                    + weights_memory            # 权重本身
```

关键点：**这部分不是查表来的，是实测的**。所以换模型、换 max_num_batched_tokens、
换量化方式，这个数都会变。这也解释了为什么它必须"起服务才能算出来"。

### 3. 两者相减，得到 KV 可用量

```python
available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)                                        # gpu_worker.py:527-531
```

顺带解释了你日志里那行 `Available KV cache memory: X GiB`（`gpu_worker.py:546-549`）。

### 4. 换算成 block 数

```python
num_blocks = int(available_memory // page_size // num_layers)   # kv_cache_utils.py:1001
```

其中 `page_size` 是**单层、单个 block** 的字节数，`num_layers` 是层数。
除完之后还要按 `num_gpu_blocks_override` 覆盖（`kv_cache_utils.py:1003`）。

### 5. token 数 = block 数 × block_size

```
KV 池容量(token) = num_blocks × block_size
```

默认 `block_size = 16`（PagedAttention 的页大小）。

---

## 二、用 Qwen3-8B-AWQ 的真实数字验算

参数（来自 `logs/vllm.log` 与配置）：

| 项 | 值 |
|---|---|
| 显存总量 | 8 GiB = 8,192 MiB |
| `gpu_memory_utilization` | 0.85 |
| 层数 | 36 |
| KV 头数 × 头维度 | 8 × 128 |
| dtype | fp16（2 B）|
| `block_size` | 16 |

**单 token、单层的 KV 字节数**：

```
2（K 和 V）× 8（头）× 128（维）× 2 B = 4,096 B = 4 KiB
```

**单 token、全部 36 层**：

```
4 KiB × 36 = 144 KiB = 0.140625 MiB
```

**单个 block（16 token、36 层）的 page 字节数**：

```
16 × 4,096 × 36 = 2,359,296 B = 2.25 MiB
```

**反推**：观测到 5,696 token

```
可用 KV 显存 = 5,696 × 144 KiB = 801.0 MiB
block 数     = 801.0 MiB ÷ 2.25 MiB = 356 个
验算：356 × 16 = 5,696 token ✓
```

**再反推 non-KV 占用**（注意：这一项已包含 CUDA graph 的估算，见第五节）：

```
non_kv = 0.85 × 8,192 - 801.0 = 6,162 MiB = 6.02 GiB
其中权重 5.68 GiB，所以激活 + CUDA context + graph 估算 ≈ 0.34 GiB
```

这个 0.34 GiB 是合理的（36 层、8B 模型、profile_run 的峰值激活 + CUDA context）。
**三个数（5,696 / 801 MiB / 356 block）互相闭合**，说明链路理解是对的。

---

## 三、可验证预测：换 `gpu_memory_utilization` 会得到多少 KV

用上面拟合出的 `non_kv = 6,162.2 MiB`（在 0.85 处拟合），**预测**其它档位：

> 这张表现在有可执行版本：`bench/kv_budget.py`（20 项单测钉住）。
> 跑 `python bench/kv_budget.py --gpu-mem-util 0.88` 即可复现，
> 且能算出手动推导容易漏掉的**并发/波次**——见最后一列。

| `gpu_memory_utilization` | 请求显存 (MiB) | 可用 KV (MiB) | 预测 block 数 | 预测 KV token | 并发 / 波次 |
|---|---|---|---|---|---|
| 0.78 | 6,389.8 | 227.6 | 101 | 1,616 | **0 / —**（连一个都装不下）|
| 0.80 | 6,553.6 | 391.4 | 173 | 2,768 | 1 / 8 |
| 0.82 | 6,717.4 | 555.2 | 246 | 3,936 | 1 / 8 |
| **0.85** | **6,963.2** | **801.0** | **356** | **5,696**（实测 ✓）| **2 / 4**（实测 ✓）|
| 0.88 | 7,209.0 | 1,046.8 | 465 | 7,440 | **3 / 3** |
| 0.90 | 7,372.8 | 1,210.6 | 538 | 8,608 | 3 / 3 |

**最后一列是模型算出来的，原来的手推表没有。** 它揭示了一件更值得讲的事：
并发不是随显存线性增长的，而是**阶梯式**的——因为块按整份请求分配，
`355 // 136 = 2`、`464 // 136 = 3`，中间那一大片显存对并发毫无帮助。
0.78 那档尤其说明问题：**池子非空，但一个请求都放不下。**

### 换成不带 GQA 的模型会怎样（反事实）

Qwen3-8B 有 8 个 KV 头（不是 32 个注意力头）。如果没有这一步压缩：

| KV 头数 | 每 token KV | 801 MiB 能装 |
|---|---|---|
| 8（实际） | 144 KiB | **5,696 token** |
| 32（假设） | 576 KiB | 1,424 token |

**1,424 token 意味着连一个 2,176 token 的请求都装不下**，这张 8 GB 卡
根本跑不了 8B。——这比"显存不够"有力得多，是"为什么能跑"的正面解释。

### 这是本轮唯一要做的一个实验（约 15 分钟）

**先写预测，再跑，再对答案。** 顺序不能反——先看结果再解释，等于没有预测。

```bash
# 1) 只起服务看它报多少 KV，不用压测
$VENV_ROOT/vllm/bin/vllm serve $MODEL --served-model-name qwen3-8b-awq \
    --quantization awq_marlin --max-model-len 4096 --max-num-seqs 8 \
    --gpu-memory-utilization 0.88 2>&1 | grep -E "Available KV cache memory|GPU KV cache size"

# 2) 把 "GPU KV cache size: N tokens" 和上表的 7,440 对比
```

**怎么读结果**（这比数字本身重要）：

- 预测落在 ±5% 内 → 说明你确实理解了这条链路，可以在面试里主动给出这个推导
- 偏差大 → 说明 `non_kv` 不是常数（很可能因为 `max_num_seqs` 或 CUDA graph 数目
  随 util 变化），这本身就是个好结论：**显存账本不能线性外推**
- 报 OOM 或 VM 崩 → 别忘了 §9.1 的教训，0.90 以上不要碰，这台机器会整机重启

**已知的模型误差**：`non_kv` 里含 CUDA graph 捕获的开销，而 graph 数量与
`max_num_seqs`、`max_num_batched_tokens` 相关。改 util 时它基本不变，但如果同时改
`--max-num-seqs`，预测就不可靠了。所以实验里**只改 util 一个变量**。

---

## 四、面试会被追问的五个问题（以及答案在哪）

| 问题 | 答案要点 | 出处 |
|---|---|---|
| 你的 KV 池 5,696 怎么来的？ | util × 总显存 − 实测 non-KV，除以 block page 字节与层数，再乘 block_size | `gpu_worker.py:527-531` + `kv_cache_utils.py:1001` |
| 为什么不按"剩余显存"线性算？ | 因为 non-KV 含实测激活峰值和 CUDA graph，不是常数；且 util 卡的是上限不是实测值 | `gpu_worker.py:470-502` |
| block_size 改大会怎样？ | 页变大 → block 数变少 → 内部碎片增加但管理开销降低；KV token 总量基本不变（page_bytes 同比放大） | 推导即可 |
| 为什么长 prompt 并发上不去？ | KV 池 ÷ 单请求 token 数 = 可同时容纳的请求数；2048+128 → 只剩 2 路 | 报告 §6.3 |
| vLLM 和 SGLang 的 KV 差异从哪来？ | 同为 0.85/0.88 的静态占比，但 non-KV 占用与 page 粒度不同 → 池子不同（5,696 vs 2,554） | 需分别按各自公式算 |

**注意**：最后一个问题不能直接套用上面的公式，SGLang 的记账方式不同
（`--mem-fraction-static` 卡的是"静态占比"而非"KV 可用量"）。
要做这个对比，得分别读两边的分配代码——这是下一步。

---

## 五、还没解决的部分（别在面试里装作懂）

1. **`page_size` 在不同 attention backend 下会不会变**：本地源码里
   `get_uniform_page_size` 要求各层一致（`kv_cache_utils.py:1006-1012`），
   但 MLA / 混合注意力的模型不满足这个前提，走的是另一条分支。
2. **CUDA graph 的显存怎么算**：源码里它有独立估算
   `profile_cudagraph_memory()`，并且**默认就会从 available 里扣**——
   `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认是 `True`
   （`envs.py:286`，环境变量缺省值 `"1"`，见 `envs.py:1959-1960`）。
   本文第二节反推出的 6,162 MiB 里因此**已经包含** graph 估算，
   不是纯权重 + 激活。真正还没搞清的是：这个估算值和实测 graph 占用差多少。
   要验它得在 `profile_cudagraph_memory()` 前后各打一次显存，属于下一步。
3. **只核对了 vLLM**；SGLang 的 KV 分配链路还没读。
