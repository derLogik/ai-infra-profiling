# 调度器怎么决定这一步放几个请求进来（源码核对）

> 承接 `kv-cache-allocation.md`（能装多少 token）和 `kv-cache-block-pool.md`
> （block 怎么分配淘汰）。这篇是最后一环：**调度器用这些 block 做什么决策**，
> 也就是报告 §6.3 里"8 个请求为什么只跑 2 个"的机制层答案。
>
> **版本声明**：代码来自本地 `work/vllm_src`，版本 `v0.25.0-21-gfabec87f`。
> 压测跑的是 0.29.0，逻辑稳定但行号可能偏移。

---

## 一、请求要进 batch，得过三道闸

入口是 `KVCacheManager.allocate_slots()`（`kv_cache_manager.py:269`）。
返回 `None` 就代表"这一步放不下这个请求"。三道闸依次是：

### 闸 1：整个序列能不能装下

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        ..., num_tokens=full_num_tokens, ..., apply_admission_cap=True)
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```
`kv_cache_manager.py:397-412`

这是**严格模式**：不光是"这一步需要的 token 装得下"，而是"这个请求从头到尾
（prompt + 全部输出）都得装得下"才准进。装不下就直接拒，让它继续在 waiting 队列里等。

⚠️ **这个开关默认是开的**（2026-10 核对，见下方"修正"一节）：
`SchedulerConfig.scheduler_reserve_full_isl: bool = True`（`config/scheduler.py:140`），
调度器每次调用都把这个值传进来（`scheduler.py:914`）。
`kv_cache_manager.py:279` 那个 `full_sequence_must_fit: bool = False` 只是**函数参数默认值**，
运行时会被调度器的值覆盖——**看函数签名会得出错误结论**，这是我第一版笔记踩的坑。

### 闸 2：给别的请求留 watermark 余量

```python
watermark_blocks = 0
# The watermark is applied to waiting/preempted requests only, and only
# when there's at least one request already scheduled.
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING, RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks
```
`kv_cache_manager.py:388-395`

注意两个限定条件，都不是随手写的：

- **只对 waiting / preempted 的请求加**——已经在跑的请求不额外收税
- **只在已经有请求被调度时才加**——空跑时不预留

目的写在注释里：`avoid frequent preemptions`（`kv_cache_manager.py:166`）。
也就是宁可少放一个进来，也不要放进来了又被踢出去、反复抖动。
**这是一个用吞吐换稳定性的显式取舍**，面试里可以直接拿来讨论。

⚠️ **但它默认是关的**：`watermark: float = Field(default=0.0, ge=0.0, lt=1.0)`
（`config/scheduler.py:146`），docstring 明写 "0.0 (the default) disables the watermark"。
于是 `watermark_blocks = int(0.0 × num_blocks) = 0`（`kv_cache_manager.py:168`）。

**所以你的压测里这个机制从未生效。** 它是"可以提供、但默认不启用"的旋钮。
要让它起作用得显式设 `--watermark 0.1` 之类。
——这个细节很能说明"读过代码"：光看 `allocate_slots` 里的 `watermark_blocks`
会以为默认有节流，实际上默认一个块都不留。

### 闸 3：扣掉 reserves 之后还够不够

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None
```
`kv_cache_manager.py:446-450`

`reserved_blocks` 是留给**已在飞行中的序列**的（in-flight），
防止一个长请求把池子吃光导致别人没法继续 decode。

⚠️ **但它在你这个场景里恒为 0**：调度器里只有一处让它非 0
（`scheduler.py:897-903`）：

```python
reserved_blocks = 0
if load_kv_async:
    # An async load holds its blocks for the whole transfer with
    # no forward progress and isn't preemptible here.
    reserved_blocks = self._inflight_prefill_reserved_blocks()
```

`load_kv_async` 只在**异步 KV 加载**（KV connector / offloading）时才是真。
单卡、无 connector 的压测里它永远是假 → **`reserved_blocks = 0`**。

于是闸 3 退化成一句话：`free_blocks >= num_blocks_to_allocate`。

---

## 二、放不进去之后：抢占怎么发生

调度器拿到 `allocate_slots()` 的返回值后（`scheduler.py:533-582`）是一个
**while True 循环**：放不下就踢人，踢完再试，直到放下或没人可踢。

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break                                  # 放下去了
    # 放不下 → 抢占优先级最低的那个
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        ...
    else:
        preempted_req = self.running.pop()     # 默认策略：踢最后进来的（LIFO）
    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break                                  # 把自己都踢了还放不下 → 放弃
```

三个细节值得记：

1. **默认策略是 LIFO**（`self.running.pop()`——踢队尾，也就是最新进来的）。
   优先级策略才按 `(priority, arrival_time)` 挑。
2. **可能把自己踢掉**：`if preempted_req == request: break`，说明抢占没有"保护自己"的特权。
3. `_preempt_request`（`scheduler.py:1143`）里 `request.num_preemptions += 1`，
   并且把这个 id 放进 `reset_preempted_req_ids`——被抢占过的请求重算时
   要重新做 prefill，这正是"抢占代价高"的来源。

---

## 三、用它算你自己的数据：为什么只有 2 路并发

这是这套代码和你实测数字**对上的地方**。

已知（来自 `notes/kv-cache-allocation.md`）：

| 项 | 值 |
|---|---|
| `block_size` | 16 token |
| KV 池总 block 数 | 356，其中 1 个是 null block → **可用 355** |
| 单请求总长度 | 2,048 prompt + 128 输出 = 2,176 token |

**单请求需要多少 block**：

```
ceil(2,176 / 16) = 136 个 block
```

**池子能装几个**：

```
355 ÷ 136 = 2.61  →  向下取整 = 2 个
```

第三个请求需要第 273~408 个 block，而池子只有 355 个 → 闸 1 直接拒绝 → 它留在 waiting 队列。

### 真正扎心的部分：池子没满，但装不下下一个

| 项 | 数值 |
|---|---|
| 已用 | 2 × 136 = **272 block** |
| 剩余 | **83 block** = 1,328 token |
| 利用率 | 272 / 355 = **76.6%** |
| 下一个请求需要 | 136 block |
| 结论 | 差 **53 个 block**（848 token）就装得下第三个 |

**这 83 个 block 就白白闲置了。** 它们不是被谁占着，是**碎片化浪费**——
单个请求的粒度是 136 块，池子只能按整块卖，剩下的零头谁也用不上。

这解释了三个现象，而且是同一件事：

1. **为什么并发卡在 2**：不是池子满了，是"剩余空间大到装不下下一个整份请求"
2. **为什么吞吐只涨 1.44 倍**（43.9 → 63.4）而不是接近 3 倍：2 路并行对比 1 路，
   理论上限就是 2 倍，实际还掉了些（prefill 争用、decode 串行插空）
3. **为什么 8 个请求变成 4 波**：⌈8 / 2⌉ = 4

> **这就是报告里"2 路 × 4 波"的机制层解释。** 之前那个模型是从三个观测量反推的
> （推断），现在有了准入规则的源码依据（机制）。两者互相印证，但仍各自成立：
> 观测部分是吞吐/TTFT/E2E；机制部分是 `allocate_slots` 的闸门逻辑。

---

## 四、三个能直接拿去面试的结论

1. **"KV 池不够"是个不精确的说法。** 准确说法是"**池容 ÷ 单请求粒度向下取整**"，
   所以总有 0 ~ (1 个请求 − 1 块) 的闲置。你的例子里闲置了 23.6%。
   ——这个数字能直接用来说明"为什么显存占用看着不满但并发上不去"。

2. **vLLM 用 watermark 主动换稳定性。** 对 waiting/preempted 请求额外预留若干块，
   宁可少放一个也不愿反复抢占。抢占的代价是重算 prefill（`num_preemptions` 累加后
   要 re-prefill），在高并发下这个代价远大于少放一个。

3. **默认抢占策略是 LIFO 而不是 LRU。** 踢最后进来的，不是最久没动的。
   这在"请求长度差异大"的场景下会让长请求更容易活下来、短请求反复被踢——
   是个可以拿来讨论的设计取舍点（不是 bug）。

---

## 五、修正记录：我第一版写错了三处（2026-10-29 核对）

这篇笔记第一版有 3 个判断是错的，都是"只看函数签名、没查配置默认值"导致的。
记在这里，比悄悄改掉更有用——它说明**读源码最容易错的地方是默认值**。

| # | 第一版的说法 | 实际 | 错在哪 |
|---|---|---|---|
| 1 | `full_sequence_must_fit` 默认 False，闸 1 未必生效 | **默认 True**（`config/scheduler.py:140`，经 `scheduler.py:914` 传入） | 看了 `kv_cache_manager.py:279` 的参数默认值，没看调度器每次调用都覆盖它 |
| 2 | watermark 会给 waiting 请求预留余量 | **默认 `watermark=0.0` → 预留 0 块**（`config/scheduler.py:146`） | 看了机制存在，没看默认值把它关了 |
| 3 | `reserved_blocks` 会扣掉 in-flight 占用 | **单卡场景恒为 0**（`scheduler.py:897-903`，仅 `load_kv_async` 时非 0） | 同上 |

修正之后，**你的压测场景其实是最简形态**：

```
闸 1：整个 2,176 token 序列要装得下     （默认开）
闸 2：watermark = 0                      （默认关，无效）
闸 3：free_blocks >= 需要块数，reserved=0（恒 0）
```

三个闸退化成一个：**整序列装得下 + 空闲块够**。
§3 用 355 / 136 算出的"只能放 2 个"，因此不是估算，而是**这个最简判定下的确定结果**。

---

## 六、控制变量实验：关掉 full-ISL 预留会怎样（可执行）

这是"从读懂到改动"的第一个候选：**不改引擎代码，只切配置**，
就能拿到一组关于准入策略的对照数据。

### 假设

把 `scheduler_reserve_full_isl` 关掉后，判定从"整序列装得下"放宽到"当前 chunk 装得下"。
理论上可以让第三个请求挤进来先做 prefill——**只要 chunk 足够小**。

### 预测（先写下来，再去验，这是关键）

| 组 | 配置 | 预测 |
|---|---|---|
| A 基线 | 默认 | 2 路并发，长 prompt c8 吞吐 ≈ 63 tok/s，TTFT 中位 ≈ 8.1 s |
| B | 只加 `--no-scheduler-reserve-full-isl` | **几乎无变化**。因为默认 chunk = 整个 prompt（2,048 token = 128 块）> 剩余 83 块，照样装不下 |
| C | B + `--max-num-batched-tokens 512` | chunk 降到 512 token = 32 块，83 块可容纳 2 个额外请求 → **并发可能升到 3~4**，但 prefill 被切碎、TTFT 中位会上升 |

**B 组无变化本身就是结论**：它证明闸 1 不是这次的约束条件，真正卡住的是**块的总量**。
这比"我觉得是显存不够"强得多——它是一个被验证的排除实验。

### 三个可观测结果，与各自的解释

| 观测 | 说明什么 |
|---|---|
| 迭代日志里的并发数（用 `bench/analyze_iterations.py` 数） | 直接看到"这一步有几个请求在算"，比从吞吐反推可靠 |
| 吞吐与 TTFT 中位 / p95 | C 组若吞吐涨而 p95 恶化，就是拿尾延迟换吞吐 |
| 日志里的抢占次数（`num_preemptions` / "Preemption" 关键字） | **关键风险**：关掉预留后可能不是"多放进来"，而是"放进来又被踢出去重算"。若抢占频繁出现，这条路线就是负收益 |

### 怎么跑（每组一条命令，不用手拼）

`bench/run_matrix.py` 现在支持 `--server-arg` 透传和 `--tag` 标签，
所以三组实验各自一条命令，结果与日志自动分开存：

```bash
# A 组：基线
python bench/run_matrix.py --model-path <model> --backends vllm \
  --workloads long --concurrency 8 --tag A

# B 组：只关掉 full-ISL 预留
python bench/run_matrix.py --model-path <model> --backends vllm \
  --workloads long --concurrency 8 --tag B \
  --server-arg=--no-scheduler-reserve-full-isl

# C 组：再叠加 chunk 切小
python bench/run_matrix.py --model-path <model> --backends vllm \
  --workloads long --concurrency 8 --tag C \
  --server-arg=--no-scheduler-reserve-full-isl \
  --server-arg=--max-num-batched-tokens --server-arg=512
```

!! 三组都要拿到"这一步几个请求在算"的证据，必须在启动参数里加
`--enable-logging-iteration-details`（同样用 `--server-arg` 传）。
少了它，`compare_experiments.py` 的"实测并发"列会是 `—`。

**先把预测写下来**，存成 `preds.json`：

```json
{
  "A": {"concurrency": 2, "throughput": 63.4, "ttft_median": 8.09,
        "note": "基线：2 路并发，4 波"},
  "B": {"concurrency": 2, "throughput": 63.4, "ttft_median": 8.09,
        "note": "预期无变化：chunk 仍是整个 prompt，照样装不下"},
  "C": {"concurrency": 4, "throughput": 80.0, "ttft_median": 12.0,
        "note": "chunk 切到 512，预期并发升到 4"}
}
```

然后一条命令出对照表：

```bash
python bench/compare_experiments.py preds.json \
  results/vllm__A.json results/vllm__B.json results/vllm__C.json
```

输出（下列数字是演示用假数据，不代表实测）：

```
| 组 | 预测并发 | 实测并发 | 预测吞吐 | 实测吞吐 | TTFT 中位(预测/实测) | 抢占行数 | 判定 |
| A | 2 | 2 | 63.4 | 63.0 | 8.09 / 8.10 | 0 | 符合 |
| B | 2 | 2 | 63.4 | 64.1 | 8.09 / 8.10 | 2 | 符合 |
| C | 4 | 4 | 80.0 | 60.0 | 12.00 / 8.10 | 0 | 偏离 |
```

**这张表的重点不是数字，是"判定"列**：预测和实测放一起，对不上就明写"偏离"。
C 组那行就是例子——并发确实升到 4（预测对），但吞吐 60 远低于预测的 80
（预测错），工具不会替人圆场。

> 先确认参数名：`vllm serve --help | grep -i reserve`。
> 若这个版本没暴露 `--no-` 形式，退路是配 `scheduler_reserve_full_isl: false`
> 写进 YAML 用 `--config` 传入，或直接改 `config/scheduler.py` 的默认值
> （改默认值属于"改动"，要记进报告并说明是实验性修改）。

### 为什么"chunk 大小"确实由 `max-num-batched-tokens` 控制

这条已核对，不是猜的（否则 C 组实验没有意义）：

- `token_budget = self.max_num_scheduled_tokens`（`scheduler.py:416`）
- `max_num_scheduled_tokens` 未显式设置时**回落到 `max_num_batched_tokens`**（`scheduler.py:109-113`）
- 单请求每步的 token 数被 budget 截断：`num_new_tokens = min(num_new_tokens, token_budget)`（`scheduler.py:480`、`:844`）

所以设 `--max-num-batched-tokens 512`，2,048 token 的 prompt 就会被切成 ≤512 的 chunk
分步 prefill，每 chunk 需 `512/16 = 32` 个 block。**83 个空闲块因此能容纳 2 个额外请求的 chunk**——
这正是 C 组预测"并发升到 3~4"的依据。

> 另需确认参数名本身：`vllm serve --help | grep -i reserve`。
> 若这个版本没暴露 `--no-` 形式，退路是配 `scheduler_reserve_full_isl: false`
> 写进 YAML 用 `--config` 传入，或直接改 `config/scheduler.py` 的默认值
> （改默认值属于"改动"，要记进报告并说明是实验性修改）。

### 这一步完成后，线一的状态会变

现在三篇笔记回答的是"它怎么工作"。跑完这个实验，就有了**"我改了一个策略、
用一个对照实验说明它不影响（或影响）性能，并解释了原因"**——这才是目标里
「做出改动、用可复现数据证明、解释取舍」要求的形态。

哪怕结果是"没变化"，也是一个成立的结论，前提是预测先写下来了。

---

## 七、还没读懂的部分

1. ~~**swap 与 recompute 两种抢占模式的区别**：我只读了 `_preempt_request` 的入口，
   没读它内部按什么条件选 swap / recompute。这决定了"被抢占"的代价是换出到 CPU
   还是直接丢弃重算，量级差别很大。~~

   **✅ 已结（2026-10-28，读源码，无需 GPU）**：**V1 只有 recompute，没有 swap。**

   三处证据：

   - `v1/core/sched/scheduler.py:1152-1156`：`_preempt_request` 里先
     `self._free_request_blocks(request)`（直接把块还回池子），然后
     `request.num_computed_tokens = 0`
   - `v1/core/sched/scheduler.py` 全文搜不到任何 `swap` / `cpu_block` 相关代码
   - `v1/request.py:333-341` 的 `RequestStatus` 里只有 `PREEMPTED`，
     **没有 `SWAPPED`**（那是 V0 遗留的状态）

   **这意味着**：被抢占的请求重新被调度时，已算的 KV 已经丢了，
   必须从第 0 个 token 重新 prefill 整条 prompt（最长 2,048 token），
   而不是把 CPU 上的 KV 搬回来。

   **对你 C 组实验的直接含义**（重要）：

   C 组（关掉 full-ISL 预留 + 把 chunk 切到 512）会让请求更容易进 running 队列，
   但如果因此触发抢占，代价是**整条 prompt 重算**，不是廉价地搬回来。
   所以 C 组的可能结局不是"吞吐变好"，而是：

   | 若观察到 | 说明 | 结论 |
   |---|---|---|
   | 并发升、吞吐升、抢占 0 | chunk 让零头显存被用上，无代价 | 假设成立，这是净收益 |
   | 并发升、吞吐升、抢占 > 0，p99 TTFT 变差 | 用重算换了并发 | 要看总 token 吞吐是否覆盖重算成本 |
   | 并发仍 2 | 约束不在块数 | 原假设错，得回头查别的闸 |

   **所以 `num_preemptions` 和 p99 TTFT 必须一起看**——只看平均吞吐会得出错误结论。

2. **`prefill_schedule_interval`（`config/scheduler.py:153`）**：只对 DP 部署有意义，
   单卡应该是 1，但没验证过它会不会影响单卡行为。
3. **multi-group（混合注意力 / MLA）下的准入**：我只走了单 group 路径，
   没看 `KVCacheCoordinator` 在多 group 时怎么把"装得下"这个判断组合起来。
4. ~~**`max_num_seqs` 与 KV 余量的耦合**：`max_num_running_reqs = max_num_seqs`
   （`scheduler.py:108`）是另一道独立上限。你的 8 并发实验里它应该不成为约束
   （默认远大于 2），但没实测确认过——如果它也是 2，那"2 路并发"的解释就要改。~~

   **✅ 已结（2026-10-28，无需重跑）**：三处证据一起看，这个替代解释被排除。

   - 源码：`v1/core/sched/scheduler.py:108`
     `self.max_num_running_reqs = self.scheduler_config.max_num_seqs`
   - 该上限在准入时生效：同文件 `:644` `if num_running >= self.max_num_running_reqs:`
   - 实测配置：`results/vllm.json` 的 `server_args` 里明确写着 `--max-num-seqs 8`

   所以上限是 **8**，不是 2。8 个请求同时来，序列数这道闸根本不拦，
   能同时跑 2 个只能是因为 KV 块不够（355 块 ÷ 136 块/请求 = 2）。
   **"并发卡在 2 是 KV 导致"这个结论站得住。**

   补一句边界：这条证据是"配置 + 源码"层面的一致性，不是"日志里数出来 2 个"。
   真要拿到实测并发，仍需 `--enable-logging-iteration-details` 后跑
   `bench/analyze_iterations.py`。
