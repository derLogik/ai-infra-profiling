# KV cache 的 block 池：数据结构与淘汰策略（源码核对）

> 承接 `kv-cache-allocation.md`（那篇讲"一共能装多少 token"），
> 这篇讲"这些 block 运行时怎么被分配、复用、淘汰"。
>
> **版本声明**：代码来自本地 `work/vllm_src`，版本 `v0.25.0-21-gfabec87f`。
> 压测跑的是 0.29.0，公式稳定但行号可能偏移。

---

## 一、两个核心数据结构

### 1. `KVCacheBlock`（`kv_cache_utils.py:118`）

`@dataclass(slots=True)`，只有六个字段，没有张量——**它只是元数据**，
真正的 KV 数据在别处的显存张量里，靠 `block_id` 索引。

| 字段 | 含义 | 注意点 |
|---|---|---|
| `block_id` | 0 ~ num_gpu_blocks−1 | 张量里的页号 |
| `ref_cnt` | 引用计数 | **0 = 在空闲队列里**，是"可被淘汰"的唯一标志 |
| `_block_hash` | `BlockHashWithGroupId \| None` | **只有块填满且已缓存时才有值** |
| `_block_hash_num_tokens` | 该 hash 覆盖多少前缀 token | 整块是边界，partial 别名可能落在块中间 |
| `prev_free_block` / `next_free_block` | 双向链表指针 | 注释明确写：**只允许队列操纵** |
| `is_null` | 是否是 null 块 | 永不被缓存 |

`set_block_hash()` 里有一句断言很关键（`kv_cache_utils.py:153-155`）：

```python
assert self.block_hash is None and self._block_hash_num_tokens is None, \
    "The block already has a hash. This should not happen."
```

即：**一个块在同一个时刻只能有一个 hash，设过就必须先 reset 才能再设**。
这保证了"hash → 块"映射不会出现一个块属于两个前缀的情况。

### 2. `FreeKVCacheBlockQueue`（`kv_cache_utils.py:179`）

不是 `deque`，是**自己实现的侵入式双向链表**。原因写在 docstring 里
（`kv_cache_utils.py:180-186`）：

> 需要支持 **O(1) 从队列中间摘除**一个块，`deque` 做不到；
> 为了追平 C++ `deque` 的性能，操作时不分配任何 Python 对象，
> 直接改块自己的 `prev/next` 指针。

中间摘除什么时候发生？**前缀缓存命中时**——一个块本来在空闲队列里（ref_cnt=0），
另一个请求命中了同一前缀，它必须立刻从队列里摘掉，否则会被当垃圾回收。
这个操作在 `touch()` 里（`block_pool.py:680-695`）。

还用了 `block_id=-1` 的 fake head/tail 哨兵节点（`kv_cache_utils.py:217-225`），
目的是省掉链表两端的边界判断——典型的"用哨兵换掉分支"写法。

---

## 二、五条关键不变量

这五条是读代码时抓的，也是面试最容易问穿的地方。

| # | 不变量 | 证据 |
|---|---|---|
| 1 | `ref_cnt == 0` ⇔ 块在空闲队列中（或可被淘汰）；`> 0` 表示正被某个请求持有 | `block_pool.py:691` 注释、`free_blocks` 的 `ref_cnt -= 1` |
| 2 | 分配时 `assert block.ref_cnt == 0`，再 `+= 1` | `block_pool.py:645-652` |
| 3 | 缓存块的 hash 在淘汰时必须 `reset_hash()`，且从 hash 表里移除 | `block_pool.py:657-678` |
| 4 | **null block（block_id=0）永远被占用，不参与分配** | `block_pool.py:190-191`、`749` |
| 5 | 块的**主 hash** `_block_hash` 只能设一次（设过必须 `reset_hash()` 才能再设）；但反向索引里一个块可以挂多个 hash | `kv_cache_utils.py:153-155`、`block_pool.py:603-623` |

**第 4 条有个实际后果**：可用块数 = `num_gpu_blocks − 1`。
`reset_prefix_cache()` 里直接断言 `num_used_blocks != 1` 就报错，
注释写 "The null block is always marked as used"（`block_pool.py:748-755`）。
所以算容量时**理论上该减 1**，但相对 356 个块，影响是 0.3%，可忽略——
不过这个细节能被问出来，答得上就说明真的读过。

---

## 三、前缀命中时发生什么（调用序列）

```
请求到达 → 算 block_hashes → 查 cached_block_hash_to_block
  命中 → BlockPool.touch(blocks)          block_pool.py:680
         · ref_cnt == 0 的块从空闲队列摘除（remove，O(1)）
         · ref_cnt += 1
  未命中 → 从空闲队列头部 popleft 拿块      block_pool.py:625-655（get_new_blocks）
         · popleft_n(num_blocks)             block_pool.py:639
         · 若该块还带着旧 hash，先 _maybe_evict_cached_block 清掉
         · ref_cnt += 1
```

注意"命中"不是复制数据，而是**给同一个物理块加引用**——
这就是 PagedAttention 前缀复用的本质：多个请求的逻辑页指向同一个物理页。

还有一层容易被忽略的设计：`BlockHashToBlockMap` 的 value **可以是单个块，
也可以是 `{block_id: block}` 字典**（`block_pool.py:57-104`）。
也就是说同一个 hash 允许对应多个物理块——因为不同请求可能各自写入了内容相同的
KV（它们不是一起算的，没法共享）。查找时随便返回一个（`get_one_block`）。
这个细节决定了：**前缀命中的去重不是靠"唯一块"，而是靠 hash 表允许多值**。

反过来看也成立：`_insert_block_hash`（`block_pool.py:603-623`）显示，
一个块已经有主 hash 时，新 hash 会被塞进 `cached_block_hashes_by_block[block_id]`
这个集合里——所以 **hash 与块是多对多**，不是一对多。
主 hash 只有一个（`set_block_hash` 会断言），反向索引可以挂多个。
淘汰时 `_remove_cached_block_hashes` 要把这些全部清干净，否则会留下悬空映射。

---

## 四、淘汰顺序：为什么哈希块留得更久

`free_blocks()`（`block_pool.py:697-718`）在 `ref_cnt` 减到 0 时把块放回队列，
但它**把两类块分开放**：

```python
self.free_block_queue.prepend_n(blocks_without_hash)   # 放到队头
self.free_block_queue.append_n(blocks_with_hash)       # 放到队尾
```

- 队列**头部先被分配**（`popleft`），所以头部的块内容最先被覆盖
- 没有 hash 的块（半满块、或缓存没开）放进头部 → **最先被淘汰**，天经地义
- 有 hash 的块放进尾部 → **活得更久**，因为它们还能被别的前缀命中

队列的排序规则在 docstring 里写得很清楚（`kv_cache_utils.py:188-195`）：

1. 最久未使用（LRU）在最前
2. 若两个块最后访问时间相同（同一批被释放），**hash token 更多的在前**
   ——也就是一条块链的尾部先被淘汰（保住链头，复用价值更高）
3. 这个顺序是靠**释放时反转块的顺序**实现的（类外完成，`free_blocks` 收到的是
   已排好序的 `ordered_blocks`）

这三条合起来说明：vLLM 的淘汰策略不是朴素 LRU，而是 **LRU + 前缀复用价值加权**。
这是"读过代码"和"听说过 PagedAttention"的分水岭。

---

## 五、把上一张图纸闭合：`1.39x` 从哪来

`kv_cache_utils.py:2160-2173`：

```python
num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
logger.info_once("Maximum concurrency for %s tokens per request: %.2fx",
                 f"{max_model_len:,}", max_concurrency)
```

注释直接写明：`GPU KV cache size in tokens = max_concurrency * max_model_len`。

所以 `max_concurrency = num_tokens / max_model_len`：

```
5,696 ÷ 4,096 = 1.3906  → 日志打出 1.39x ✓
```

和上一张笔记的推导接上了：`gpu_memory_utilization` → 实测 non-KV → 可用显存
→ block 数 356 → token 数 5,696 → 4K 上下文并发 1.39。

**这条链现在是完整闭合的四个数**：
util 0.85 → 356 blocks → 5,696 tokens → 1.39x @ 4K。

---

## 六、自测五问（答不上来就回去看对应段落）

1. 为什么空闲队列不用 `deque`，要自己写双向链表？
   （提示：`touch` 需要中间摘除；`deque` 是 O(n)）
2. `ref_cnt` 从 0 变 1 和从 1 变 0，分别在什么代码路径上发生？
3. 同一个前缀 hash 为什么会对应多个物理块？查找时怎么选？
4. 释放一块时，有 hash 和没 hash 的块分别插到哪一端？为什么？
5. `num_gpu_blocks = 356` 时，真正可分配的块是几个？

---

## 七、还没读懂的部分（别装作懂）

1. **`_block_hash_num_tokens` 的 "partial 别名" 是什么场景**：
   注释说 partial alias 可以停在 cache block 内部（`kv_cache_utils.py:128-130`），
   这应该与 block_size ≠ hash_block_size 的混合分组有关，我只读到接口，没读透。
2. **多 group（混合注意力 / MLA）下这个队列怎么共享**：
   `BlockPool` 是全局一个，但分组由 `KVCacheCoordinator` 管（`kv_cache_coordinator.py`），
   我只看了单 group 的路径。
3. **`prev/next_free_block` 被错误操作会怎样**：没有防护，
   只有注释约束"只允许队列操作"（`kv_cache_utils.py:132-135`）。
   这是个**没有断言保护的隐式不变量**——如果要找可以改进的点，这里算一个候选
   （但改它需要能跑完整测试套件，暂不适合作为第一个 PR）。
