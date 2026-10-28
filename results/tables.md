# Results

> 数值格式为 `中位数 [最小 - 最大]`，区间来自每次重复的独立测量。
> 逐轮、逐请求的原始记录见对应 `*.json` 的 `samples` 字段。

## Startup

| backend | version | startup (s) | VRAM after load (GiB) |
|---|---|---|---|
| transformers (HF generate) | — | 9.24 | 5.68 |
| vLLM | — | 74.17 | 7.07 |
| SGLang | — | 73.1 | 7.48 |

## Benchmark


### workload `short`

| backend | conc | TTFT (ms) | ITL (ms) | E2E (s) | throughput (tok/s) | ok% | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|
| transformers (HF generate) | 1 | 197.6 | 101.4 | 13.2 | 9.8 | — | 5.7 |
| transformers (HF generate) | 4 | — | — | 14.3 | 35.9 | — | 5.9 |
| vLLM | 1 | 100.0 | 13.5 | 1.8 | 71.1 | — | — |
| vLLM | 2 | 126.9 | 13.8 | 1.9 | 135.5 | — | — |
| vLLM | 4 | 283.8 | 14.0 | 2.1 | 249.0 | — | — |
| vLLM | 8 | 555.1 | 14.2 | 2.4 | 434.4 | — | — |
| SGLang | 1 | 94.2 | 13.0 | 1.7 | 73.6 | — | — |
| SGLang | 2 | 156.8 | 13.2 | 1.8 | 139.7 | — | — |
| SGLang | 4 | 296.1 | 13.5 | 2.0 | 255.1 | — | — |
| SGLang | 8 | 375.5 | 15.5 | 2.3 | 436.5 | — | — |

### workload `medium`

| backend | conc | TTFT (ms) | ITL (ms) | E2E (s) | throughput (tok/s) | ok% | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|
| transformers (HF generate) | 1 | 370.0 | 94.1 | 24.5 | 10.5 | — | 5.9 |
| transformers (HF generate) | 4 | — | — | 27.3 | 37.5 | — | 6.2 |
| vLLM | 1 | 261.3 | 13.6 | 3.7 | 68.5 | — | — |
| vLLM | 2 | 386.9 | 14.6 | 4.1 | 124.0 | — | — |
| vLLM | 4 | 987.2 | 15.0 | 4.8 | 213.3 | — | — |
| vLLM | 8 | 1585.0 | 18.9 | 6.2 | 276.9 | — | — |
| SGLang | 1 | 301.7 | 13.1 | 3.6 | 70.1 | — | — |
| SGLang | 2 | 615.2 | 13.6 | 4.1 | 125.1 | — | — |
| SGLang | 4 | 894.4 | 14.0 | 4.5 | 125.9 | — | — |
| SGLang | 8 | 5406.1 | 14.1 | 9.0 | 156.2 | — | — |

### workload `long`

| backend | conc | TTFT (ms) | ITL (ms) | E2E (s) | throughput (tok/s) | ok% | peak VRAM (GiB) |
|---|---|---|---|---|---|---|---|
| transformers (HF generate) | 1 | 1252.0 | 115.2 | 15.9 | 8.1 | — | 6.2 |
| vLLM | 1 | 1029.5 | 14.8 | 2.9 | 43.9 | — | — |
| vLLM | 2 | 1976.6 | 15.7 | 4.0 | 64.4 | — | — |
| vLLM | 4 | 3957.3 | 15.6 | 5.9 | 64.6 | — | — |
| vLLM | 8 | 8088.9 | 15.8 | 10.1 | 63.4 | — | — |
| SGLang | 1 | 1064.0 | 13.7 | 2.8 | 45.6 | — | — |
| SGLang | 2 | 2479.1 | 13.8 | 4.2 | 45.4 | — | — |
| SGLang | 4 | 5494.6 | 13.8 | 7.2 | 44.2 | — | — |
| SGLang | 8 | 11724.2 | 14.0 | 13.5 | 42.6 | — | — |

## 数据质量

| backend | 请求总数 | 失败 | 成功率 | 失败类型 | 非 ok 格子 |
|---|---|---|---|---|---|
| transformers (HF generate) | — | 0 | — | — | — |
| vLLM | — | 0 | — | — | — |
| SGLang | — | 0 | — | — | — |

> 注：本结果由旧版客户端采集，未记录逐请求明细，成功率一栏无法计算。