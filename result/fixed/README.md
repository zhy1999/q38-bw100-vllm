# fixed — 固定长度压测原始数据

`run_bench_fixed.sh` 输出：`<input>in_<output>out_c<concurrency>/`

- `config.json` / `progress.json`: 场景配置与进度
- `warmup-01/02`, `round-01..03`: 每轮 `summary.json` + `requests.jsonl`（逐请求明细）
- `summary.json`: 该场景 3 轮统计（TPS / TTFT / TPOT 的 mean/min/max/p50/p95/p99）

汇总分析见 `../benchmark-fixed-report.md`。
