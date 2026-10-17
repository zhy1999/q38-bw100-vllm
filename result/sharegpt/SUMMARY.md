# vLLM 赛道 ShareGPT 压测结果（对齐 vllm 官方剪裁口径）

- 时间：2026-10-16 14:05 ~ 2026-10-17 01:05 UTC（11h，含 C16-C64 前半）
- 口径：vllm 0.31 is_valid_sequence 官方默认（prompt 4~1024, out≥4, total≤2048，超限丢弃）
- 数据集：ShareGPT_V3_unfiltered_cleaned_split.vllm2048.json（94145→82577 条）
- 每档 3 轮 × 1000 请求，seed 20260912，temp=0，流式
- 成功率：全部 3000/3000（无 error）
- 服务端：vllm 0.21 m1b (MTP spec=4, TP2, GPU 0,1, port 18680)

| 档位 | 每轮耗时 | 输出吞吐 tps | TTFT mean | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | 时延 mean |
|---|---|---|---|---|---|---|---|---|
| C1 | 126 min | 28.0 | 163ms | 166ms | 235ms | 46.2ms | 1125.8ms | 7.6s |
| C4 | 33 min | 106.8 | 317ms | 303ms | 369ms | 52.4ms | 1079.3ms | 7.9s |
| C16 | 11.3 min | 313.5 | 390ms | 357ms | 478ms | 65.6ms | 1539.3ms | 10.8s |
| C32 | 8.0 min | 440.8 | 479ms | 434ms | 589ms | 96.8ms | 2233.1ms | 15.4s |
| C64 | 4.9 min | 723.8 | 583ms | 541ms | 964ms | 112.0ms | 2767.8ms | 18.7s |

注：
- 输出长度均值 212 token（重建自数据集真实回复长度；客户端流式未取到 usage，tpot=(lat-ttft)/out_len 口径与官方一致）
- TPOT p95 高为短输出样本 prefill 均摊所致；decode 单流 ~30 tok/s（MTP 投机低并发负收益）
- 官方基线包（result/base/20261015-092257）仅含 fixed SLA，无 ShareGPT 数值

## 官方 fixed 4K/1K 基线（供参考）
| C1 | C8 | C16 | C32 | C64 | C128 |
|---|---|---|---|---|---|
| tps=37.5 | 247.6 | 398.8 | 490.7 | 666.6 | 790.6 |
| ttft95=0.60s | 4.32s | 8.44s | 16.75s | 32.82s | 61.95s |
| tpot95=26.1ms | 31.4ms | 38.8ms | 63.4ms | 93.3ms | 160.3ms |
