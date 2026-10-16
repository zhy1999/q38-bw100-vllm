# 官方基线合规说明（对应赛题文档 §4.1.3）

本机为 **海光 DCU BW100**，属于赛题官方发布基线的卡型（子赛道 A · vLLM，
官方镜像 `vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`）。
按赛题规则，**直接采用官方基线作为对比分母**，不引入自测基线。

## 一、官方基线出处

《社区qwen3.8挑战赛.md》§4.1.3「随机数据 Baseline」—— BW100 行：

| 卡型 | 测试场景 | SLA 条件 | 官方最大并发 Baseline |
|---|---|---|---|
| BW100 | 4K 输入 / 1K 输出 | TTFT < 5s 且 TPOT < 50ms | 1 |
| BW100 | 64K 输入 / 1K 输出 | TTFT < 20s 且 TPOT < 60ms | 0（未通过）|

官方文档仅公布 SLA 最大并发数，未公布 TPS/TTFT/TPOT 明细；
故本提交的对比口径统一为 **SLA 最大并发**（详见 `REPORT.md` §2.1）。

## 二、合规声明

1. **同镜像底座**：本方案镜像即官方镜像 `vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`，
   未替换官方 .so / vllm_hcu / aiter / torch / DTK 组件；`patch/` 只覆盖
   `dist-packages/vllm/` 的 Python 层代码（可 `verify.sh` 逐锚点校验）。
2. **同评测口径**：固定长度压测使用与官方一致的随机 token 序列方案
   （`run_bench_fixed.sh`：每档 2 轮预热 + 3 轮正式，指标取 3 轮均值；
   原始数据 `result/fixed/` 逐轮归档 summary.json + requests.jsonl）。
3. **不调低基线**：未改动官方基准数据；官方 64K 档 Baseline=0（未通过）如实保留，
   不换算为数值分母。
4. **证据完整**：性能原始数据 `result/fixed/`（130 文件），启动命令
   `scripts/start_server.sh`（含全部 flag 与 env），补丁与校验
   `patch/`（apply.sh / verify.sh / SHA256SUMS）。

## 三、与"自测基线"的区别说明

开放组/其他卡型（如 BW150）因无官方基线需自测基线（§3.1）；
BW100 属官方覆盖范围，按 §4.1.3 直接采用官方基线即可，两套口径不混用。
