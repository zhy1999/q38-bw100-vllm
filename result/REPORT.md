# Qwen3.8-27B × 2× 海光 DCU BW100 · vLLM · 实测报告（五段式）

> 子赛道 A（vLLM）。卡型 BW100 属于官方发布基线的平台（赛题文档 §4.1.3），
> 本报告分母**直接采用官方基线**，不引入自测基线（合规说明见 `BASELINE_COMPLIANCE.md`）。

---

## 第 1 段 · 环境声明

完整机读版本见 `env_declare.yaml`，此处为与实测一致的关键项摘要。

| 模板项 | 填写 |
|---|---|
| 卡型 | 海光 DCU BW100 × 2（TP=2，`HIP_VISIBLE_DEVICES=0,1`） |
| 互联 | PCIe / Hygon Interconnect |
| OS / Python | Ubuntu 22.04 LTS / Python 3.10 |
| DTK | dtk2604 |
| 容器镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 推理框架 | vLLM 0.21.0（镜像内版本 `0.21.0+das.8c979d2.dtk2604`，海光 vllm_hcu 生态版本） |
| 权重 | `/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8`（仅只读挂载，未替换未裁剪） |
| 量化 | Channel-wise INT8 W8A8（`--quantization compressed-tensors` + `VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=True`） |
| 启动命令全文 | `scripts/start_server.sh`（与目标机实测一致：`python3 -u -m vllm.entrypoints.openai.api_server`） |
| 关键参数 | TP=2 · bf16 激活 · max-model-len 131072 · max-num-batched-tokens 16384 · max-num-seqs 128 · gpu-memory-utilization 0.85 · `--gdn-prefill-backend triton` · MTP 投机（`num_speculative_tokens=4`）· cudagraph 捕获档位至 128 · `--disable-custom-all-reduce` · 关闭 prefix-caching |
| 关键环境变量 | `HSA_FORCE_FINE_GRAIN_PCIE=1` · `HF_HUB_OFFLINE=1` · `LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib` |
| 性能测试口径 | `run_bench_fixed.sh`（固定 token 序列，每档 2 轮预热 + 3 轮正式，指标取 3 轮均值） |
| 服务端 | port 18680 · served-model-name `qwen38-vllm021` |

**代码补丁链**（`patch/`，应用到镜像内 `dist-packages/vllm/`）：

| 补丁 | 内容 |
|---|---|
| `00-base-merge.patch` | base 镜像 → merge：INT8 KV 全链路、DCU 适配（gfx936 GEMM 等）、0.21 基线对齐（14 files） |
| `02-m1b-chunk-o.patch` | fla `chunk_o.py` TP2 shared-QK 触发条件修复（H 48/24、Hg 16/8） |
| `03-m1b-rocm.patch` | `rocm.py` 追加 `on_gfx936/942/950` 内核函数 |
| `04-fix1-attention.patch` | `attention.py` `unified_kv_cache_update` getattr 兜底（num_decodes/num_extends）|

---

## 第 2 段 · 基线对比

### 2.1 官方基线（赛题文档 §4.1.3，BW100 行）

| 卡型 | 测试场景 | SLA 条件 | 官方最大并发 Baseline |
|---|---|---|---|
| BW100 | 4K 输入 / 1K 输出 | TTFT < 5s 且 TPOT < 50ms | **1** |
| BW100 | 64K 输入 / 1K 输出 | TTFT < 20s 且 TPOT < 60ms | **0**（官方未通过）|

官方文档仅给出 SLA 最大并发数，未公开 BW100 基线的 TPS / TTFT / TPOT 明细，
故第 4 段的对比以 **SLA 最大并发**为统一口径，TPS 等数据仅作本方成绩记录。

### 2.2 为什么不引入自测基线

本机卡型（BW100）在官方基线覆盖范围内，按赛题规则直接以官方基线为分母。
官方 64K 档 Baseline 为 0（未通过），不存在可比较的数值分母，报告中如实标注为
「官方未通过 → 本方通过」。

### 2.3 提升率

| 场景 | 官方 Baseline | 本方案 | 提升 |
|---|---|---|---|
| 4K/1K SLA 最大并发 | 1 | **32** | **×32（+3100%）** |
| 64K/1K SLA 最大并发 | 0（未通过） | **1（通过）** | 官方 0 → 通过 |

---

## 第 3 段 · 优化方法

按收益排序，全部改动均可追溯（补丁在 `patch/`，启动配置在 `scripts/start_server.sh`）。

### 3.1 启动参数层（目标机实测调优）

- **MTP 投机解码**：`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`。
  Qwen3.8-27B 自带 MTP 层，vLLM 0.21 原生支持，零代码改动拿到解码加速（SLA 并发容量的主要来源）。
- **W8A8 量化 GEMM**：`--quantization compressed-tensors` + `VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=True`。
  INT8 权重走 DCU 量化 GEMM，TPOT 关键项。
- **DCU prefill 后端**：`--gdn-prefill-backend triton`（海光 gdn 路线）。
- **cudagraph 捕获档位**：覆盖 1~128 全并发（`[1,2,4,...,120,128]`），消除高并发下 cudagraph 重捕获抖动。
- **`--disable-custom-all-reduce` + 关 prefix-caching**：实测在 DCU/mtp 组合下更稳（目标机逐项验证）。

### 3.2 代码补丁层（`patch/` 全链，base 镜像 → 交付镜像）

- **INT8 KV 全链路**（00）：KV cache 相关路径的 INT8 支持补齐，配合 W8A8 减少显存/带宽压力。
- **DCU 架构适配**：gfx936 GEMM 适配（00）+ `on_gfx936/942/950` 内核注册（03）+ fla `chunk_o`
  TP2 shared-QK 触发条件修复（02，H/Hg 判断 48/16 → 兼容 24/8）。
- **稳定性修复**（04）：`unified_kv_cache_update` 对缺失字段的 getattr 兜底，避免 MTP/多 decode 场景下
  AttributeError（fix1 的由来）。

### 3.3 未采纳 / 待验证

- prefix-caching：实测关闭（`--no-enable-prefix-caching`）；ShareGPT 场景可再评估开启收益。
- 精度（GSM8K 等）：提交时未跑，见第 5 段 5.3。

---

## 第 4 段 · 数据表格

采样口径：`run_bench_fixed.sh`（固定 token 序列），每档 **2 轮预热 + 3 轮正式**，
指标取 3 轮均值；原始数据 `result/fixed/<scenario>/round-*/summary.json`、`requests.jsonl`。

### 4.1 4K 输入 / 1K 输出（SLA：TTFT < 5s 且 TPOT < 50ms）

| 并发 | TPS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|---|
| 1 | 47.9 | 0.72s | 0.72s | 20.7ms | 24.3ms | ✅ |
| 8 | 275.0 | 1.78s | 4.63s | 23.3ms | 31.6ms | ✅ |
| 16 | 399.8 | 2.25s | 9.55s | 31.4ms | 43.8ms | ✅ |
| 32 | 546.0 | 2.95s | 18.36s | 44.0ms | 70.6ms | ✅ |
| 64 | 617.5 | 8.31s | 35.89s | 76.3ms | 121.6ms | ❌ TTFT+TPOT |
| 128 | 641.1 | 105.99s | 142.96s | 82.3ms | 118.8ms | ❌ TTFT+TPOT |

**SLA 最大并发 = 32**（官方 1 → ×32）。
瓶颈拆解：c32 的 TPOT 44.0ms（余量 12%）、TTFT p50 2.95s（余量 41%）均通过，但
**TTFT p95 已到 18.36s**——高并发下 prefill 排队长尾是下一档（c64）失守的直接原因；
c64 起 TTFT 8.31s 与 TPOT 76.3ms 双双超线。吞吐在 c64→c128 仍单调上升（617→641 TPS），
说明解码未饱和，**瓶颈在 prefill 排队而非 decode**。

### 4.2 64K 输入 / 1K 输出（SLA：TTFT < 20s 且 TPOT < 60ms）

| 并发 | TPS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|---|
| 1 | 15.6 | 17.50s | 17.53s | 49.1ms | 56.3ms | ✅ |
| 8 | 34.6 | 31.80s | 141.95s | 153.6ms | 268.8ms | ❌ TTFT+TPOT |
| 16 | 39.0 | 102.33s | 351.58s | 238.3ms | 421.3ms | ❌ |
| 32 | 39.8 | 491.23s | 693.05s | 267.5ms | 403.1ms | ❌ |

**SLA 最大并发 = 1**（官方 0 → 通过）。
如实说明：c1 的 TTFT 17.50s 距 20s 门槛仅 **12.5% 余量**、TPOT 49.1ms 余量 18%，属贴线通过；
64K 超长 prefill 是明显短板——c8 起 TTFT p50 即 31.8s 失守，长尾 p95 达 142s。
该场景的增量全部来自「跑通 + 单并发达标」，与 4K 档的并发扩展形成反差：
4K 吃 MTP 解码红利 + prefill 排队仍可控；64K 主要吃 prefill，MTP 收益有限。

### 4.3 与官方基线的总对照

| 场景 | 官方 SLA 并发 | 本方案 | 提升 |
|---|---|---|---|
| 4K/1K | 1 | 32 | ×32 |
| 64K/1K | 0 | 1 | 官方未通过 → 通过 |

---

## 第 5 段 · 结论与适用边界

### 5.1 结论

在 2 × 海光 DCU BW100、Qwen3.8-27B（W8A8 INT8）、vLLM 0.21.0 上，以**官方基线**为分母：

| 指标 | 结果 |
|---|---|
| 4K/1K SLA 最大并发 | 官方 1 → **32（×32）** |
| 64K/1K SLA 最大并发 | 官方 0（未通过）→ **1（通过）** |
| 4K c32 性能 | 546 TPS / TTFT 2.95s / TPOT 44.0ms |
| 64K c1 性能 | 15.6 TPS / TTFT 17.5s / TPOT 49.1ms |

### 5.2 适用边界（必读）

- **64K 超长输入是短板**：并发 >1 即失守，TTFT 17.5s 贴线。若评测更看重 64K 高并发，
  本配置不具备优势（长 prefill 排队未解决）。
- **TTFT 长尾**：4K c32 的 p95=18.36s 远高于 p50=2.95s，高并发下首字延迟抖动大；
  若评审关注 p95 而非 p50，4K 档边界会显著收紧。
- **W8A8 专用**：本配置针对 Channel-wise INT8 权重设计（compressed-tensors + 量化 GEMM 强制开启），
  换非量化权重需移除 `VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM` 与 `--quantization` 参数。
- **关闭 prefix-caching**：实测配置未开；若评测负载含高比例共享前缀，需重新评估。

### 5.3 精度校验（GSM8K 已补测）

- **GSM8K**（OpenCompass `gsm8k_gen_1d7fe4`，4-shot CoT，精确数字匹配）：
  **95.4511%**（1319/1319），官方基线 97.0% / 阈值 95.0% → **PASS**（降幅 1.55pp，阈值余量 0.45pp）。
- 实测命令：`scripts/run_accuracy.sh run --benchmarks gsm8k --concurrency 8 --output-dir result/accuracy/gsm8k`
- 原始数据：`result/accuracy/gsm8k/gsm8k/`（`result.json` + `predictions.jsonl` + `evaluation.json`），日志 `logs/gsm8k_run.log`
- 结论：W8A8 量化 + MTP 投机解码未造成精度跌破阈值（95.0），量化精度风险可控。
- 其余三项（LiveCodeBench / RealWorldQA / MathVision）按当前提交范围**不执行**，
  报告仅覆盖 GSM8K 精度校验。

### 5.4 未吃到的收益（后续可挖掘）

- 64K prefill：chunked prefill 参数（`--max-num-batched-tokens` 16K 的档位扫描）、
  prefill/decode 分离调度、64K 前缀复用（若共享题干占比高，可考虑重开 prefix-caching）。
- MTP 步数扫描：4 → 6/8 档位对比（当前实测 4）。
- cudagraph capture sizes 已覆盖 128，但 c64+ 的 TTFT 长尾提示 prefill 侧另有调度瓶颈。

---

## 附录 · 原始数据索引

- 性能原始数据：`result/fixed/`（10 场景 × round-01..03 + warmup-01/02，共 130 文件）
- 启动配置：`scripts/start_server.sh` · 压测脚本：`scripts/run_bench_fixed.sh`
- 补丁：`patch/`（00/02/03/04 + combined-base，含 `apply.sh`/`verify.sh`/`SHA256SUMS`）
