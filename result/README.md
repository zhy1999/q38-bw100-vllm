# vLLM 框架优化补丁（qwen35-merge-m1b-fix1） — 本目录同时存放评测结果与优化补丁
# vLLM 框架优化补丁（qwen35-merge-m1b-fix1）

> 来源：`/root/.openclaw/workspace/qwen35-merge-m1b-fix1/`（2026-10-16 交付）
> 适用镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
> 交付镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.21.0-qwen35-merge-m1b-fix1`
> 对应文档：《qwen35-merge-优化说明.md》

## 补丁链（base 镜像 → m1b-fix1）

| 补丁 | 说明 |
|---|---|
| 00-base-merge.patch | base → merge：INT8 KV 全链路、DCU 适配（gfx936 GEMM）、0.21 基线对齐（14 files）|
| 02-m1b-chunk-o.patch | fla chunk_o.py：use_share_qk3 TP2 触发条件 H==48→H in (48,24) / Hg==16→Hg in (16,8) |
| 03-m1b-rocm.patch | rocm.py：追加 on_gfx936/942/950 三函数 |
| 04-fix1-attention.patch | attention.py：unified_kv_cache_update getattr 兜底（num_decodes/num_extends）|
| combined-base-qwen35-m1b-fix1.patch | 上述合并单文件（17 files）|

## 应用

```bash
# base 镜像容器内：
VLLM_DIR=/usr/local/lib/python3.10/dist-packages bash apply.sh base
# 或单文件：
patch -p1 -d /usr/local/lib/python3.10/dist-packages < combined-base-qwen35-m1b-fix1.patch
bash verify.sh   # 8 锚点校验（VLLM_DIR 指向 vllm 包目录）
```

> 注意：容器内需 patch 命令；.so 二进制与 vllm_hcu 钩子不在此补丁范围（沿用 base 镜像）。

## 验证锚点（verify.sh）

- chunk_o.py `H in (48, 24)` / `Hg in (16, 8)`（323 行附近）
- rocm.py `def on_gfx936/942/950`
- attention.py `getattr(attn_metadata, "num_decodes", 0) == 0`
- modular_kernel.py 符号链接存在

2026-10-16 已用目标机实物验证：base + 00→02→03→04 = 目标机 vllm/ 逐文件一致。
