#!/usr/bin/env bash
# qwen35-merge-m1b-fix1 补丁应用脚本（base 镜像路径，真实交付链路）
# 用法:
#   ./apply.sh base          base 镜像 vllm/ 上重建      (00+02+03+04)  ← 推荐
#   ./apply.sh incremental   已含 merge 内容的 vllm/ 热补丁 (02+03+04)
#   ./apply.sh combined-base 单文件合并补丁
# 环境: VLLM_DIR 默认 /usr/local/lib/python3.10/dist-packages
set -euo pipefail
VLLM_DIR="${VLLM_DIR:-/usr/local/lib/python3.10/dist-packages}"
PATCHES_DIR="$(cd "$(dirname "$0")" && pwd)/patches"
mode="${1:-base}"
command -v patch >/dev/null || { echo "ERROR: patch 命令不存在"; exit 1; }
case "$mode" in
  base)          files="00-base-merge.patch 02-m1b-chunk-o.patch 03-m1b-rocm.patch 04-fix1-attention.patch" ;;
  incremental)   files="02-m1b-chunk-o.patch 03-m1b-rocm.patch 04-fix1-attention.patch" ;;
  combined-base) files="combined-base-qwen35-m1b-fix1.patch" ;;
  *) echo "usage: $0 {base|incremental|combined-base}"; exit 1 ;;
esac
for p in $files; do
  echo "==> $p"
  patch -p1 -d "$VLLM_DIR" < "$PATCHES_DIR/$p"
done
echo "==> DONE ($mode)"
