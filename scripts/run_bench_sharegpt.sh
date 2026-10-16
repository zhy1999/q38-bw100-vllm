#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_bench_sharegpt.sh
# 脚本功能: 自动化执行 ShareGPT 真实对话数据集基准测试（vLLM 客户端版）
# 测试规范:
#   - 并发档位: C1, C4, C16, C32, C64（可用 SHAREGPT_CONCURRENCIES 覆盖）
#   - 每档执行 3 轮正式测试，每轮 1000 个请求，2×并发进行预热
# 客户端: scripts/sharegpt_bench.py（零框架依赖，SSE 流式测 TTFT/TPOT/吞吐）
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_BASE_DIR="${PROJECT_DIR}/result/sharegpt"

BASE_URL=${BASE_URL:-"http://127.0.0.1:18680"}
MODEL_PATH=${MODEL_PATH:-"/model"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"qwen38-vllm021"}
DATASET_PATH=${DATASET_PATH:-"/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8"}  # 默认容器内模型路径；宿主机跑时指向 /data/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8

# vLLM openai backend 要求 base-url 带 /v1（sglang 不带，这里自动补）
VLLM_BASE="${BASE_URL%/}"
[[ "${VLLM_BASE}" != */v1 ]] && VLLM_BASE="${VLLM_BASE}/v1"

# vLLM openai backend 要求 base-url 带 /v1（sglang 不带，这里自动补）
VLLM_BASE="${BASE_URL%/}"
[[ "${VLLM_BASE}" != */v1 ]] && VLLM_BASE="${VLLM_BASE}/v1"

if [ ! -f "${DATASET_PATH}" ]; then
  echo "错误: 未找到 ShareGPT 数据集文件: ${DATASET_PATH}"
  echo "请设置环境变量 DATASET_PATH 指向实际路径，例如:"
  echo "  export DATASET_PATH=/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
  exit 1
fi

mkdir -p "${RESULT_BASE_DIR}"

CONCURRENCIES=(${SHAREGPT_CONCURRENCIES:-1 4 16 32 64})
NUM_PROMPTS=${NUM_PROMPTS:-1000}
ROUNDS=${ROUNDS:-3}

echo "================================================================"
echo "开始 ShareGPT 真实数据集基准测试（vLLM 客户端）"
echo "服务地址: ${VLLM_BASE}"
echo "模型名称: ${SERVED_MODEL_NAME}"
echo "数据集路径: ${DATASET_PATH}"
echo "并发档位: ${CONCURRENCIES[*]}"
echo "每档轮次: ${ROUNDS} 轮 (每轮 ${NUM_PROMPTS} 请求)"
echo "结果目录: ${RESULT_BASE_DIR}"
echo "================================================================"

for c in "${CONCURRENCIES[@]}"; do
  c_dir="${RESULT_BASE_DIR}/sharegpt-c${c}"
  mkdir -p "${c_dir}"
  warmup_reqs=$((2 * c))

  echo "----------------------------------------------------------------"
  echo ">>> 开始执行并发档位: C${c} (预热请求数: ${warmup_reqs})"
  echo "----------------------------------------------------------------"

  for r in $(seq 1 "${ROUNDS}"); do
    round_str=$(printf "%02d" "${r}")
    output_file="${c_dir}/round-${round_str}.jsonl"

    echo "[C${c}] 开始第 ${r}/${ROUNDS} 轮测试 -> ${output_file}"

    python3 -u "${PROJECT_DIR}/scripts/sharegpt_bench.py" \
      --base-url "${VLLM_BASE}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tokenizer-dir "${TOKENIZER_DIR}" \
      --dataset-path "${DATASET_PATH}" \
      --sharegpt-context-len "${SHAREGPT_CONTEXT_LEN:-4096}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${c}" \
      --warmup-requests "${warmup_reqs}" \
      --seed 20260912 \
      --output-file "${output_file}" 2>&1 | tee "${c_dir}/round-${round_str}.log"

    echo "[C${c}] 第 ${r}/${ROUNDS} 轮完成"
    sleep 5
  done
done

echo "================================================================"
echo "ShareGPT 评测全部完成！测试结果保存在: ${RESULT_BASE_DIR}"
echo "================================================================"
EOF
