#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: start_server.sh
# 脚本功能: 启动 vLLM 推理服务 (海光 BW100 双卡 + Qwen3.8-27B-Channel-INT8-w8a8)
# 默认端口: 18680
# 说明: 参数与目标机实测跑通的启动命令一致（MTP 投机解码开启、
#       prefix-caching 关闭、gdn-prefill-backend=triton、cudagraph 128 档）
# ==============================================================================

set -euo pipefail

# 基础目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 环境变量 (与目标机实测一致)
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1"}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export HSA_FORCE_FINE_GRAIN_PCIE=1
# INT8 W8A8 (compressed-tensors) 量化 GEMM 必需
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=True
# DTK rocm_smi 库路径（非交互 shell 必需）
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}

# 服务配置参数 (默认值与目标机实测一致)
MODEL_PATH=${MODEL_PATH:-"/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"qwen38-vllm021"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-18680}
TP_SIZE=${TP_SIZE:-2}
DTYPE=${DTYPE:-"bfloat16"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
# MTP 投机解码 (vLLM 0.21 原生，目标机实测开启)
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-'{"method":"mtp","num_speculative_tokens":4}'}
# cudagraph 捕获档位（覆盖到最大并发 128）
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_capture_sizes":[1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]}'}

LOG_FILE="${LOG_DIR}/server.log"

echo "================================================================"
echo "准备启动 vLLM 推理服务"
echo "硬件设备: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} (海光 BW100)"
echo "模型路径: ${MODEL_PATH}"
echo "服务名称: ${SERVED_MODEL_NAME}"
echo "服务地址: http://${HOST}:${PORT}"
echo "并行规模: TP=${TP_SIZE} | dtype=${DTYPE}"
echo "上下文长: ${CONTEXT_LENGTH} | batch_tokens=${MAX_NUM_BATCHED_TOKENS} | seqs=${MAX_NUM_SEQS}"
echo "MTP: ${SPECULATIVE_CONFIG}"
echo "日志输出: ${LOG_FILE}"
echo "================================================================"

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "警告: 模型路径 ${MODEL_PATH} 不存在或未挂载，请检查！"
fi

RUN_IN_BACKGROUND=false
if [ "${1:-}" = "--daemon" ] || [ "${1:-}" = "-d" ]; then
  RUN_IN_BACKGROUND=true
fi

LAUNCH_CMD=(
  python3 -u -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --dtype "${DTYPE}"
  --quantization compressed-tensors
  --max-model-len "${CONTEXT_LENGTH}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
  --disable-custom-all-reduce
  --no-enable-prefix-caching
  --gdn-prefill-backend triton
  --speculative-config "${SPECULATIVE_CONFIG}"
  --compilation-config "${COMPILATION_CONFIG}"
)

if [ "${RUN_IN_BACKGROUND}" = true ]; then
  echo "正在后台启动服务，日志记录至: ${LOG_FILE}"
  nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  SERVER_PID=$!
  echo "服务 PID: ${SERVER_PID}"

  echo "等待服务就绪 (轮询 http://127.0.0.1:${PORT}/health)..."
  MAX_RETRIES=120
  COUNT=0
  until curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 || [ ${COUNT} -eq ${MAX_RETRIES} ]; do
    sleep 3
    COUNT=$((COUNT + 1))
    echo -n "."
  done
  echo ""

  if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "vLLM 推理服务已就绪！"
  else
    echo "警告: 等待超时，服务可能仍在初始化或发生错误，请查看日志: ${LOG_FILE}"
  fi
else
  # 前台执行
  exec "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
