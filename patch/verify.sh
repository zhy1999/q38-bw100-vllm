#!/usr/bin/env bash
# merge-m1b-fix1 状态锚点校验
VLLM_DIR="${VLLM_DIR:-/usr/local/lib/python3.10/dist-packages}"
ok=1
chk(){ if grep -q "$2" "$VLLM_DIR/$1" 2>/dev/null; then echo "OK   $1"; else echo "FAIL $1"; ok=0; fi; }
chk vllm/model_executor/layers/fla/ops/chunk_o.py "H in (48, 24)"
chk vllm/model_executor/layers/fla/ops/chunk_o.py "Hg in (16, 8)"
chk vllm/platforms/rocm.py "def on_gfx936"
chk vllm/platforms/rocm.py "def on_gfx942"
chk vllm/platforms/rocm.py "def on_gfx950"
chk vllm/model_executor/layers/attention/attention.py 'getattr(attn_metadata, "num_decodes", 0) == 0'
chk vllm/model_executor/layers/attention/attention.py '(getattr(attn_metadata, "num_prefills", 0) + getattr(attn_metadata, "num_extends", 0)) > 0'
if [ -L "$VLLM_DIR/vllm/model_executor/layers/fused_moe/modular_kernel.py" ]; then echo "OK   modular_kernel.py symlink"; else echo "FAIL modular_kernel.py symlink"; ok=0; fi
[ $ok -eq 1 ] && echo "ALL ANCHORS PASS" || { echo "ANCHOR FAILURES"; exit 1; }
