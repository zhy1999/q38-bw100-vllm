# 数据集说明（不入库，占位）

## ShareGPT_V3_unfiltered_cleaned_split.json

- **本机路径**：`/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`
- **大小**：642 MB（> GitHub 100MB 单文件限制，故以占位符入库）
- **来源**：huggingface `anon8231489123/ShareGPT_Vicuna_unfiltered`（ShareGPT V3 unfiltered cleaned split）
- **用途**：`scripts/run_bench_sharegpt.sh` 性能基准（C1/4/16/32/64 × 3 轮 × 1000 请求/轮）
- **运行方式**：本地用 `DATASET_PATH=/data/datasets/ShareGPT_V3_unfiltered_cleaned_split.json bash scripts/run_bench_sharegpt.sh`（或 docker 挂载 `-v /data/datasets:/dataset:ro`）
