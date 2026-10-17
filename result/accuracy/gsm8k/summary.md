| Benchmark | 基线 % | 阈值 % | 实测 % | 降幅 pp | 完成题数 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| GSM8K | 97.0 | 95.0 | 95.4511 | 1.5489 | 1319/1319 | PASS |
| LiveCodeBench v6 | 90.3 | 88.3 | — | — | 0/175 | MISSING |
| RealWorldQA | 85.9 | 83.9 | — | — | 0/765 | MISSING |
| MathVision (Without CI) | 90.0 | 88.0 | — | — | 0/304 | MISSING |

整体判定：**INCOMPLETE**

分数以百分数记录，降幅单位为百分点；使用未舍入分数判定。负降幅表示提升。
本报告验证所选协议下的数值阈值；官方基线的完整生成/判分配置尚未提供，不能证明与官方分数完全同口径。
MathVision local 使用规则提取 + VLMEvalKit 数学匹配；api 使用 VLMEvalKit LLM 答案提取。二者须分别比较。

