#!/usr/bin/env bash
# Single benchmark run for the live demo: concurrency 24, 24 prompts (one wave).
# Runs inside the bench-client pod:
#   kubectl exec -i deploy/bench-client -- bash -s -- <LAB> <BASE_URL> [SEED] [NPROMPTS] < bench-c24.sh
set -euo pipefail

LAB="${1:?lab name, e.g. demo-01-agg}"
BASE="${2:?base url, e.g. http://vllm-g4-agg:8000}"
SEED="${3:-0}"
NPROMPTS="${4:-24}"
C=24

OUT="/results/${LAB}-c${C}.json"
echo "=========== ${LAB}  c=${C}  n=${NPROMPTS}  seed=${SEED} ==========="

vllm bench serve \
  --backend openai-chat \
  --base-url "$BASE" \
  --endpoint /v1/chat/completions \
  --model gemma4-12b \
  --tokenizer /models/unsloth/gemma-4-12b-it-NVFP4 \
  --dataset-name speed_bench \
  --dataset-path /datasets/speed-bench \
  --speed-bench-dataset-subset throughput_8k \
  --num-prompts "$NPROMPTS" \
  --max-concurrency "$C" \
  --request-rate inf \
  --seed "$SEED" \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-filename "$OUT"

echo "=== saved: ${OUT} ==="

python3 - "$OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("\n============ Ket qua (don vi giay) ============")
print(f"Output throughput (tok/s):  {d['output_throughput']:.1f}")
print(f"TPOT median (s):            {d['median_tpot_ms'] / 1000:.4f}")
print(f"TTFT median (s):            {d['median_ttft_ms'] / 1000:.2f}")
print(f"E2E latency median (s):     {d['median_e2el_ms'] / 1000:.2f}")
print(f"E2E latency p99 (s):        {d['p99_e2el_ms'] / 1000:.2f}")
print(f"Benchmark duration (s):     {d['duration']:.1f}")
if d.get("spec_decode_acceptance_length"):
    print(f"Acceptance length:          {d['spec_decode_acceptance_length']:.2f}")
print(f"Successful requests:        {d['completed']}/{d['num_prompts']}")
print("===============================================")
EOF
