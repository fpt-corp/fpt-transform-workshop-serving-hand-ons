"""Print the baseline vs DSpark table for the live demo (concurrency 24).

Baseline numbers are fixed from the rehearsal run (lab 01, c=24, n=24, seed 0),
so only DSpark has to run during the demo.

Run inside the bench-client pod:
  kubectl exec -i deploy/bench-client -- python3 - < compare-c24.py
"""
import json
import sys

BASELINE_N = 24  # baseline was one full wave: 24 prompts at concurrency 24
BASELINE = {
    "output_throughput": 1237.6,
    "median_tpot_ms": 18.66,
    "median_ttft_ms": 2966,
    "median_e2el_ms": 79370,
    "p99_e2el_ms": 79428,
    "duration": 79.4,
}
SPEC = "/results/demo-02-dspark-c24.json"

b = BASELINE
s = json.load(open(SPEC))

# Throughput and duration only compare when both runs send the same wave. A
# smoke test (n=1) left in SPEC measures one stream, not 24 in parallel.
if s["num_prompts"] != BASELINE_N or s["completed"] != BASELINE_N:
    sys.exit(
        f"!! {SPEC}: {s['completed']}/{s['num_prompts']} requests, baseline used "
        f"{BASELINE_N}/{BASELINE_N}. Re-run bench-c24.sh with the default NPROMPTS (24)."
    )

rows = [
    ("Output throughput (tok/s)", "output_throughput", "{:.1f}", "↑", 1),
    ("TPOT median (s)", "median_tpot_ms", "{:.4f}", "↓", 1000),
    ("TTFT median (s)", "median_ttft_ms", "{:.2f}", "↓", 1000),
    ("E2E latency median (s)", "median_e2el_ms", "{:.2f}", "↓", 1000),
    ("E2E latency p99 (s)", "p99_e2el_ms", "{:.2f}", "↓", 1000),
    ("Benchmark duration (s)", "duration", "{:.1f}", "↓", 1),
]

print(f"{'Metric':<28}{'Baseline':>12}{'DSpark':>12}{'DSpark/Base':>14}")
print("-" * 66)
for name, key, fmt, better, div in rows:
    ratio = s[key] / b[key]
    print(f"{name:<28}{fmt.format(b[key] / div):>12}{fmt.format(s[key] / div):>12}{ratio:>12.2f}x {better}")
if s.get("spec_decode_acceptance_length"):
    print(f"\nDSpark acceptance length: {s['spec_decode_acceptance_length']:.2f} tokens/step")
print(f"requests ok: dspark {s['completed']}/{s['num_prompts']}")
