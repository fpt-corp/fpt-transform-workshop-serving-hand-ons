# Live demo: Baseline (bài 01) vs DSpark (bài 02)

**Mục tiêu.** Trong ~5 phút: serve DSpark, chạy một lượt benchmark c=24 và điền tay
bảng so sánh với số baseline đã đo sẵn. Cần bài 00 hoàn tất và GPU rảnh.

| File | Vai trò |
|---|---|
| `bench-c24.sh` | Chạy `vllm bench serve` với c=24, n=24 trong pod `bench-client` |
| `compare-c24.py` | (Dự phòng) In bảng so sánh, baseline là số đã đo trước |

Mọi lệnh dưới đây chạy từ thư mục `05-live-demo/` của repo.

---

## 1. Chuẩn bị

### 1.1. Kiểm tra môi trường

```bash
alias k="kubectl -n token-factory"
cd 05-live-demo

k get deploy bench-client
k get pods -l 'app in (vllm-g4-agg,vllm-g4-spec)'
k exec deploy/bench-client -- ls -lh /datasets/speed-bench/throughput_8k.jsonl
```

| Lệnh | Kết quả đúng |
|---|---|
| `get deploy bench-client` | `READY 1/1` |
| `get pods -l ...` | `No resources found in token-factory namespace.` (GPU đang trống) |
| `ls -lh ...jsonl` | In ra một file, không báo `No such file` |

---

## 2. Kịch bản

| Phút | Việc | Thời gian thực đo |
|---|---|---|
| 0:00–1:30 | Mở đầu, chiếu cột Baseline của bảng kết quả | — |
| 1:30–2:30 | Serve DSpark | ~60 s |
| 2:30–4:00 | Bench DSpark | ~90 s |
| 4:00–5:00 | Điền bảng kết quả | — |

- Model Gemma-4-12B NVFP4, vLLM 0.29, 1 GPU RTX PRO 6000 Blackwell.
- Tải: 24 người dùng đồng thời, mỗi prompt khoảng 9k token, mỗi câu trả lời 4096 token.
- Baseline: mỗi bước decode chỉ sinh **1 token** cho mỗi request, TPOT ~0,0187 s.
  GPU bị giới hạn bởi băng thông bộ nhớ chứ không phải compute.

### Bước 2 — serve DSpark

```bash
grep -n "speculative-config" ../02-gemma4-spec-dspark/deployment.yaml
k apply -f ../02-gemma4-spec-dspark/deployment.yaml
k rollout status deploy/vllm-g4-spec --timeout=15m
```

**Mong đợi:** `grep` in ra dòng có `"method":"dspark"`, và lệnh cuối báo
`deployment "vllm-g4-spec" successfully rolled out` sau khoảng 60 giây.

### Bước 3 — Bench DSpark

```bash
k exec -i deploy/bench-client -- bash -s -- demo-02-dspark http://vllm-g4-spec:8000 < bench-c24.sh
```

**Mong đợi:** sau ~90 giây, khối `Ket qua (don vi giay)` in throughput, TPOT, TTFT, E2E,
thời gian chạy, acceptance length và `Successful requests: 24/24`. Chép các số đó
vào cột DSpark ở Bước 4.

### Bước 4 — Điền bảng kết quả

Tỉ lệ = DSpark ÷ Baseline; ↑ càng cao càng tốt, ↓ càng thấp càng tốt.

| Chỉ số | Baseline | DSpark | DSpark / Baseline |
|---|---|---|---|
| Output throughput (tok/s) ↑ | 1237,6 | | |
| TPOT median (s) ↓ | 0,0187 | | |
| TTFT median (s) ↓ | 2,97 | | |
| E2E latency median (s) ↓ | 79,37 | | |
| E2E latency p99 (s) ↓ | 79,43 | | |
| Benchmark duration (s) ↓ | 79,4 | | |
| Acceptance length (token/bước) | 1 | | — |
| Successful requests | 24/24 | | — |


Kiểm tra lại bằng script (tuỳ chọn): in cùng bảng với tỉ lệ đã tính sẵn.

```bash
k exec -i deploy/bench-client -- python3 - < compare-c24.py
```

---

## 3. Dọn dẹp

```bash
k delete deploy vllm-g4-spec
k exec deploy/bench-client -- sh -c 'rm -f /results/demo-* /results/warmup-*'
```
