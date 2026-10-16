# Bài 01 — Baseline: Gemma-4-12B ở chế độ agg

**Mục tiêu.** Chạy Gemma-4-12B NVFP4 ở chế độ aggregated (một engine vừa prefill
vừa decode) trên **1 GPU**, không speculative decoding, rồi đo throughput, TPOT,
ITL và TTFT ở 11 mức concurrency. Đây là **mốc so sánh** cho bài 02 và 03.

**Thời gian.** ~45 phút: deploy 3–5 phút, quét concurrency ~30 phút.

| Thành phần | Giá trị |
|---|---|
| Model | `/models/unsloth/gemma-4-12b-it-NVFP4` |
| Deployment / Service | `vllm-g4-agg`, cổng `8000` |
| GPU | 1 |
| Kết quả sweep | `/results/01-g12-agg-c*.json` |

## Điều kiện tiên quyết

- Đã hoàn thành [bài 00](../00-prerequisites/README.md), gồm cả Bước 7 (thư viện offline).
- 1 GPU rảnh trên node có nhãn `fpt-transform-lab=true`.

Kiểm tra nhanh, rồi vào thư mục bài:

```bash
kubectl get deploy bench-client -n token-factory
kubectl get pods -n token-factory | grep vllm || echo "GPU rảnh"
cd 01-gemma4-baseline-agg
```

**Mong đợi:** `bench-client` `READY 1/1`, và dòng `GPU rảnh` (không còn pod vLLM
nào của bài khác).

## Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-agg -n token-factory --timeout=15m
```

**Mong đợi:** `deployment "vllm-g4-agg" successfully rolled out`. Lần đầu mất vài
phút vì vLLM nạp trọng số và capture CUDA graph.

**Nếu không đúng:** pod `Pending` → `kubectl describe pod -l app=vllm-g4-agg -n
token-factory`; thường là GPU đang bị pod khác giữ, hoặc node chưa có nhãn.

## Bước 2 — Đọc dung lượng KV cache

```bash
kubectl logs -n token-factory -l app=vllm-g4-agg --tail=1000 \
  | grep -E "Available KV cache|KV cache size"
```

**Mong đợi:** hai dòng `Available KV cache memory: ... GiB` và `GPU KV cache
size: ... tokens`. Ghi lại hai con số này vào bảng ở cuối bài. Tham khảo trên RTX
PRO 6000 96 GB: khoảng 2,68 triệu token.

## Bước 3 — Smoke test text

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-agg:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'
```

**Mong đợi:** JSON có `choices[0].message.content` là câu trả lời mạch lạc có code
Python, không phải chuỗi ký tự lộn xộn.

## Bước 4 — Xác nhận chế độ chỉ văn bản

Server chạy `--language-model-only`: SPEED-Bench chỉ có văn bản, nên bỏ hẳn
phần xử lý ảnh/âm thanh (không nạp vision encoder, không cần chế độ attention
`mm_prefix` cho token ảnh). Mọi request có ảnh phải bị từ chối:

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-agg:8000
```

**Mong đợi:**

```
[ 8 anh] 400  ... At most 0 image(s) may be provided in one prompt.
[ 9 anh] 400  ... At most 0 image(s) may be provided in one prompt.
[10 anh] 400  ... At most 0 image(s) may be provided in one prompt.
```

## Bước 5 — Quét concurrency

```bash
kubectl delete job sweep-bench-agg -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench-agg
```

Sweep chạy lần lượt `c = 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128` trên
`throughput_8k`; mỗi mức gửi `2 × c` request (tối thiểu 8, tối đa 256), mỗi
request sinh khoảng 4096 token. Mất ~30 phút, nên để chạy và làm việc khác;
`Ctrl+C` chỉ thoát phần xem log, job vẫn chạy.

**Mong đợi:** mỗi mức in `Successful requests`, `Output token throughput`, `TTFT`,
`TPOT`; cuối cùng là dòng `=== xong: /results/01-g12-agg-c*.json ===`.

**Nếu không đúng:** dòng `!! DANG CO benchmark khac chay` nghĩa là còn một sweep
khác đang chạy trên server, dừng nó trước (`kubectl get jobs -n token-factory`).
Mỗi server chỉ chạy **một** sweep tại một thời điểm.

## Bước 6 — Xem kết quả

```bash
kubectl exec -n token-factory deploy/bench-client -- python3 -c '
import json,glob,re
rows=[]
for f in glob.glob("/results/01-g12-agg-c*.json"):
    c=int(re.search(r"-c(\d+)\.json",f).group(1)); d=json.load(open(f))
    rows.append((c,d["output_throughput"],d["median_tpot_ms"],d["median_itl_ms"],d["median_ttft_ms"]))
print("%4s %9s %9s %9s %9s" % ("c","tok/s","TPOT ms","ITL ms","TTFT ms"))
for r in sorted(rows):
    print("%4d %9.1f %9.2f %9.2f %9.0f" % r)
'
```

**Mong đợi:** 11 dòng, từ `c=1` đến `c=128`. Tham khảo trên RTX PRO 6000: `c=1`
khoảng 97 tok/s, TPOT khoảng 10 ms; `c=128` khoảng 3,5k tok/s.

## Điền bảng kết quả của bạn

| c | Output tok/s | TPOT p50 (ms) | ITL p50 (ms) | TTFT p50 (ms) |
|---:|---:|---:|---:|---:|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |
| 128 | | | | |

| KV cache | Giá trị |
|---|---|
| Available KV cache memory | |
| GPU KV cache size (tokens) | |

## Câu hỏi để quan sát

- Throughput tăng thế nào khi `c` tăng, và bão hoà ở mức nào?
- TPOT ở `c = 1` là bao nhiêu? Đây là tốc độ tối đa một người dùng đơn lẻ nhận được.
- TTFT bắt đầu tăng mạnh từ mức `c` nào?

## Dọn dẹp

Giữ nguyên bảng kết quả của bạn rồi giải phóng GPU cho bài 02:

```bash
kubectl delete -f deployment.yaml
kubectl delete -f sweep-job.yaml --ignore-not-found
```

---

**Tiếp theo:** [Bài 02 — Speculative decoding với DSpark](../02-gemma4-spec-dspark/README.md)
