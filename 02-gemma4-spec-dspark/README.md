# Bài 02 — Speculative decoding với DSpark

**Mục tiêu.** Giữ nguyên cấu hình bài 01 và bật speculative decoding bằng drafter
DSpark trên **1 GPU**, rồi so sánh trực tiếp với kết quả bài 01: TPOT giảm bao
nhiêu, throughput đổi ra sao, và drafter chiếm bao nhiêu KV cache.

**Thời gian.** ~45 phút: deploy 3–5 phút, quét concurrency ~30 phút.

| Thành phần | Giá trị |
|---|---|
| Model | `/models/unsloth/gemma-4-12b-it-NVFP4` |
| Drafter | `/models/speculators/deepseek-ai/dspark_gemma4_12b_block7` |
| `--speculative-config` | `{"method":"dspark","model":"…/dspark_gemma4_12b_block7","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN"}` |
| Deployment / Service | `vllm-g4-spec`, cổng `8000` |
| GPU | 1 |
| Kết quả sweep | `/results/02-g12-dspark-c*.json` |

## Kiến thức nền

Speculative decoding: drafter đoán trước một khối token, model chính kiểm tra cả
khối trong một bước và giữ lại phần đúng. Mỗi bước decode có thể sinh nhiều hơn
1 token, nên TPOT giảm khi GPU đang bị giới hạn bởi băng thông bộ nhớ (decode ở
`c` thấp).

- `num_speculative_tokens` bằng `block_size` của drafter (xem `config.json` của drafter).
- `attention_backend` trong `--speculative-config` chỉ áp cho drafter. Ở chế độ
  `--language-model-only`, nếu để tự chọn thì drafter lấy `FLASHINFER` và server
  treo sau bước capture CUDA graph (GPU 100%, không bao giờ mở cổng 8000) trên
  RTX PRO 6000 với vLLM 0.29.0. Ghim `TRITON_ATTN` cho cùng backend với model chính.

## Điều kiện tiên quyết

- Đã chạy [bài 01](../01-gemma4-baseline-agg/README.md) và có `/results/01-g12-agg-c*.json`.
- Đã **xoá deployment bài 01** để GPU rảnh.

```bash
kubectl exec -n token-factory deploy/bench-client -- sh -c 'ls /results/01-g12-agg-c*.json | wc -l'
kubectl get pods -n token-factory | grep vllm || echo "GPU rảnh"
cd 02-gemma4-spec-dspark
```

**Mong đợi:** `11` file kết quả của bài 01, và dòng `GPU rảnh`.

## Bước 1 — Deploy

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deploy/vllm-g4-spec -n token-factory --timeout=15m
```

**Mong đợi:** `deployment "vllm-g4-spec" successfully rolled out`.

**Nếu không đúng:** pod chạy nhưng không bao giờ sẵn sàng, GPU 100% → kiểm tra
`attention_backend` của drafter (xem Kiến thức nền).

## Bước 2 — Xác nhận drafter được nạp

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=1500 \
  | grep -E "speculative_config=|Available KV cache|KV cache size|multimodal embeddings" \
  | cut -c1-250
```

**Mong đợi:** thấy dòng `speculative_config=` nhắc tới `dspark` và
`num_speculative_tokens=7`, cùng hai dòng KV cache. So KV cache với bài 01 và ghi
vào bảng ở cuối bài. Tham khảo trên RTX PRO 6000: khoảng 1,78 triệu token (bài 01
là 2,68 triệu), vì drafter cũng chiếm bộ nhớ.

## Bước 3 — Smoke test

```bash
kubectl exec -n token-factory deploy/bench-client -- curl -s \
  http://vllm-g4-spec:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-12b","messages":[{"role":"user","content":"Viet ham Python tinh Fibonacci thu n. Chi code."}],"max_tokens":200}'
```

**Mong đợi:** câu trả lời mạch lạc có code Python, như bài 01.

## Bước 4 — Xác nhận chế độ chỉ văn bản

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../01-gemma4-baseline-agg/test-images.py token-factory/$POD:/tmp/test-images.py
kubectl exec -n token-factory $POD -- python3 /tmp/test-images.py http://vllm-g4-spec:8000
```

**Mong đợi:** giống bài 01, cả ba dòng đều `400 ... At most 0 image(s)`.

## Bước 5 — Quét concurrency

```bash
kubectl delete job sweep-bench-spec -n token-factory --ignore-not-found
kubectl apply -f sweep-job.yaml
kubectl logs -n token-factory -f job/sweep-bench-spec
```

**Mong đợi:** như bài 01, mỗi mức in các chỉ số, kết thúc bằng
`=== xong: /results/02-g12-dspark-c*.json ===`. Mất ~30 phút. Trong lúc chờ, làm
Bước 6 ở một cửa sổ terminal khác.

## Bước 6 — Đo acceptance length

Trong lúc sweep đang chạy:

```bash
kubectl logs -n token-factory -l app=vllm-g4-spec --tail=3000 \
  | grep "Mean acceptance length" | tail -5
```

**Mong đợi:** các dòng `Mean acceptance length: ...`. Tham khảo trên RTX PRO 6000:
khoảng 3–4 token mỗi bước (tối đa 8 với 7 token đoán). Ghi một giá trị vào bảng
cuối bài.

## Bước 7 — So với bài 01

```bash
POD=$(kubectl get pod -n token-factory -l app=bench-client -o jsonpath='{.items[0].metadata.name}')
kubectl cp ../04-compare-results/compare-spec.py token-factory/$POD:/tmp/compare-spec.py
kubectl exec -n token-factory $POD -- python3 /tmp/compare-spec.py 01-g12-agg 02-g12-dspark baseline dspark
```

**Mong đợi:** bảng 11 dòng với cột tỉ lệ `B/A` cho output tok/s, TPOT, ITL, TTFT.
Với tok/s, `> 1` là DSpark tốt hơn; với TPOT, ITL, TTFT, `< 1` là DSpark tốt hơn.

## Điền bảng kết quả của bạn

| c | Output tok/s (01 → 02) | TPOT p50 ms (01 → 02) | ITL p50 ms (01 → 02) | TTFT p50 ms (01 → 02) |
|---:|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |
| 128 | | | | |

| | Bài 01 | Bài 02 |
|---|---|---|
| GPU KV cache size (tokens) | | |
| Mean acceptance length | — | |

## Câu hỏi để quan sát

- TPOT giảm bao nhiêu lần? Mức giảm thay đổi thế nào khi `c` tăng?
- ITL của bài 02 cao hơn hay thấp hơn bài 01? Vì sao? (Gợi ý: mỗi lần trả về có thể mang nhiều token.)
- KV cache thay đổi thế nào khi thêm drafter, và ảnh hưởng gì tới TTFT ở `c` cao?

## Dọn dẹp

```bash
kubectl delete -f deployment.yaml
kubectl delete -f sweep-job.yaml --ignore-not-found
```

---

**Tiếp theo:** [Bài 03 — PD disaggregation + DSpark](../03-gemma4-pd-spec/README.md)
