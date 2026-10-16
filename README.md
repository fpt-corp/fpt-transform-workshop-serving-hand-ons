# FPT Transform — Token Factory: Tối ưu Serving cho Agentic Coding

Chuỗi bài hands-on tối ưu phục vụ model **Gemma-4-12B (NVFP4)** bằng **vLLM** trên
**Kubernetes**, đo bằng **`vllm bench serve`** với dataset **SPEED-Bench**.

Bạn đi từ một baseline chạy được, bật speculative decoding (DSpark), rồi tách
prefill/decode (PD disaggregation), và tự đo để so sánh các cấu hình. Mỗi bài là
một chuỗi bước **copy–chạy–kiểm tra**: lệnh nào cũng có kết quả mong đợi để bạn
biết mình đang đi đúng.

---

## 1. Lộ trình bài học

| Bài | Bạn làm gì | GPU | Thời gian (ước tính) | Kết quả đầu ra |
|---|---|---|---|---|
| [00](00-prerequisites/) | Chuẩn bị: nhãn node, namespace, secret, storage, trọng số, dataset, bench client | 0 | 30–60 phút | `bench-client` chạy được, dataset và trọng số đã sẵn sàng |
| [01](01-gemma4-baseline-agg/) | **Baseline** — vLLM aggregated, không speculative decoding | 1 | ~45 phút | `/results/01-g12-agg-c*.json` |
| [02](02-gemma4-spec-dspark/) | **Speculative decoding với DSpark** | 1 | ~45 phút | `/results/02-g12-dspark-c*.json` |
| [03](03-gemma4-pd-spec/) | **PD disaggregation + DSpark** — 1 GPU chia đôi bằng KAI Scheduler + HAMi-core (pod P, pod D, router), kèm [demo HAMi-core](03-gemma4-pd-spec/hami-demo/) | 1 | 60–90 phút | `/results/03-g12-pd-1gpu-kai-c*.json` |
| [04](04-compare-results/) | Tổng hợp và so sánh kết quả của bài 01–03 | 0 | ~15 phút | `summary.md`, `summary.csv` |
| [05](05-live-demo/) | Live demo 5 phút: baseline đã đo sẵn, chạy DSpark và điền bảng so sánh ở c=24 | 1 | ~10 phút | Bảng so sánh điền tay |

Hai bài chiếm nhiều thời gian nhất là phần **quét concurrency** (mỗi lần ~30
phút) và **nạp model lần đầu** (vài phút). Bạn có thể làm việc khác trong lúc
chờ, nhưng **không chạy hai sweep cùng lúc trên một server**: kết quả sẽ sai.

Các bài 01, 02, 03 đều cần GPU, nên **xoá deployment của bài trước** rồi mới làm
bài sau. Bài 05 chỉ chạy lại một mức c=24, dùng sau khi bạn đã hiểu bài 01–02.

## 2. Cấu hình dùng xuyên suốt

| Tham số | Giá trị |
|---|---|
| Engine | `vllm/vllm-openai:v0.29.0` |
| Model | `unsloth/gemma-4-12b-it-NVFP4` |
| Drafter DSpark (bài 02, 03) | `deepseek-ai/dspark_gemma4_12b_block7`, `num_speculative_tokens: 7` |
| `--tensor-parallel-size` | `1` |
| `--max-num-batched-tokens` | `8192` |
| `--max-num-seqs` | `128` |
| `--max-model-len` | `131072` (bài 03: `32768`) |
| `--attention-backend` | `TRITON_ATTN` (KV cache là FP8 theo checkpoint; trên SM120 `FLASH_ATTN` không nhận KV FP8). Drafter ghim cùng backend qua `"attention_backend":"TRITON_ATTN"` trong `--speculative-config` |
| Đầu vào | `--language-model-only` (chỉ văn bản; SPEED-Bench không có ảnh) |
| Dataset benchmark | SPEED-Bench `throughput_8k` |
| Mức concurrency quét | `1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128` |

## 3. Chỉ số cần ghi lại

| Chỉ số | Ý nghĩa |
|---|---|
| **Output throughput** (tok/s) | Công suất sinh token của cả server |
| **TPOT** p50 / p99 | Thời gian trung bình mỗi token sau token đầu (càng thấp càng tốt) |
| **ITL** p50 / p99 | Khoảng cách giữa hai lần trả token liên tiếp |
| **TTFT** p50 / p99 | Độ trễ tới token đầu tiên (càng thấp càng tốt) |
| **Acceptance length** | Số token được chấp nhận mỗi bước speculative (bài 02, 03) |

Mọi sweep lưu JSON vào PVC `bench-results` (`/results`), để bài 04 tổng hợp.

## 4. Yêu cầu

Kiểm tra từng mục trước khi bắt đầu:

| Yêu cầu | Cách kiểm tra |
|---|---|
| `kubectl` trỏ đúng cluster | `kubectl config current-context` |
| Cluster có node GPU, đã cài NVIDIA GPU Operator hoặc device plugin (workshop dùng RTX PRO 6000 96GB; xem "Phần cứng" ở mục 5) | bài 00 Bước 0 và Bước 8 |
| Một node GPU gắn nhãn `fpt-transform-lab=true`; mọi pod của chuỗi bài chọn node bằng nhãn này | bài 00 Bước 0 |
| 1 GPU rảnh cho mỗi bài 01, 02, 03 | `kubectl get pods -A` không còn pod khác giữ GPU |
| Bài 03: [KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) v0.17.1 (bật GPU sharing + plugin `hamicore`) và `kai-resource-isolator` 1.1.0. Cài một lần cho cả cụm, cần quyền cluster-admin | bài 03 Bước 1 |
| `HF_TOKEN`: model và drafter không bị gated, nhưng SPEED-Bench đầy đủ cần token đã được duyệt dataset gated `cais/hle`. Không có token thì chuẩn bị dataset bằng `--skip-gated` (1268/1536 bản ghi) | bài 00 Bước 2 và Bước 5 |
| Nhánh tự học: ~16 GB trống cho PVC trọng số (model 9,3 GB + drafter 6,9 GB) | bài 00 Bước 3B |

## 5. Môi trường workshop

| Đã chuẩn bị sẵn | Ở đâu |
|---|---|
| Trọng số model | Host path `/mnt/hps/fp8_models` trên node GPU (tên thư mục giữ từ bản FP8; bên trong là `unsloth/gemma-4-12b-it-NVFP4`) |
| Image Docker | Đã pull sẵn trên node (`imagePullPolicy: IfNotPresent`) |

**Phần cứng.** Workshop demo chạy trên **RTX PRO 6000 Blackwell 96 GB**; mọi số
liệu, KV cache và cấu hình trong bài đều đo trên card này. Khi **tự thực hành**,
bạn có thể triển khai trên bất kỳ dòng GPU nào, với các lưu ý:

- Kết quả benchmark sẽ khác bảng mẫu; so sánh giữa các bài trên cùng máy của bạn
  mới có ý nghĩa.
- Checkpoint NVFP4 chỉ dùng kernel FP4 gốc trên GPU Blackwell. Trên GPU khác, hãy
  dùng checkpoint phù hợp (ví dụ FP8 hoặc BF16) thay cho
  `unsloth/gemma-4-12b-it-NVFP4`.
- Cần chỉnh theo dung lượng và kiến trúc card: `--gpu-memory-utilization`,
  `--max-model-len`, `--attention-backend`, và ở bài 03 là kích thước slice
  `gpu-memory` (ghi theo card 96 GB).

Bài 00 có hai nhánh: **Workshop** (bọc host path thành PVC) và **Tự học**
(tải từ HuggingFace). Từ bài 01 trở đi, hai nhánh chạy giống hệt nhau.

## 6. Bắt đầu

```bash
git clone https://github.com/fpt-corp/fpt-transform-workshop-serving-hand-ons.git
cd fpt-transform-workshop-serving-hand-ons/00-prerequisites
```

Rồi làm theo [00-prerequisites/README.md](00-prerequisites/README.md).

**Cách đọc mỗi bài.** Các bài 01–04 có cùng khung: *Mục tiêu → Điều kiện tiên
quyết → Các bước → Điền bảng kết quả → Câu hỏi → Dọn dẹp*. Mỗi bước có lệnh, rồi
phần **Mong đợi** (kết quả đúng) và, khi cần, **Nếu không đúng**. Dừng lại ở bước
nào chưa ra kết quả mong đợi, đừng làm tiếp.

**Quy ước lệnh.** Mọi lệnh `kubectl` ghi rõ `-n token-factory`, và mỗi bài bắt
đầu bằng `cd` vào thư mục của bài. Không chạy `kubectl logs -f` ngay sau
`kubectl apply`: luôn `rollout status` hoặc `wait` trước.

**Giữa các bài**, xoá deployment của bài trước để giải phóng GPU:

```bash
kubectl get deploy -n token-factory
kubectl delete deploy <tên> -n token-factory
```

## 7. Kết quả benchmark: Baseline (bài 01) vs DSpark (bài 02)

Đo trên 1 GPU RTX PRO 6000 Blackwell, SPEED-Bench `throughput_8k`, số request
mỗi mức = `2 × concurrency` (tối thiểu 8). TPOT và TTFT là giá trị median.
Cột so sánh = DSpark / Baseline: **Throughput gain ↑** càng cao càng tốt,
**TPOT ratio ↓** và **TTFT ratio ↓** càng thấp càng tốt.

| Concurrency | Baseline tok/s | DSpark tok/s | **Throughput gain ↑** | Baseline TPOT (ms) | DSpark TPOT (ms) | **TPOT ratio ↓** | Baseline TTFT (ms) | DSpark TTFT (ms) | **TTFT ratio ↓** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 96.7 | 190.0 | **1.96x** | 10.25 | 4.48 | **0.44x** | 366 | 380 | 1.04x |
| 2 | 192.1 | 324.2 | **1.69x** | 10.39 | 4.67 | **0.45x** | 52 | 426 | 8.24x |
| 4 | 371.8 | 429.0 | 1.15x | 10.74 | 5.00 | **0.47x** | 77 | 88 | 1.15x |
| 8 | 692.7 | 597.8 | 0.86x | 11.23 | 5.38 | **0.48x** | 413 | 295 | 0.72x |
| 16 | 1221.1 | 1339.0 | 1.10x | 12.50 | 6.14 | **0.49x** | 493 | 756 | 1.53x |
| 24 | 1314.4 | 1997.8 | **1.52x** | 17.72 | 7.13 | **0.40x** | 246 | 825 | 3.35x |
| 32 | 1673.6 | 2286.6 | 1.37x | 18.48 | 7.92 | **0.43x** | 291 | 501 | 1.72x |
| 48 | 2195.1 | 2735.5 | 1.25x | 20.50 | 10.51 | **0.51x** | 454 | 646 | 1.42x |
| 64 | 2667.2 | 3060.0 | 1.15x | 22.63 | 12.98 | 0.57x | 554 | 891 | 1.61x |
| 96 | 3061.1 | 3452.0 | 1.13x | 28.54 | 18.61 | 0.65x | 875 | 1293 | 1.48x |
| 128 | 3540.4 | 3173.2 | 0.90x | 33.58 | 26.26 | 0.78x | 1009 | 5627 | 5.58x |

## 8. Kết quả benchmark: DSpark (bài 02) vs PD + DSpark trên 1 GPU (bài 03)

Cùng điều kiện đo như mục 7. Bài 03 chia một card cho P và D bằng KAI Scheduler +
HAMi-core (0,47 + 0,47 card), KV đi thẳng trong GPU bằng `cuda_ipc`. Cột so sánh
= PD / DSpark.

| Concurrency | DSpark tok/s | PD tok/s | **Throughput gain ↑** | DSpark TPOT (ms) | PD TPOT (ms) | **TPOT ratio ↓** | DSpark TTFT (ms) | PD TTFT (ms) | **TTFT ratio ↓** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 190.0 | 196.3 | 1.03x | 4.48 | 4.46 | 0.99x | 380 | 447 | 1.18x |
| 2 | 324.2 | 350.3 | 1.08x | 4.67 | 4.85 | 1.04x | 426 | 963 | 2.26x |
| 4 | 429.0 | 696.7 | **1.62x** | 5.00 | 4.67 | 0.93x | 88 | 237 | 2.68x |
| 8 | 597.8 | 682.4 | 1.14x | 5.38 | 5.25 | 0.98x | 295 | 668 | 2.26x |
| 16 | 1339.0 | 1339.0 | 1.00x | 6.14 | 6.39 | 1.04x | 756 | 2142 | 2.84x |
| 24 | 1997.8 | 1580.4 | 0.79x | 7.13 | 7.50 | 1.05x | 825 | 2021 | 2.45x |
| 32 | 2286.6 | 2287.6 | 1.00x | 7.92 | 8.20 | 1.03x | 501 | 1212 | 2.42x |
| 48 | 2735.5 | 2616.1 | 0.96x | 10.51 | 11.25 | 1.07x | 646 | 1599 | 2.48x |
| 64 | 3060.0 | 2620.4 | 0.86x | 12.98 | 14.70 | 1.13x | 891 | 4905 | 5.51x |
| 96 | 3452.0 | 2479.7 | 0.72x | 18.61 | 18.49 | 0.99x | 1293 | 41932 | 32.43x |
| 128 | 3173.2 | 2134.4 | 0.67x | 26.26 | 21.70 | **0.83x** | 5627 | 103438 | 18.38x |

Chạy lại sweep PD lần hai cho kết quả lệch trong khoảng ±10% ở hầu hết các mức
(riêng c=96 lệch 17%, và TTFT ở c≥96 dao động mạnh); đỉnh ở c=4 lặp lại, nên đó
không phải nhiễu. Khi tự chạy, hãy coi chênh lệch dưới 10% giữa hai cấu hình là
ngang nhau.

Trên **1 GPU**, PD có TPOT ngang DSpark và throughput ngang DSpark đến c=48,
nhưng TTFT luôn chậm hơn, rồi tăng vọt từ c≥64 khi KV cache của D (nửa card) đầy.
P và D vẫn tranh nhau một card, lại phải nạp hai bản model, nên PD chỉ thực sự
có lợi khi P và D có GPU riêng. Phân tích chi tiết, và cách chuyển KV từ TCP
(TTFT 27 s ở c=24) sang `cuda_ipc` (2 s), xem
[bài 03](03-gemma4-pd-spec/README.md#bước-5--so-với-bài-02).

## 9. Tài liệu tham khảo

- [vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM — Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM — NixlConnector Usage](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [KAI Scheduler — GPU sharing](https://github.com/NVIDIA/KAI-Scheduler)
- [HAMi-core — giới hạn bộ nhớ GPU trong container](https://github.com/Project-HAMi/HAMi-core)
- [vLLM — `vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- Model: [`unsloth/gemma-4-12b-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4)
- Drafter: [`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7)
