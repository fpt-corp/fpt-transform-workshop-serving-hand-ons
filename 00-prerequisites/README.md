# Bài 00 — Chuẩn bị môi trường

**Mục tiêu.** Dựng nền cho toàn bộ chuỗi bài. Khi xong bạn có: một node GPU đã
gắn nhãn, namespace `token-factory`, secret HuggingFace, bốn PVC, trọng số model
và drafter, dataset SPEED-Bench, và pod `bench-client` để chạy benchmark.

**Thời gian.** 30–60 phút, phần lớn là chờ tải dataset (Bước 5).

Có **hai nhánh**, chỉ khác nhau ở Bước 3:

| Nhánh | Khi nào dùng |
|---|---|
| **A — Workshop** | Trọng số đã có sẵn tại `/mnt/hps/fp8_models` trên node (tên thư mục giữ từ bản FP8) |
| **B — Tự học** | Tự tải trọng số từ HuggingFace về PVC |

## Mục lục

0. [Chuẩn bị](#chuẩn-bị)
1. [Bước 0 — Gắn nhãn node chạy lab](#bước-0--gắn-nhãn-node-chạy-lab)
2. [Bước 1 — Namespace](#bước-1--namespace)
3. [Bước 2 — Secret HF_TOKEN](#bước-2--secret-hf_token)
4. [Bước 3A — Storage (workshop)](#bước-3a--storage-workshop) hoặc [Bước 3B — Storage + tải trọng số (tự học)](#bước-3b--storage--tải-trọng-số-tự-học)
5. [Bước 4 — Xác minh trọng số](#bước-4--xác-minh-trọng-số)
6. [Bước 5 — Dataset benchmark](#bước-5--dataset-benchmark)
7. [Bước 6 — Bench client](#bước-6--bench-client)
8. [Bước 7 — Thư viện offline cho sweep job](#bước-7--thư-viện-offline-cho-sweep-job)
9. [Bước 8 — Kiểm tra GPU](#bước-8--kiểm-tra-gpu)
10. [Xử lý sự cố](#xử-lý-sự-cố)

---

## Chuẩn bị

```bash
git clone https://github.com/fpt-corp/fpt-transform-workshop-serving-hand-ons.git
cd fpt-transform-workshop-serving-hand-ons/00-prerequisites
```

Mọi lệnh trong bài này chạy từ thư mục `00-prerequisites/`.

Kiểm tra `kubectl` trỏ đúng cluster và cluster thấy GPU:

```bash
kubectl config current-context
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.'nvidia\.com/gpu'
```

**Mong đợi:** có ít nhất một node có cột `GPU` bằng `1` trở lên. Không có thì
dừng lại và báo quản trị viên cụm: device plugin chưa chạy.

Nhu cầu GPU của từng bài:

| Bài | GPU |
|---|---|
| 01, 02 | 1 |
| 03 | 1 (chia đôi bằng KAI + HAMi-core, bài 03 Bước 1 hướng dẫn kiểm tra) |
| 04 | 0 |
| 05 | 1 |

Phiên bản dùng xuyên suốt:

| Thành phần | Giá trị |
|---|---|
| vLLM | `vllm/vllm-openai:v0.29.0` |
| Model | `unsloth/gemma-4-12b-it-NVFP4` |
| Drafter DSpark | `deepseek-ai/dspark_gemma4_12b_block7` |

## Bước 0 — Gắn nhãn node chạy lab

Mọi pod của chuỗi bài có `nodeSelector: fpt-transform-lab: "true"`, và PV ở
Bước 3A có `nodeAffinity` theo cùng nhãn (hostPath chỉ có dữ liệu trên đúng node
đó). Gắn nhãn cho **một** node GPU:

```bash
export NODE=<tên-node-gpu>
kubectl get node $NODE -L nvidia.com/gpu.product,nvidia.com/gpu.family
kubectl label node $NODE fpt-transform-lab=true
kubectl get nodes -l fpt-transform-lab=true
```

**Mong đợi:** lệnh cuối in đúng **một** dòng, là node bạn vừa gắn nhãn. Nếu báo
`already has a value`, nhãn đã có sẵn từ trước: kiểm tra xem đó có đúng node bạn
định dùng không rồi đi tiếp.

Nhãn có sẵn kiểu `gpu-type=...` có thể trùng trên nhiều node, nên không dùng để ghim.

## Bước 1 — Namespace

```bash
kubectl get ns token-factory >/dev/null 2>&1 || kubectl apply -f 00-namespace.yaml
kubectl config set-context --current --namespace=token-factory
kubectl get pods,deploy,svc,job,pvc -n token-factory
```

**Mong đợi:** `No resources found in token-factory namespace.` (namespace mới) hoặc
danh sách tài nguyên có sẵn.

> Nhánh workshop: namespace đã được cấp sẵn. Nếu `kubectl apply` báo
> `Forbidden ... cannot patch resource "namespaces"` thì bỏ qua — namespace
> đã dùng được.

## Bước 2 — Secret HF_TOKEN

```bash
kubectl create secret generic hf-token \
  --from-literal=token="hf_xxxxxxxxxxxxxxxxxxxx" \
  -n token-factory --dry-run=client -o yaml | kubectl apply -f -
kubectl get secret hf-token -n token-factory
```

**Mong đợi:** secret `hf-token` loại `Opaque`, `DATA` bằng `1`.

Thay `hf_xxxx...` bằng token của bạn. Không commit token vào git;
`01-hf-secret.yaml` chỉ là template tham khảo.

Model và drafter không bị gated. Token chỉ cần cho **dataset**: SPEED-Bench
`throughput_8k` đầy đủ lấy một phần câu hỏi từ dataset gated `cais/hle`, nên
token phải thuộc tài khoản đã được duyệt truy cập `cais/hle`. Nếu không có
token, vẫn tạo secret với `--from-literal=token=""` (manifest bài 01–03 tham
chiếu nó) và chuẩn bị dataset theo nhánh `--skip-gated` ở Bước 5.

## Bước 3A — Storage (workshop)

```bash
kubectl apply -f 02-storage-workshop.yaml
kubectl get pvc -n token-factory
```

**Mong đợi:** cả bốn PVC `bench-datasets`, `bench-results`, `model-cache`,
`vllm-cache` ở trạng thái `Bound`. Sang [Bước 4](#bước-4--xác-minh-trọng-số).

**Nếu không đúng:** PVC `Pending` → `kubectl describe pvc <tên>`; thường là
chưa gắn nhãn node ở Bước 0 nên PV không có node thoả `nodeAffinity`.

## Bước 3B — Storage + tải trọng số (tự học)

Sửa `storageClassName` trong `02-storage-home.yaml` cho khớp cluster
(`kubectl get sc`), rồi:

```bash
kubectl apply -f 02-storage-home.yaml
kubectl get pvc -n token-factory
```

**Mong đợi:** bốn PVC `Bound`. Rồi tải trọng số:

```bash
kubectl apply -f 03-model-download-job.yaml
kubectl wait --for=condition=ready pod -l job-name=model-download -n token-factory --timeout=300s
kubectl logs -f job/model-download -n token-factory
```

Chờ tới dòng `==> Download completed` (Ctrl+C để thoát `logs -f`), rồi dọn job:

```bash
kubectl wait --for=condition=complete job/model-download -n token-factory --timeout=3600s
kubectl delete job model-download -n token-factory
```

Job tải hai repo:

| Repo | Dùng ở bài |
|---|---|
| `unsloth/gemma-4-12b-it-NVFP4` | 01, 02, 03 |
| `deepseek-ai/dspark_gemma4_12b_block7` | 02, 03 |

## Bước 4 — Xác minh trọng số

Chạy ở **cả hai nhánh**:

```bash
kubectl apply -f 06-verify-models-job.yaml
kubectl wait --for=condition=complete job/verify-models -n token-factory --timeout=180s || true
kubectl logs job/verify-models -n token-factory
kubectl delete job verify-models -n token-factory
```

**Mong đợi:**

```
===== [1/2] Target model =====
Mong đợi: /models/unsloth/gemma-4-12b-it-NVFP4
  OK     config.json
  OK     model.safetensors
  OK     processor_config.json
  OK     tokenizer.json
  ...
===== [2/2] DSpark drafter =====
Mong đợi: /models/speculators/deepseek-ai/dspark_gemma4_12b_block7
  OK     config.json
  OK     model.safetensors
  "block_size": 7

==================================================
DAT — trọng số đã sẵn sàng. Tiếp tục sang bài 01.
==================================================
```

**Nếu không đúng:** log báo `KHONG DAT` và liệt kê các thư mục ứng viên dưới
`/models`. Sửa đường dẫn trong manifest cho khớp.

> Đừng chạy `kubectl logs -f` ngay sau `kubectl apply`: container chưa kịp
> khởi động sẽ báo `ContainerCreating`. Luôn `wait` rồi mới `logs`.

## Bước 5 — Dataset benchmark

Chuỗi bài đo bằng [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench),
subset `throughput_8k`.

```bash
kubectl apply -f 07-dataset-prep-job.yaml
kubectl wait --for=condition=complete job/dataset-prep -n token-factory --timeout=3600s || true
kubectl logs job/dataset-prep -n token-factory --tail=20
kubectl delete job dataset-prep -n token-factory
```

Job có thể chạy 10–30 phút, đừng huỷ giữa chừng. Nếu dataset đã có sẵn, job
báo `ĐÃ CÓ SẴN, bỏ qua bước tải` và thoát ngay.

**Mong đợi:** log kết thúc không có lỗi và nhắc tới file `throughput_8k.jsonl`.
Bước 6 sẽ kiểm tra file này.

**Nếu không đúng:**

- Job lặp lỗi `DatasetNotFoundError: Dataset 'cais/hle' is a gated dataset` nghĩa
  là `hf-token` trống hoặc chưa được duyệt `cais/hle`. Xoá job rồi dùng nhánh
  `--skip-gated` ở mục "tải ở máy ngoài" bên dưới (có thể chạy ngay trong một pod
  `python:3.12-slim` mount PVC `bench-datasets`). Kết quả còn 1268/1536 bản ghi
  (bỏ 268 câu HLE); số liệu vẫn so được giữa các bài vì mọi bài dùng cùng file.
- `FSTimeoutError` → chạy lại job, cache trên PVC giữ phần đã tải.

Tải thêm subset khác (tuỳ chọn; ví dụ `throughput_32k`). Job không cho đổi biến môi
trường sau khi tạo, nên sửa giá trị mặc định của `SUBSET` ngay trong YAML:

```bash
kubectl apply -f 07-dataset-prep-job.yaml --dry-run=client -o yaml \
  | sed -e 's/name: dataset-prep/name: dataset-prep-32k/' \
        -e 's/SUBSET:-throughput_8k/SUBSET:-throughput_32k/' \
  | kubectl apply -f -
kubectl wait --for=condition=complete job/dataset-prep-32k -n token-factory --timeout=3600s || true
kubectl logs job/dataset-prep-32k -n token-factory --tail=20
kubectl delete job dataset-prep-32k -n token-factory
```

### Nếu mạng cluster quá chậm: tải ở máy ngoài rồi copy vào PVC

```bash
python3 -m venv .venv && .venv/bin/pip install datasets pandas tiktoken numpy
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py -o prepare.py
.venv/bin/python prepare-speedbench.py --config throughput_8k --output_dir out --prepare prepare.py --skip-gated

kubectl apply -f 08-dataset-uploader-pod.yaml
kubectl wait --for=condition=ready pod/dataset-uploader -n token-factory --timeout=120s
kubectl cp out/throughput_8k.jsonl token-factory/dataset-uploader:/datasets/speed-bench/throughput_8k.jsonl
kubectl exec dataset-uploader -n token-factory -- ls -la /datasets/speed-bench/
kubectl delete pod dataset-uploader -n token-factory
```

**Mong đợi:** `ls` in ra file `throughput_8k.jsonl` có dung lượng lớn hơn 0.

## Bước 6 — Bench client

`bench-client` là pod không GPU, chứa `vllm bench serve`, dataset và PVC kết quả.
Mọi sweep của bài sau chạy qua pod này (hoặc job cùng image).

```bash
kubectl apply -f 04-bench-client.yaml
kubectl wait --for=condition=ready pod -l app=bench-client -n token-factory --timeout=300s
```

Kiểm tra:

```bash
kubectl exec deploy/bench-client -n token-factory -- bash -c '
  python3 -c "import vllm; print(\"vLLM\", vllm.__version__)"
  ls $MODEL_PATH/config.json
  touch /results/.probe && echo "/results ghi được" && rm /results/.probe
  ls $DATASET_DIR/throughput_8k.jsonl
'
```

**Mong đợi:** `vLLM 0.29.0`, đường dẫn `config.json` của model, `/results ghi
được`, và file `throughput_8k.jsonl`. Thiếu dòng nào thì quay lại bước tương ứng
(Bước 4 cho model, Bước 5 cho dataset, Bước 3 cho `/results`).

## Bước 7 — Thư viện offline cho sweep job

Sweep job của bài 01–03 cần `pandas`. Node workshop có thể không ra được PyPI,
nên chép `pandas` từ `bench-client` vào PVC `bench-results` **một lần**:

```bash
kubectl exec -n token-factory deploy/bench-client -- bash -c '
set -e
D=/usr/local/lib/python3.12/dist-packages
mkdir -p /results/pylibs
for p in pandas pytz dateutil six.py tzdata \
         pandas-*.dist-info pytz-*.dist-info \
         python_dateutil-*.dist-info six-*.dist-info tzdata-*.dist-info; do
  for m in $D/$p; do [ -e "$m" ] && cp -r "$m" /results/pylibs/ || true; done
done
ls /results/pylibs
'

kubectl exec -n token-factory deploy/bench-client -- \
  env PYTHONPATH=/results/pylibs python3 -c 'import pandas; print(pandas.__version__)'
```

**Mong đợi:** `ls` liệt kê `pandas`, `pytz`, `dateutil`..., và lệnh cuối in số phiên
bản của pandas.

## Bước 8 — Kiểm tra GPU

```bash
kubectl apply -f 05-gpu-check-job.yaml
kubectl wait --for=condition=complete job/gpu-check -n token-factory --timeout=180s || true
kubectl logs job/gpu-check -n token-factory
kubectl delete job gpu-check -n token-factory
```

**Mong đợi:** một dòng dạng
`NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, <driver>, 12.0`
(tên GPU, dung lượng, driver, compute capability).

- **Workshop (RTX PRO 6000):** compute capability `12.0`, đúng cấu hình các bài được
  viết và đo.
- **Tự thực hành trên GPU khác:** vẫn triển khai được, nhưng đọc ghi chú
  "Phần cứng" ở [README gốc](../README.md#5-môi-trường-workshop): số liệu sẽ khác, và
  có thể phải đổi checkpoint hoặc một vài cờ vLLM.

## Danh sách kiểm tra cuối bài

- [ ] `kubectl get nodes -l fpt-transform-lab=true` ra đúng một node GPU
- [ ] Bốn PVC đều `Bound`
- [ ] `verify-models` in `DAT`
- [ ] `bench-client` `Running` và thấy `throughput_8k.jsonl`
- [ ] `/results/pylibs` có `pandas`
- [ ] `gpu-check` in đúng tên GPU và dung lượng của node bạn dùng

---

## Xử lý sự cố

| Triệu chứng | Kiểm tra |
|---|---|
| PVC ở `Pending` | `kubectl describe pvc <tên>` |
| Pod kẹt `ContainerCreating` lâu | `kubectl describe pod <tên>` → mục `Events`, tìm `FailedAttachVolume` |
| `verify-models` báo `KHONG DAT` | Đọc danh sách thư mục ứng viên trong log, sửa đường dẫn trong manifest |
| Job `dataset-prep` báo `FSTimeoutError` | Chạy lại job — cache trên PVC giữ phần đã tải |
| Job gọi `pip install` treo ở `Running` | Node không ra được PyPI — làm [Bước 7](#bước-7--thư-viện-offline-cho-sweep-job) |
| Nghi hai benchmark chạy chồng nhau | `kubectl get jobs -n token-factory` — chỉ chạy **một** sweep trên mỗi server |

---

**Tiếp theo:** [Bài 01 — Baseline agg](../01-gemma4-baseline-agg/README.md)
