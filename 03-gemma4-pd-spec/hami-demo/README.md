# Bài 03 · Demo — HAMi-core chia một GPU như thế nào

Hai pod nhỏ cùng xin một lát (slice) ~24 GB trên cùng một card, rồi ta xem
HAMi-core làm gì và không làm gì.

Stack của cụm có hai lớp, cần phân biệt:

| Lớp | Thành phần | Việc làm |
|---|---|---|
| **Quyết định** | KAI Scheduler (`binder`, plugin `hamicore`) | Chọn card, đổi `gpu-memory` thành con số, ghi `CUDA_DEVICE_MEMORY_LIMIT` vào pod |
| **Thực thi** | `kai-resource-isolator` → HAMi-core (`libvgpu.so`) | `LD_PRELOAD` vào container, chặn `cuMemAlloc` vượt giới hạn, báo `nvidia-smi` theo slice |

KAI chỉ **ghi** con số. Nếu không có HAMi-core thì không có gì chặn process
cấp phát vượt giới hạn đó.

## Trước khi bắt đầu

- Đã làm [bài 03 Bước 1](../README.md#bước-1--kiểm-tra-kai-scheduler--hami-core):
  KAI và isolator `Running`, đã có queue `fpt-transform-lab`.
- GPU của node lab đang rảnh (đã xoá deployment bài 02).

```bash
cd 03-gemma4-pd-spec/hami-demo
kubectl apply -f demo-pods.yaml
kubectl wait -n token-factory pod -l app=hami-demo --for=condition=Ready --timeout=3m
kubectl get pod -n token-factory -l app=hami-demo -o wide
```

Để ý `demo-pods.yaml`: **không** xin `nvidia.com/gpu`. Pod chỉ có
`schedulerName: kai-scheduler`, nhãn `kai.scheduler/queue` và annotation
`gpu-memory: "24000"`.

## 1. Mỗi pod chỉ thấy lát của mình

```bash
for p in hami-demo-a hami-demo-b; do
  echo "== $p"
  kubectl exec -n token-factory $p -c cuda -- sh -c \
    'cat /etc/ld.so.preload; echo LIMIT=$CUDA_DEVICE_MEMORY_LIMIT; \
     nvidia-smi --query-gpu=uuid,memory.total,memory.used --format=csv,noheader'
done
```

Mong đợi, với mỗi pod:

- `/usr/local/vgpu/libvgpu.so`: HAMi-core đã được nạp sẵn.
- `LIMIT` khác rỗng.
- `memory.total` khoảng **24 GB**, không phải ~96 GB của cả card.
- **Cả hai pod cùng một `GPU-<UUID>`**: hai lát nằm trên cùng một card.

Con số thật lớn hơn 24000 một chút vì KAI làm tròn **lên** phần card đến 2 chữ
số thập phân: 24000 / 97 887 MiB = 0,245, làm tròn thành **0,25**, tức 24 471 MiB.
Xem phần đã đặt chỗ:

```bash
CM=$(kubectl get pod hami-demo-a -n token-factory \
  -o jsonpath='{range .spec.volumes[*]}{.configMap.name}{"\n"}{end}' | grep shared-gpu)
kubectl get cm $CM -n token-factory -o yaml | grep -E 'GPU_PORTION|CUDA_DEVICE_MEMORY_LIMIT'
```

Kết quả đã đo trên `rtx6000`:

```text
/usr/local/vgpu/libvgpu.so
LIMIT=24471m
GPU-99e2f280-922e-94ea-4ae3-0b0a3a163b61, 24471 MiB, 0 MiB     # cả hai pod, cùng UUID
  CUDA_DEVICE_MEMORY_LIMIT: 24471m
  GPU_PORTION: "0.25"
```

Các dòng `[HAMI-core Msg ...] Cleanup on exit` in ra sau mỗi lệnh là bình thường:
đó là HAMi-core tự dọn khi process thoát, và chứng tỏ thư viện đang được nạp.

## 2. Giới hạn bộ nhớ là thật

Cấp phát từng khối 1 GiB cho tới khi CUDA từ chối:

```bash
kubectl exec -n token-factory hami-demo-a -c cuda -- python3 /demo/alloc.py 2>&1 | grep -v HAMI-core | tail -2
```

Đã đo: `OOM after 23 GiB ... GPU 0 has a total capacity of 23.90 GiB`, dù card
còn trống hơn 70 GB. Log có dòng `HAMI-core ERROR ... Device 0 OOM`: chính
HAMi-core trả lỗi, lời gọi chưa tới driver.

## 3. Không có HAMi-core thì giới hạn chỉ là con số

`demo-pod-noiso.yaml` xin đúng slice 24000 như pod A, nhưng gắn nhãn
`kai-resource-isolator.io/webhook: ignore` nên webhook của isolator bỏ qua nó.
Pod trông y hệt khi webhook isolator bị sập: webhook đặt `failurePolicy: Ignore`,
nên pod vẫn chạy bình thường, chỉ là không có giới hạn.

```bash
kubectl apply -f demo-pod-noiso.yaml
kubectl wait -n token-factory pod/hami-demo-noiso --for=condition=Ready --timeout=3m
kubectl exec -n token-factory hami-demo-noiso -c cuda -- sh -c \
  'cat /etc/ld.so.preload; echo LIMIT=$CUDA_DEVICE_MEMORY_LIMIT; \
   nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader'
kubectl exec -n token-factory hami-demo-noiso -c cuda -- python3 /demo/alloc.py 40 | tail -1
kubectl delete -f demo-pod-noiso.yaml
```

Đã đo: không có `/etc/ld.so.preload`, `LIMIT=24471m` vẫn có, nhưng `nvidia-smi`
báo **97887 MiB**, và `reached 40 GiB without OOM`. KAI vẫn ghi số và vẫn tính
0,25 card vào queue, mà pod lấn sang phần của hàng xóm thoải mái. Người bị OOM
sẽ là **hàng xóm**, không phải pod này.

Bài học vận hành: kiểm tra giới hạn **bên trong từng container GPU**
(`/etc/ld.so.preload`, `nvidia-smi`), không tin trạng thái `Running`.

> `CUDA_DISABLE_CONTROL=true` không tắt được HAMi-core trong bản isolator 1.1.0
> (đã thử: vẫn OOM ở 23 GiB, `nvidia-smi` vẫn báo slice). Muốn chạy không giới
> hạn thì phải dùng nhãn ignore như trên.

## 4. HAMi-core không chia SM

HAMi-core chỉ giới hạn **bộ nhớ**. Compute (SM) thì các pod chia nhau tự do.
Đo TFLOPS khi chạy một mình:

```bash
kubectl exec -n token-factory hami-demo-a -c cuda -- python3 /demo/matmul.py 20
```

Rồi chạy cả hai pod cùng lúc:

```bash
kubectl exec -n token-factory hami-demo-a -c cuda -- python3 /demo/matmul.py 30 | sed 's/^/[A] /' &
kubectl exec -n token-factory hami-demo-b -c cuda -- python3 /demo/matmul.py 30 | sed 's/^/[B] /'
wait
```

Đã đo (BF16 matmul 8192³, RTX PRO 6000):

| | TFLOPS |
|---|---|
| A một mình | 411 |
| A khi có B | 192 |
| B khi có A | 192 |

Mỗi pod chỉ còn một nửa, không pod nào lỗi, chỉ chậm đi.

Ở PD bài 03 cũng vậy: lúc P prefill nặng, D decode chậm lại.

## 5. Card không chứa quá 100%

Xin thêm một lát ~60 GB (0,62 card) trong khi A + B đã giữ 0,50:

```bash
kubectl apply -f demo-pod-big.yaml
kubectl get pod hami-demo-big -n token-factory
kubectl get events -n token-factory --field-selector involvedObject.name=hami-demo-big | tail -3
```

Đã đo: `Pending`, sự kiện `Unschedulable: ... didn't have enough resources: GPU
memory`. Tăng quota của queue cũng không cứu được: card vật lý chỉ có 1,00.

## 6. Phía hạ tầng

```bash
# KAI giữ nguyên card qua device plugin bằng một pod "reservation", rồi tự chia bên dưới
kubectl -n kai-resource-reservation get pods -o wide

# Bộ nhớ thật từng container giữ trên card (monitor của isolator, cổng 9394)
MON=$(kubectl -n kai-resource-isolator get pod -o name | grep monitor | head -1)
kubectl -n kai-resource-isolator port-forward $MON 9394:9394 &
curl -s localhost:9394/metrics | grep hami_container_device_memory_bytes | grep hami-demo
```

## Dọn dẹp (bắt buộc trước Bước 2 của bài 03)

```bash
kubectl delete -f demo-pod-big.yaml -f demo-pod-noiso.yaml --ignore-not-found
kubectl delete -f demo-pods.yaml
```

PD ở bài 03 cần 0,47 + 0,47 card. Nếu còn pod demo giữ 0,50 thì PD sẽ `Pending`.

## Tóm tắt

| HAMi-core làm | HAMi-core không làm |
|---|---|
| Chặn cấp phát vượt slice (OOM tại slice) | Chia SM / compute |
| `nvidia-smi`, `mem_get_info` báo theo slice, nên `--gpu-memory-utilization` của vLLM tính trên **slice** | Bảo vệ pod mà webhook isolator bỏ qua (nhãn ignore, webhook sập) |
| | Đếm đúng bộ nhớ khi chạy cùng NIXL/UCX: bộ đếm tràn, nên D ở bài 03 phải chạy không có HAMi-core |
| Áp cho mọi process trong container | Inject vào container khác container index 0 |

Đọc thêm: `llm-ops/docs/gpu-fraction/hami-core-deep-dive.md` và
`kai-scheduler-deep-dive.md`.

---

**Quay lại:** [Bài 03 — Bước 2](../README.md#bước-2--deploy)
