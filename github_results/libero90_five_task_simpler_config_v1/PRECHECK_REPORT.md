# Formal Precheck Report

## Resource envelope

- CPU cores: 256
- RAM: 1.5 TiB total, 1.4 TiB available
- Disk: `/dev/sda2`, 44 TiB total, 27 TiB available, 37% used
- `/dev/shm`: 756 GiB
- GPU: 8 x H100 80GB; formal workers restricted to physical GPUs 1, 2, 3
- GPU0 was not assigned to any formal worker; its observed usage remained 24 MiB and did not increase during prechecks

## Renderer/device mapping

Direct `CUDA_VISIBLE_DEVICES=N` with `MUJOCO_EGL_DEVICE_ID=N` worked on GPU1 but failed framebuffer creation on GPU2/3. The validated mapping is:

| worker model GPU | CUDA_VISIBLE_DEVICES | EGL device | model device |
|---:|---|---:|---|
| 1 | `1` | 1 | `cuda:0` |
| 2 | `1,2` | 1 | `cuda:1` |
| 3 | `1,3` | 1 | `cuda:1` |

This keeps model inference on GPUs 1/2/3 and uses GPU1 for EGL for the GPU2/3 workers; it does not use GPU0. The worker records `physical_gpu` and `render_gpu`.

## Single-worker precheck

3 fresh paired cases, one worker on each of GPU1/2/3. These results are not merged into the formal evaluation.

| Case | Vanilla steps | Matched steps | Arm runtime sum (s) | Vanilla success | Matched success |
|---|---:|---:|---:|---|---|
| task03__init001 | 343 | 179 | 138.6 | True | True |
| task49__init001 | 400 | 400 | 244.9 | False | False |
| task72__init001 | 216 | 400 | 209.8 | True | False |

- Wall interval: `2026-09-22T19:00:46+08:00` to `2026-09-22T19:05:17+08:00`
- 3 paired cases / 271 s = **0.0111 paired cases/s**
- Peak GPU memory: GPU1 `15.6GB`, GPU2 `17.4GB`, GPU3 `15.6GB`
- No OOM, no NaN/Inf, no runner exceptions

## Two-workers-per-GPU precheck

6 fresh paired cases, two workers per physical GPU. These results are not merged into the formal evaluation.

| Case | Vanilla steps | Matched steps | Arm runtime sum (s) | Vanilla success | Matched success |
|---|---:|---:|---:|---|---|
| task03__init002 | 400 | 400 | 263.5 | False | False |
| task03__init003 | 180 | 173 | 133.0 | True | True |
| task10__init002 | 107 | 113 | 83.0 | True | True |
| task49__init002 | 192 | 141 | 118.2 | True | True |
| task72__init002 | 400 | 147 | 154.9 | False | True |
| task73__init002 | 400 | 160 | 162.5 | False | True |

- Wall interval: `2026-09-22T19:05:44+08:00` to `2026-09-22T19:10:46+08:00`
- 6 paired cases / 302 s = **0.0199 paired cases/s**
- Throughput ratio versus one-worker precheck: **1.80x**
- Peak GPU memory: GPU1 `31.2GB`, GPU2 `34.3GB`, GPU3 `31.1GB`
- Maximum observed GPU utilization: GPU1/2/3 approximately 99-100%
- No OOM, no paging, no renderer error in completed cases

## Frozen concurrency decision

Use **2 workers per physical GPU**, 6 persistent workers total. Each worker loads the model once and drains the shared file queue for the formal 250 paired cases.
