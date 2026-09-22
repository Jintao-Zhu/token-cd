# Throughput Report

- 1 worker/GPU baseline: 3 paired cases in 271 s = 0.0111 cases/s
- 2 workers/GPU: 6 paired cases in 302 s = 0.0199 cases/s
- Measured throughput gain: 1.80x
- Peak memory at 2 workers/GPU: GPU1 31.2GB, GPU2 34.3GB, GPU3 31.1GB
- GPU0 was not assigned; its usage stayed at 24 MiB
- Final concurrency: **2 workers per GPU, 6 workers total**

The formal run uses the 250-case queue in `cases/pending/`; precheck cases live in separate directories and are never merged into `PAIRED_RESULTS.csv` or `FINAL_REPORT.md`.
