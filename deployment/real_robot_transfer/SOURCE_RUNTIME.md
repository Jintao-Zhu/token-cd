# Source runtime snapshot and migration notes

Captured from `/home/leju-suzhou/zjt_ws/openvla-ar-h100` on 2026-10-09.

| Component | Source version / note |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| PyTorch CUDA build | 12.8 |
| Transformers | 4.40.1 |
| Tokenizers | 0.19.1 |
| timm | 0.9.10 |
| NumPy | 1.24.4 |
| SAPIEN | 2.2.2; used by SIMPLER replay/rendering, not required for a real-camera deployment |
| NVIDIA driver | 570.133.20 on source host; destination needs a driver compatible with its installed PyTorch/CUDA build |

The source environment directory is about 7.6 GB. Do not copy it as a portable environment: CUDA extensions and robot drivers are host-specific. Rebuild a clean environment on the target machine and install the model/runtime dependencies there. Pin the final tested package versions in the target machine's own lock file.

Simulation-specific pieces that can usually be omitted from a physical inference host include SAPIEN, SIMPLER assets, RT shaders, Vulkan/EGL configuration, snapshots, and rollout artifacts. Keep the OpenVLA model code, tokenizer/processor, model weights, action normalization statistics, and the VTCG implementation.

This branch now contains a simulator-free inference runtime and a generic adapter
protocol. The robot-specific driver, calibration, action mapping, and safety
implementation must still be supplied for the actual arm/camera.
