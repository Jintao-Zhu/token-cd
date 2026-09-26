#!/usr/bin/env python3
"""Resolve an authorized NVML physical GPU index to MuJoCo's EGL ordinal.

MuJoCo's MUJOCO_EGL_DEVICE_ID indexes eglQueryDevicesEXT(), whose ordering is
not guaranteed to match nvidia-smi/CUDA device numbering. This resolver uses
the EGL DRM render node's sysfs PCI address and fails closed if it cannot
prove the mapping.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ALLOWED_PHYSICAL_GPUS = {1, 2, 3, 6, 7}
EGL_EXTENSIONS = 0x3055
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377


def normalize_bdf(value: str) -> str:
    match = re.search(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if not match:
        raise ValueError(f"cannot parse PCI bus address: {value!r}")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{function}"


def nvml_index_to_bdf() -> dict[int, str]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True,
    )
    result = {}
    for line in raw.splitlines():
        index, bdf = (part.strip() for part in line.split(",", 1))
        result[int(index)] = normalize_bdf(bdf)
    return result


def egl_device_map() -> list[dict[str, object]]:
    # This only calls EGL device enumeration/string queries. It does not create
    # an EGLDisplay, GL context, framebuffer, or submit rendering work.
    from mujoco.egl import egl_ext as egl
    from OpenGL import EGL
    import ctypes

    query_device_string = ctypes.CFUNCTYPE(
        ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint
    )(EGL.eglGetProcAddress("eglQueryDeviceStringEXT"))
    devices = egl.eglQueryDevicesEXT()
    mapped = []
    for ordinal, device in enumerate(devices):
        ext_blob = query_device_string(device, EGL_EXTENSIONS) or b""
        extensions = ext_blob.decode("ascii", errors="replace").split()
        row: dict[str, object] = {"egl_ordinal": ordinal, "extensions": extensions}
        if "EGL_EXT_device_drm_render_node" not in extensions:
            row["pci_bdf"] = None
        else:
            node_blob = query_device_string(device, EGL_DRM_RENDER_NODE_FILE_EXT)
            if not node_blob:
                row["pci_bdf"] = None
            else:
                render_node = node_blob.decode("ascii", errors="replace")
                row["render_node"] = render_node
                sysfs_device = Path("/sys/class/drm") / Path(render_node).name / "device"
                # Resolve to the DRM device's terminal PCI directory. Parsing
                # the entire sysfs path could accidentally match an upstream
                # PCI bridge address before the GPU's own BDF.
                row["pci_bdf"] = normalize_bdf(sysfs_device.resolve().name)
        mapped.append(row)
    return mapped


def resolve(physical_gpu: int) -> dict[str, object]:
    if physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(
            f"physical GPU {physical_gpu} is not in the authorized set "
            f"{sorted(ALLOWED_PHYSICAL_GPUS)}"
        )
    nvml = nvml_index_to_bdf()
    if physical_gpu not in nvml:
        raise RuntimeError(f"physical GPU {physical_gpu} is absent from nvidia-smi")
    target_bdf = nvml[physical_gpu]
    matches = [d for d in egl_device_map() if d.get("pci_bdf") == target_bdf]
    if len(matches) != 1:
        raise RuntimeError(
            f"fail-closed: expected exactly one EGL device for physical GPU "
            f"{physical_gpu} PCI {target_bdf}, found {matches}"
        )
    return {
        "physical_gpu": physical_gpu,
        "pci_bdf": target_bdf,
        "egl_ordinal": int(matches[0]["egl_ordinal"]),
        "render_node": matches[0].get("render_node"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("physical_gpu", type=int)
    parser.add_argument("--print-index", action="store_true")
    args = parser.parse_args()
    result = resolve(args.physical_gpu)
    if args.print_index:
        print(result["egl_ordinal"])
    else:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
