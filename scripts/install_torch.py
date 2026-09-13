"""Instala torch según el hardware disponible (CPU vs GPU NVIDIA).

El wheel de torch depende del hardware: en una máquina sin GPU el wheel
CUDA (~2.5GB) es desperdicio puro; en una con GPU, el wheel CPU deja la
aceleración sin usar. Este script detecta la presencia de `nvidia-smi`
y elige el wheel correcto.

Uso:
    python scripts/install_torch.py            # auto-detección
    python scripts/install_torch.py --gpu      # forzar wheel CUDA
    python scripts/install_torch.py --cpu      # forzar wheel CPU

En Docker no hace falta: el Dockerfile instala torch con los ARGs
TORCH_INDEX_URL / TORCH_PACKAGE (default CPU, GPU opt-in).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

TORCH_VERSION = "2.12.0"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"  # CUDA 12.6 — ajustar si el driver lo requiere


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="forzar wheel CUDA")
    group.add_argument("--cpu", action="store_true", help="forzar wheel CPU")
    args = parser.parse_args()

    if args.cpu:
        index, pkg = CPU_INDEX, f"torch=={TORCH_VERSION}+cpu"
        print(f"CPU forzado -> {pkg} desde {index}")
    elif args.gpu or _has_nvidia_gpu():
        index, pkg = CUDA_INDEX, f"torch=={TORCH_VERSION}+cu126"
        print(f"GPU detectada/forzada -> {pkg} desde {index}")
    else:
        index, pkg = CPU_INDEX, f"torch=={TORCH_VERSION}+cpu"
        print(f"Sin GPU NVIDIA -> {pkg} desde {index}")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        index,
        pkg,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
