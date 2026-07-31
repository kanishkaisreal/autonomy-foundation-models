from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

import torch

from common.config import load_project_config, validate_project_config


class DeviceResolutionError(RuntimeError):
    """Raised when the requested compute device cannot be used."""


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    """Resolved PyTorch device and the capabilities detected at runtime."""

    requested: str
    resolved: str
    torch_device: torch.device
    device_name: str
    cuda_available: bool
    mps_available: bool
    mps_built: bool


def resolve_compute_device(requested_device: str) -> ComputeDevice:
    """
    Resolve an explicitly requested CPU, MPS, CUDA, or automatic device.

    Resolution order for ``auto``:

        CUDA -> MPS -> CPU

    Explicit device requests fail rather than silently falling back.
    """

    requested = requested_device.strip().lower()
    allowed_devices = {"auto", "cpu", "mps", "cuda"}

    if requested not in allowed_devices:
        raise DeviceResolutionError(
            f"Unsupported device {requested_device!r}. "
            f"Expected one of {sorted(allowed_devices)}."
        )

    cuda_available = torch.cuda.is_available()

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            resolved = "cuda"
        elif mps_available:
            resolved = "mps"
        else:
            resolved = "cpu"

    elif requested == "cuda":
        if not cuda_available:
            raise DeviceResolutionError(
                "CUDA was requested, but CUDA is not available in the "
                "current PyTorch environment."
            )

        resolved = "cuda"

    elif requested == "mps":
        if not mps_built:
            raise DeviceResolutionError(
                "MPS was requested, but the installed PyTorch build "
                "does not include MPS support."
            )

        if not mps_available:
            raise DeviceResolutionError(
                "MPS was requested, but it is not available on this machine."
            )

        resolved = "mps"

    else:
        resolved = "cpu"

    torch_device = torch.device(resolved)

    if resolved == "cuda":
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    elif resolved == "mps":
        get_mps_name = getattr(mps_backend, "get_name", None)

        if callable(get_mps_name):
            try:
                device_name = str(get_mps_name())
            except RuntimeError:
                device_name = "Apple MPS"
        else:
            device_name = "Apple MPS"

    else:
        device_name = platform.processor() or "CPU"

    return ComputeDevice(
        requested=requested,
        resolved=resolved,
        torch_device=torch_device,
        device_name=device_name,
        cuda_available=cuda_available,
        mps_available=mps_available,
        mps_built=mps_built,
    )


def main() -> None:
    """Resolve the configured compute device using F5."""
    repository_root = Path(__file__).resolve().parents[2]
    config_path = repository_root / "configs" / "base.yaml"

    config = load_project_config(config_path)
    validate_project_config(config, project_root=repository_root)
    device = resolve_compute_device(config.runtime.device)
    print(
        f"Device: requested={device.requested}, "
        f"resolved={device.resolved}, "
        f"name={device.device_name}"
    )


if __name__ == "__main__":
    main()
