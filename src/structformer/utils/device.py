"""Device selection helpers.

Training should not silently fall back to CPU. CPU is useful for tiny tests, but
full runs should use CUDA or MPS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


DeviceName = Literal["auto", "cuda", "mps", "cpu"]


class DeviceError(RuntimeError):
    """Raised when the requested device policy cannot be satisfied."""


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    type: str
    is_accelerated: bool
    torch_version: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _import_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise DeviceError(
            "PyTorch is not installed. Install a runtime-appropriate PyTorch "
            "build before running training."
        ) from exc
    return torch


def select_device(requested: DeviceName = "auto", *, allow_cpu: bool = False) -> DeviceInfo:
    """Select a device using the project policy.

    Policy:
    - `auto`: prefer CUDA, then MPS, then fail unless CPU is explicitly allowed.
    - explicit `cpu`: allowed only with `allow_cpu=True`.
    """

    torch = _import_torch()
    torch_version = getattr(torch, "__version__", None)

    if requested == "cuda":
        if torch.cuda.is_available():
            return DeviceInfo("cuda", "cuda", True, torch_version, torch.cuda.get_device_name(0))
        raise DeviceError("CUDA was requested, but torch.cuda.is_available() is false.")

    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceInfo("mps", "mps", True, torch_version, "Apple Metal Performance Shaders")
        raise DeviceError("MPS was requested, but torch.backends.mps.is_available() is false.")

    if requested == "cpu":
        if not allow_cpu:
            raise DeviceError("CPU was requested, but CPU use requires allow_cpu=True.")
        return DeviceInfo("cpu", "cpu", False, torch_version, "CPU allowed explicitly")

    if requested != "auto":
        raise DeviceError(f"Unknown device request: {requested}")

    if torch.cuda.is_available():
        return DeviceInfo("cuda", "cuda", True, torch_version, torch.cuda.get_device_name(0))

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "mps", True, torch_version, "Apple Metal Performance Shaders")

    if allow_cpu:
        return DeviceInfo("cpu", "cpu", False, torch_version, "CPU allowed explicitly")

    raise DeviceError(
        "No CUDA or MPS accelerator is available. Refusing to run on CPU by "
        "default. Pass an explicit allow-CPU option only for tests or tiny smoke runs."
    )


def memory_gb(device_type: str) -> float | None:
    """Return current accelerator memory in GB when available."""

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None

    if device_type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.memory_allocated() / (1024**3))
    return None

