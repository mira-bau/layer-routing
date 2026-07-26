"""Environment inspection without mutating the runtime."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass


REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "pyyaml": "yaml",
    "tokenizers": "tokenizers",
}


@dataclass(frozen=True)
class PackageStatus:
    package: str
    import_name: str
    installed: bool
    version: str | None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_statuses() -> list[PackageStatus]:
    statuses: list[PackageStatus] = []
    for package, import_name in REQUIRED_PACKAGES.items():
        installed = importlib.util.find_spec(import_name) is not None
        statuses.append(PackageStatus(package, import_name, installed, _version(package) if installed else None))
    return statuses


def torch_device_summary() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {
            "torch_available": False,
            "cuda_available": False,
            "mps_available": False,
            "device_policy_ready": False,
            "message": "PyTorch is not installed.",
        }

    import torch  # type: ignore[import-not-found]

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and torch.backends.mps.is_available())
    return {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps_available": mps_available,
        "device_policy_ready": cuda_available or mps_available,
    }


def environment_report() -> dict[str, object]:
    packages = [asdict(status) for status in package_statuses()]
    python_version = sys.version_info
    python_recommended = (3, 10) <= (python_version.major, python_version.minor) <= (3, 12)
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "recommended_for_project": python_recommended,
            "recommendation": "Use Python 3.10-3.12 unless the target PyTorch runtime is known to support newer versions.",
        },
        "packages": packages,
        "devices": torch_device_summary(),
    }


def missing_packages() -> list[str]:
    return [status.package for status in package_statuses() if not status.installed]
