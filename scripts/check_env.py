#!/usr/bin/env python3
"""Report environment readiness without installing anything."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from structformer.utils.env import environment_report, missing_packages  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check environment readiness without installing packages.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args()

    report = environment_report()
    missing = missing_packages()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if missing else 0

    py = report["python"]
    devices = report["devices"]
    print("Environment check")
    print(f"- Python: {py['version'].split()[0]} ({py['executable']})")
    print(f"- Platform: {py['platform']}")
    print(f"- Python recommendation met: {py['recommended_for_project']}")
    if not py["recommended_for_project"]:
        print(f"  Note: {py['recommendation']}")
    print(f"- PyTorch installed: {devices['torch_available']}")
    if devices.get("torch_version"):
        print(f"- PyTorch version: {devices['torch_version']}")
    print(f"- CUDA available: {devices['cuda_available']}")
    if devices.get("cuda_device"):
        print(f"- CUDA device: {devices['cuda_device']}")
    print(f"- MPS available: {devices['mps_available']}")
    print(f"- Training accelerator ready: {devices['device_policy_ready']}")

    print("\nPackages")
    for package in report["packages"]:
        marker = "OK" if package["installed"] else "MISSING"
        version = f" {package['version']}" if package["version"] else ""
        print(f"- {marker}: {package['package']}{version}")

    if missing:
        print("\nMissing packages detected.")
        print("Install only what you need in your active environment. Do not reinstall PyTorch blindly.")
        print("Suggested small-package command:")
        small = [pkg for pkg in missing if pkg != "torch"]
        if small:
            print(f"  pip install {' '.join(small)}")
        if "torch" in missing:
            print("PyTorch is missing. Install the build appropriate for your machine/runtime.")
        return 1

    if not devices["device_policy_ready"]:
        print("\nNo CUDA or MPS accelerator detected. Training scripts should refuse CPU by default.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
