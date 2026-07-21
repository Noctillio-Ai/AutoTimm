"""Collect system metrics (CPU, memory, disk, GPU).

Outputs a JSON object to stdout with keys: cpu_cores, mem_total, mem_used,
disk_total, disk_used, gpus, loadavg.

Usage::

    python -m autotimm.flow.system_metrics
    # or
    autotimm-flow system-metrics
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def collect() -> dict:
    """Gather system metrics and return as a dict."""
    res: dict = {}
    if hasattr(os, "cpu_count"):
        res["cpu_cores"] = os.cpu_count()

    # Memory
    try:
        if sys.platform == "darwin":
            mem_total = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
            )
            vm = subprocess.check_output(["vm_stat"]).decode("utf-8")
            pages: dict[str, int] = {}
            for line in vm.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip().rstrip(".")
                    if val.isdigit():
                        pages[key.strip()] = int(val)
            ps = int(subprocess.check_output(["sysctl", "-n", "vm.pagesize"]).strip())
            anonymous = pages.get("Anonymous pages", 0)
            stored = pages.get("Pages stored in compressor", 0)
            wired = pages.get("Pages wired down", 0)
            app_mem = (anonymous - stored) * ps
            wired_mem = wired * ps
            compressed_mem = stored * ps
            mem_used = app_mem + wired_mem + compressed_mem
            res["mem_total"] = mem_total
            res["mem_used"] = max(0, min(mem_used, mem_total))
        elif sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            res["mem_total"] = mem.ullTotalPhys
            res["mem_used"] = mem.ullTotalPhys - mem.ullAvailPhys
        else:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_total = 0
            mem_free = 0
            mem_avail = 0
            for line in lines:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemFree:"):
                    mem_free = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
            if mem_total > 0:
                res["mem_total"] = mem_total
                res["mem_used"] = mem_total - (mem_avail if mem_avail > 0 else mem_free)
    except Exception:
        pass

    # Disk
    try:
        disk_path = "C:\\" if sys.platform == "win32" else "/"
        usage = shutil.disk_usage(disk_path)
        res["disk_total"] = usage.total
        res["disk_used"] = usage.used
    except Exception:
        pass

    # GPU
    try:
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        gpus = []
        for line in gpu_out.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization": float(parts[2]) if parts[2].isdigit() else 0,
                        "mem_total": float(parts[3]),
                        "mem_used": float(parts[4]),
                        "temperature": float(parts[5]) if parts[5].isdigit() else 0,
                    }
                )
        res["gpus"] = gpus
    except Exception:
        res["gpus"] = []

    # CPU Load
    try:
        if hasattr(os, "getloadavg"):
            res["loadavg"] = os.getloadavg()
        else:
            try:
                import psutil

                cpu_pct = psutil.cpu_percent(interval=0.5)
                res["loadavg"] = [cpu_pct / 100.0 * os.cpu_count(), 0, 0]
            except ImportError:
                pass
    except Exception:
        pass

    return res


def main() -> None:
    print(json.dumps(collect()))


if __name__ == "__main__":
    main()
