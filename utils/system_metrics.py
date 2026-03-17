import os

import psutil
import torch


def _bytes_to_gb(num_bytes):
    return num_bytes / (1024 ** 3)


def collect_memory_metrics(device_type, device):
    metrics = {}

    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    metrics["process/rss_gb"] = _bytes_to_gb(process_memory.rss)
    metrics["system/ram_used_gb"] = _bytes_to_gb(virtual_memory.used)
    metrics["system/ram_available_gb"] = _bytes_to_gb(virtual_memory.available)
    metrics["system/ram_percent"] = float(virtual_memory.percent)

    device_module = getattr(torch, device_type, None)
    if device_module is None:
        return metrics

    try:
        if hasattr(device_module, "is_available") and not device_module.is_available():
            return metrics
    except Exception:
        return metrics

    memory_fns = {
        "gpu/allocated_gb": "memory_allocated",
        "gpu/reserved_gb": "memory_reserved",
        "gpu/max_allocated_gb": "max_memory_allocated",
        "gpu/max_reserved_gb": "max_memory_reserved",
    }

    for metric_name, fn_name in memory_fns.items():
        fn = getattr(device_module, fn_name, None)
        if not callable(fn):
            continue
        try:
            metrics[metric_name] = _bytes_to_gb(fn(device))
        except TypeError:
            metrics[metric_name] = _bytes_to_gb(fn())
        except Exception:
            continue

    return metrics


def reset_peak_memory_stats(device_type, device):
    device_module = getattr(torch, device_type, None)
    if device_module is None:
        return

    fn = getattr(device_module, "reset_peak_memory_stats", None)
    if not callable(fn):
        return

    try:
        fn(device)
    except TypeError:
        fn()
    except Exception:
        pass


def prefix_metrics(metrics, prefix):
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def format_memory_metrics(metrics):
    parts = []

    if "process/rss_gb" in metrics:
        parts.append(f"proc_rss={metrics['process/rss_gb']:.2f}GB")
    if "system/ram_used_gb" in metrics and "system/ram_percent" in metrics:
        parts.append(f"ram_used={metrics['system/ram_used_gb']:.2f}GB ({metrics['system/ram_percent']:.1f}%)")
    if "gpu/allocated_gb" in metrics:
        parts.append(f"gpu_alloc={metrics['gpu/allocated_gb']:.2f}GB")
    if "gpu/reserved_gb" in metrics:
        parts.append(f"gpu_reserved={metrics['gpu/reserved_gb']:.2f}GB")
    if "gpu/max_allocated_gb" in metrics:
        parts.append(f"gpu_peak_alloc={metrics['gpu/max_allocated_gb']:.2f}GB")
    if "gpu/max_reserved_gb" in metrics:
        parts.append(f"gpu_peak_reserved={metrics['gpu/max_reserved_gb']:.2f}GB")

    return " | ".join(parts)
