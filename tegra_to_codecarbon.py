
import os
import time
import json
import threading
import math
import uuid
from datetime import datetime

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    _CC_AVAILABLE = True
except Exception:
    _CC_AVAILABLE = False


class TegraStatsSampler:
    """
    Drop-in replacement for the Jetson-only TegraStats sampler.
    This implementation uses CodeCarbon for total energy and NVML (pynvml) for GPU sampling.

    Parameters
    ----------
    interval_ms : int
        Sampling interval in milliseconds for GPU telemetry and CodeCarbon polling.
    rail : str
        Ignored (kept only for API compatibility).
    save_trace_path : str or None
        If provided, writes a JSONL trace of GPU telemetry samples to this path.
    tracking_mode : str
        CodeCarbon tracking_mode ("process" is recommended to scope energy to this process).
    log_level : str
        CodeCarbon log level ("INFO", "ERROR", etc.).
    """
    def __init__(self, interval_ms=200, rail="orin_sum", save_trace_path=None, tracking_mode="process", log_level="ERROR"):
        self.interval_ms = max(50, int(interval_ms))  # clamp to 50ms minimum
        self.rail = rail
        self.save_trace_path = save_trace_path
        self.tracking_mode = tracking_mode
        self.log_level = log_level

        self._tracker = None
        self._tracker_started = False
        self._start_time = None
        self._end_time = None
        self._gpu_thread = None
        self._stop = threading.Event()
        self._samples = []  # [(t, power_w, util_pct, mem_used_mb)]
        self._trace_fp = None
        self._emissions_kg = float('nan')
        self._total_energy_kwh = float('nan')

        self._nvml_handle = None
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    # Context manager API
    def __enter__(self):
        if _CC_AVAILABLE:
            measure_power_secs = max(0.1, self.interval_ms / 1000.0)
            self._tracker = EmissionsTracker(
                tracking_mode=self.tracking_mode,
                measure_power_secs=measure_power_secs,
                log_level=self.log_level,
            )
            self._tracker.start()
            self._tracker_started = True
        else:
            self._tracker = None

        self._start_time = time.time()

        if self.save_trace_path:
            os.makedirs(os.path.dirname(self.save_trace_path), exist_ok=True)
            self._trace_fp = open(self.save_trace_path, "w", buffering=1)

        if self._nvml_handle is not None:
            self._gpu_thread = threading.Thread(target=self._gpu_sampler_loop, daemon=True)
            self._gpu_thread.start()

        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)

        self._end_time = time.time()

        if self._tracker_started and self._tracker is not None:
            try:
                self._emissions_kg = self._tracker.stop()  # returns kg CO2e
                te = getattr(self._tracker, "_total_energy", None)
                self._total_energy_kwh = te.kwh if te is not None else float('nan')
            except Exception:
                pass

        if self._trace_fp is not None:
            try:
                self._trace_fp.flush()
                self._trace_fp.close()
            except Exception:
                pass

    # Public metrics API
    def duration_s(self):
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.time()
        return max(0.0, end - self._start_time)

    def mean_power(self):
        dur = self.duration_s()
        if self._total_energy_kwh == self._total_energy_kwh and dur > 0:
            total_j = self._total_energy_kwh * 3_600_000.0  # kWh -> J
            return total_j / dur
        if self._samples:
            return sum(s[1] for s in self._samples) / len(self._samples)
        return float('nan')

    def energy_joules(self):
        if self._total_energy_kwh == self._total_energy_kwh:
            return self._total_energy_kwh * 3_600_000.0
        return float('nan')

    def emissions_kgco2(self):
        return self._emissions_kg

    def gpu_energy_joules(self):
        if not self._samples:
            return float('nan')
        dt = self.interval_ms / 1000.0
        return sum(s[1] for s in self._samples) * dt

    def gpu_power_trace(self):
        keys = ("t", "gpu_power_w", "gpu_util_pct", "gpu_mem_used_mb")
        return [dict(zip(keys, s)) for s in self._samples]

    # Internal GPU sampler
    def _gpu_sampler_loop(self):
        interval = self.interval_ms / 1000.0
        while not self._stop.is_set():
            t = time.time()
            try:
                power_w = float("nan")
                util = float("nan")
                mem_mb = float("nan")
                if self._nvml_handle is not None:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                    power_w = p_mw / 1000.0
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    mem_mb = mem.used / 1e6
                self._samples.append((t, power_w, util, mem_mb))

                if self._trace_fp is not None:
                    self._trace_fp.write(json.dumps({
                        "ts": datetime.utcfromtimestamp(t).isoformat(),
                        "gpu_power_w": power_w,
                        "gpu_util_pct": util,
                        "gpu_mem_used_mb": mem_mb
                    }) + "\n")
            except Exception:
                pass
            finally:
                time.sleep(interval)
