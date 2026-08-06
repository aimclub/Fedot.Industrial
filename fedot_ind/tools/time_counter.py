"""Device-aware timing helpers."""

import time

import torch


class DeviceTimer:
    """Device-aware wall/device timer.

    * CPU / non-CUDA: ``time.perf_counter``.
    * CUDA: ``torch.cuda.Event`` pair (with optional GEMM warmup), same pattern as
      the transformation benchmark fixtures. There is no shared production timer
      elsewhere; OKHS only has a private ``synchronize`` + ``perf_counter`` helper.
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._use_cuda_events = (
            self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._cpu_start: float | None = None

    @property
    def uses_cuda_events(self) -> bool:
        return self._use_cuda_events

    def warmup(self, n_iters: int = 3, size: int = 256) -> None:
        """Prime CUDA kernels/context before timed regions."""

        if not self._use_cuda_events:
            return
        left = torch.randn(size, size, device=self.device)
        right = torch.randn(size, size, device=self.device)
        for _ in range(n_iters):
            product = left @ right
            product = torch.sin(product)
            _ = product.sum()
        torch.cuda.synchronize(device=self.device)

    def start(self) -> None:
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
            self._cpu_start = None
            return
        self._start_event = None
        self._end_event = None
        self._cpu_start = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed seconds since :meth:`start`."""

        if self._use_cuda_events:
            if self._start_event is None or self._end_event is None:
                raise RuntimeError("DeviceTimer.start() must be called before stop().")
            self._end_event.record()
            self._end_event.synchronize()
            return float(self._start_event.elapsed_time(self._end_event) / 1000.0)
        if self._cpu_start is None:
            raise RuntimeError("DeviceTimer.start() must be called before stop().")
        return float(time.perf_counter() - self._cpu_start)
