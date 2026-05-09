"""Async inference server using ONNX Runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class InferenceServer:
    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]

    def warmup(self, shape: tuple[int, ...], n: int = 5) -> None:
        for _ in range(n):
            self(np.zeros(shape, dtype=np.float32))
