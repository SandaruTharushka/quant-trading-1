"""Model export — ONNX with optional TensorRT optimization."""
from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, out_path: str | Path,
                opset: int = 17, dynamic_axes: dict | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes or {"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    return out_path
