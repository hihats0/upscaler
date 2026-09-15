"""TensorRT motorunu torch tensorleriyle calistiran ince sarmalayici."""
from __future__ import annotations

import tensorrt as trt
import torch

_DTYPES = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


class TrtEngine:
    def __init__(self, path: str, device: str = "cuda") -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        # Varsayilan akista TensorRT ek senkronizasyon yapar (uyari veriyor); ayri akis kullan.
        self.stream = torch.cuda.Stream(device=device)
        self.inputs: list[str] = []
        self.outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _DTYPES[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs[name] = torch.empty(shape, dtype=dtype, device=device)
                self.ctx.set_tensor_address(name, self.outputs[name].data_ptr())

    def input_dtype(self, name: str) -> torch.dtype:
        return _DTYPES[self.engine.get_tensor_dtype(name)]

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        keep = []
        for name in self.inputs:
            t = inputs[name].contiguous()
            keep.append(t)
            self.ctx.set_tensor_address(name, t.data_ptr())
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)       # girdiler hazir olsun
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        current.wait_stream(self.stream)       # ciktilari kullanan islemler motoru beklesin
        return self.outputs
