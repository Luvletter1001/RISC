from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable


def classify_openrsd_module(module_name: str, module_type: str = "") -> str | None:
    name = module_name.lower()
    mtype = module_type.lower()
    if re.fullmatch(r"bbox_head\.rtm_cls_heads\.\d+", name):
        return "dense_logits"
    if re.fullmatch(r"bbox_head\.rtm_cls\.\d+", name):
        return "class_embedding_projection"
    if re.fullmatch(r"bbox_head\.cls_convs\.\d+\.\d+", name):
        return "dense_feature"
    if name == "visual_support_mapping" or name.endswith(".visual_support_mapping"):
        return "visual_support_embedding"
    if name == "text_support_mapping" or name.endswith(".text_support_mapping"):
        return "text_embedding"
    if (
        name == "aux_bbox_head"
        or name.startswith("aux_bbox_head.rtm_cls_heads.")
        or "obj_align" in name
        or "align" in name
        or "score" in name
        or "alignment" in mtype
    ):
        return "alignment_score"
    if name == "bbox_head":
        return "pre_nms_head"
    return None


def select_openrsd_hook_modules(named_modules: Iterable[tuple[str, Any]]) -> list[dict]:
    rows = []
    for name, module in named_modules:
        if not name:
            continue
        hook_target = classify_openrsd_module(name, module.__class__.__name__)
        if hook_target is None:
            continue
        rows.append(
            {
                "module_name": name,
                "module_type": module.__class__.__name__,
                "hook_target": hook_target,
                "status": "PLANNED",
            }
        )
    priority = {
        "dense_logits": 0,
        "visual_support_embedding": 1,
        "text_embedding": 2,
        "pre_nms_head": 3,
        "class_embedding_projection": 4,
        "dense_feature": 5,
        "alignment_score": 6,
    }
    return sorted(rows, key=lambda row: (priority.get(row["hook_target"], 99), row["module_name"]))


def _iter_tensors(value: Any, prefix: str = "out"):
    if hasattr(value, "detach") and hasattr(value, "shape"):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}_{idx}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}_{key}")


def _tensor_scalar(value: Any, attr: str):
    tensor = value.detach().float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if attr == "std":
        out = tensor.std(unbiased=False)
    else:
        out = getattr(tensor, attr)()
    return float(out.item() if hasattr(out, "item") else out)


def _tensor_stats(tensor: Any) -> dict:
    shape = list(getattr(tensor, "shape", []))
    dtype = str(getattr(tensor, "dtype", ""))
    device = str(getattr(tensor, "device", ""))
    numel = int(tensor.numel()) if hasattr(tensor, "numel") else math.prod(shape or [0])
    row = {
        "tensor_shape_example": shape,
        "dtype": dtype,
        "device": device,
        "numel": numel,
    }
    if numel:
        row.update(
            {
                "mean": _tensor_scalar(tensor, "mean"),
                "std": _tensor_scalar(tensor, "std"),
                "min": _tensor_scalar(tensor, "min"),
                "max": _tensor_scalar(tensor, "max"),
            }
        )
    return row


class OpenRSDHookRecorder:
    def __init__(self, max_spatial_size: int = 32):
        self.max_spatial_size = max_spatial_size
        self.handles = []
        self.plan: list[dict] = []
        self.records: OrderedDict[str, dict] = OrderedDict()

    def clear(self):
        self.records = OrderedDict()

    def _record(self, row: dict, value: Any):
        captured = []
        for tensor_path, tensor in _iter_tensors(value):
            captured.append({"tensor_path": tensor_path, **_tensor_stats(tensor)})
        status = "DONE_SMOKE" if captured else "NOT_RUN"
        self.records[row["module_name"]] = {
            **row,
            "status": status,
            "captured_tensors": captured,
            "tensor_shape_example": captured[0]["tensor_shape_example"] if captured else [],
            "dtype": captured[0]["dtype"] if captured else "",
            "device": captured[0]["device"] if captured else "",
        }

    def _hook(self, row: dict):
        def inner(module, inputs, output):
            self._record(row, output)

        return inner

    def register(self, model: Any) -> list[dict]:
        named = dict(model.named_modules())
        self.plan = select_openrsd_hook_modules(named.items())
        for row in self.plan:
            module = named[row["module_name"]]
            self.handles.append(module.register_forward_hook(self._hook(row)))
        return self.plan

    def to_rows(self) -> list[dict]:
        rows = []
        for row in self.plan:
            record = self.records.get(row["module_name"])
            if record is None:
                rows.append({**row, "status": "NOT_RUN", "tensor_shape_example": [], "dtype": "", "device": ""})
            else:
                rows.append(record)
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
