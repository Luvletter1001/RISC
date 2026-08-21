from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Iterable


RISC_READOUT_TARGETS = (
    'parent_embedding',
    'final_readout_adapter',
    'semantic_readout',
    'bbox_regression',
    'angle_prediction',
    'objectness',
)
RISC_READOUT_CORE_TARGETS = RISC_READOUT_TARGETS[:-1]


def classify_risc_readout_module(module_name: str) -> str | None:
    name = module_name.lower()
    patterns = (
        (r'bbox_head\.rtm_cls\.\d+', 'parent_embedding'),
        (r'bbox_head\.risc_final_readout', 'final_readout_adapter'),
        (r'bbox_head\.rtm_cls_heads\.\d+', 'semantic_readout'),
        (r'bbox_head\.rtm_reg\.\d+', 'bbox_regression'),
        (r'bbox_head\.rtm_ang\.\d+', 'angle_prediction'),
        (r'bbox_head\.rtm_obj\.\d+', 'objectness'),
    )
    for pattern, target in patterns:
        if re.fullmatch(pattern, name):
            return target
    return None


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


def _clone_tensor_tree(value: Any):
    if hasattr(value, 'detach') and hasattr(value, 'shape'):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _clone_tensor_tree(item) for key, item in value.items()
        }
    return value


class RISCReadoutHookRecorder:
    """Capture full OpenRSD semantic and geometry tensors in call order."""

    schema_version = 1

    def __init__(self):
        self.handles = []
        self.plan: list[dict] = []
        self.events: list[dict] = []

    def clear(self):
        self.events = []

    def _hook(self, row: dict):
        def inner(module, inputs, output):
            del module
            self.events.append({
                **row,
                'call_index': len(self.events),
                'inputs': _clone_tensor_tree(inputs),
                'output': _clone_tensor_tree(output),
            })

        return inner

    def register(self, model: Any) -> list[dict]:
        if self.handles:
            raise RuntimeError('RISC readout recorder is already registered')
        self.plan = []
        for module_name, module in model.named_modules():
            target = classify_risc_readout_module(module_name)
            if target is None:
                continue
            row = {
                'module_name': module_name,
                'module_type': module.__class__.__name__,
                'hook_target': target,
            }
            self.plan.append(row)
            self.handles.append(module.register_forward_hook(self._hook(row)))
        return [dict(row) for row in self.plan]

    def snapshot(self) -> dict:
        self.validate_complete()
        planned_targets = {
            row['hook_target'] for row in self.plan
        }
        return {
            'schema_version': self.schema_version,
            'plan': [dict(row) for row in self.plan],
            'structurally_absent_targets': [
                target for target in RISC_READOUT_TARGETS
                if target not in planned_targets
            ],
            'events': _clone_tensor_tree(self.events),
        }

    def validate_complete(self) -> dict[str, int]:
        counts = {target: 0 for target in RISC_READOUT_TARGETS}
        for event in self.events:
            target = event['hook_target']
            if target in counts:
                counts[target] += 1
        planned_targets = {
            row['hook_target'] for row in self.plan
        }
        required_targets = set(RISC_READOUT_CORE_TARGETS)
        if 'objectness' in planned_targets:
            required_targets.add('objectness')
        missing = [
            target for target in RISC_READOUT_TARGETS
            if target in required_targets and counts[target] == 0
        ]
        if missing:
            raise RuntimeError(
                'RISC readout capture is missing: {}'.format(
                    ', '.join(missing)))
        captured_modules = {
            event['module_name'] for event in self.events
        }
        missing_modules = [
            row['module_name'] for row in self.plan
            if row['module_name'] not in captured_modules
        ]
        if missing_modules:
            raise RuntimeError(
                'RISC readout capture is missing modules: {}'.format(
                    ', '.join(missing_modules)))
        return counts

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
