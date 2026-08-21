from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..constants import DOTA1_CLASSES, SMALL_VEHICLE
from .open_vocab_hooks import tensor_checksum


SMALL_VEHICLE_ID = DOTA1_CLASSES.index(SMALL_VEHICLE)
LARGE_VEHICLE_ID = DOTA1_CLASSES.index("large-vehicle")


@dataclass
class ClassifierLayer:
    name: str
    module: Any


def classifier_channel_indices(out_channels: int, num_classes: int, class_index: int) -> list[int]:
    if out_channels == num_classes:
        return [class_index]
    if out_channels % num_classes != 0:
        return []
    return [anchor * num_classes + class_index for anchor in range(out_channels // num_classes)]


def find_classifier_layers(model: Any, num_classes: int = len(DOTA1_CLASSES)) -> list[ClassifierLayer]:
    import torch

    candidates = []
    for name, module in model.named_modules():
        lname = name.lower()
        if "bbox_head" not in lname or "cls" not in lname:
            continue
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if classifier_channel_indices(module.out_channels, num_classes, SMALL_VEHICLE_ID):
            candidates.append(ClassifierLayer(name, module))
    return candidates


def classifier_layers_checksum(layers: list[ClassifierLayer]) -> str:
    h = hashlib.sha256()
    for layer in layers:
        h.update(layer.name.encode("utf-8"))
        h.update(tensor_checksum(layer.module.weight).encode("ascii"))
        if layer.module.bias is not None:
            h.update(tensor_checksum(layer.module.bias).encode("ascii"))
    return h.hexdigest()


def snapshot_classifier_layers(layers: list[ClassifierLayer]) -> dict[str, tuple[Any, Any]]:
    snapshots = {}
    for layer in layers:
        weight = layer.module.weight.detach().clone()
        bias = layer.module.bias.detach().clone() if layer.module.bias is not None else None
        snapshots[layer.name] = (weight, bias)
    return snapshots


def restore_classifier_layers(layers: list[ClassifierLayer], snapshots: dict[str, tuple[Any, Any]]) -> None:
    for layer in layers:
        weight, bias = snapshots[layer.name]
        layer.module.weight.data.copy_(weight)
        if layer.module.bias is not None and bias is not None:
            layer.module.bias.data.copy_(bias)


def apply_closedset_classifier_intervention(
    layers: list[ClassifierLayer],
    *,
    intervention: str,
    target_class_index: int = SMALL_VEHICLE_ID,
    control_class_index: int = LARGE_VEHICLE_ID,
    bias_delta: float = 20.0,
) -> dict:
    if not layers:
        raise RuntimeError("no classifier conv layers with DOTA class channels were found")

    import torch

    before = classifier_layers_checksum(layers)
    modified_layers = []
    with torch.no_grad():
        for layer in layers:
            target = classifier_channel_indices(layer.module.out_channels, len(DOTA1_CLASSES), target_class_index)
            control = classifier_channel_indices(layer.module.out_channels, len(DOTA1_CLASSES), control_class_index)
            if not target:
                continue
            if intervention == "suppress_small_vehicle_classifier_channel":
                layer.module.weight[target] = 0
                if layer.module.bias is not None:
                    layer.module.bias[target] -= bias_delta
            elif intervention == "swap_small_large_vehicle_classifier_channel":
                n = min(len(target), len(control))
                if n == 0:
                    continue
                target_idx = target[:n]
                control_idx = control[:n]
                target_weight = layer.module.weight[target_idx].clone()
                target_bias = layer.module.bias[target_idx].clone() if layer.module.bias is not None else None
                layer.module.weight[target_idx] = layer.module.weight[control_idx]
                layer.module.weight[control_idx] = target_weight
                if layer.module.bias is not None and target_bias is not None:
                    layer.module.bias[target_idx] = layer.module.bias[control_idx]
                    layer.module.bias[control_idx] = target_bias
            else:
                raise ValueError(f"unknown closed-set classifier intervention: {intervention}")
            modified_layers.append(layer.name)
    after = classifier_layers_checksum(layers)
    return {
        "intervention_type": intervention,
        "actual_weight_modified": before != after,
        "actual_logit_modified": False,
        "actual_weight_or_logit_modified": before != after,
        "modified_layer_name": ";".join(modified_layers),
        "modified_tensor_name": "classifier_conv.weight,bias",
        "modified_class_index": int(target_class_index),
        "control_class_index": int(control_class_index),
        "target_class_name": DOTA1_CLASSES[target_class_index],
        "control_class_name": DOTA1_CLASSES[control_class_index],
        "classifier_checksum_before": before,
        "classifier_checksum_after": after,
    }
