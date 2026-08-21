"""Auxiliary positive/negative text prompts for FOCUS-EQText margins."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_POSITIVE_PROMPTS = (
    "small vehicle viewed from above",
    "compact vehicle in aerial image",
    "road vehicle from overhead",
    "parking-lot vehicle",
)

DEFAULT_NEGATIVE_PROMPTS = (
    "not a court line",
    "not a roof edge",
    "not a runway marking",
    "not a ship deck",
    "not a storage tank",
    "not a roundabout",
    "not background texture",
    "not padding border",
)


@dataclass(frozen=True)
class FocusNegativeTextBank:
    """Prompt container for auxiliary margin losses only."""

    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]
    auxiliary_only: bool = True
    replaces_class_support: bool = False

    @classmethod
    def default(cls) -> "FocusNegativeTextBank":
        return cls(
            positive_prompts=DEFAULT_POSITIVE_PROMPTS,
            negative_prompts=DEFAULT_NEGATIVE_PROMPTS)

    @property
    def all_prompts(self) -> tuple[str, ...]:
        return self.positive_prompts + self.negative_prompts

    def metadata(self) -> dict[str, object]:
        return {
            "positive_prompts": list(self.positive_prompts),
            "negative_prompts": list(self.negative_prompts),
            "auxiliary_only": self.auxiliary_only,
            "replaces_class_support": self.replaces_class_support,
        }

    def deterministic_embeddings(
            self,
            dim: int = 768,
            device: torch.device | None = None,
            dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        """Build deterministic test embeddings without invoking a text encoder."""
        return {
            "positive": _hash_prompt_embeddings(
                self.positive_prompts, dim, device=device, dtype=dtype),
            "negative": _hash_prompt_embeddings(
                self.negative_prompts, dim, device=device, dtype=dtype),
        }

    def encode_with(
            self,
            tokenizer: Callable[[Sequence[str]], object],
            text_encoder: Callable[[object], Tensor],
            prompts: Iterable[str] | None = None) -> Tensor:
        """Encode prompts with a caller-supplied frozen text encoder."""
        prompt_tuple = tuple(prompts) if prompts is not None else self.all_prompts
        tokens = tokenizer(prompt_tuple)
        with torch.no_grad():
            encoded = text_encoder(tokens)
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]
        return F.normalize(encoded.float(), dim=-1)


def _hash_prompt_embeddings(
        prompts: Sequence[str],
        dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32) -> Tensor:
    rows = []
    for prompt in prompts:
        values = []
        counter = 0
        while len(values) < dim:
            digest = hashlib.sha256(
                f"{prompt}:{counter}".encode("utf-8")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        rows.append(values[:dim])
    return F.normalize(
        torch.tensor(rows, device=device, dtype=dtype),
        dim=-1)
