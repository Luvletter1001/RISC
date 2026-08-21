#!/usr/bin/env python3
"""Unified encoder adapters for text/image embedding swap experiments."""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

ImageInput = Union[Sequence[Path], Sequence[Any]]


@dataclass
class EncoderAdapter:
    name: str
    root: Path
    modality: str = 'unknown'  # text | image | text_image
    native_dim: int = 0
    output_dim: int = 0
    load_status: str = 'UNKNOWN'
    error: str = ''
    required_package: str = ''
    checkpoint: Optional[Path] = None
    arch: str = ''
    _model: Any = field(default=None, repr=False)
    _preprocess: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def load(self, device: str = 'cpu') -> 'EncoderAdapter':
        try:
            if self.name in ('clip', 'open_clip', 'remoteclip', 'georsclip'):
                self._load_open_clip_family(device)
            elif self.name == 'dino':
                self._load_dino(device)
            elif self.name == 'dinov2':
                self._load_dinov2(device)
            elif self.name == 'lae-dino':
                self._load_lae_dino(device)
            elif self.name == 'original':
                self.load_status = 'LOAD_OK'
                self.modality = 'text_image'
            else:
                self.load_status = 'UNKNOWN'
                self.error = f'unknown encoder {self.name}'
        except Exception as exc:
            self.load_status = 'LOAD_FAILED'
            self.error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}'
        return self

    def _load_open_clip_family(self, device: str):
        import open_clip
        import torch

        self.required_package = 'open_clip'
        cfg = {
            'clip': ('ViT-B-32', self.root / 'ViT-B-32.pt'),
            'open_clip': ('ViT-B-32', self.root / 'OpenCLIP-ViT-B-32-laion2b.bin'),
            'remoteclip': ('ViT-L-14', self.root / 'RemoteCLIP-ViT-L-14.pt'),
            'georsclip': ('ViT-L-14', self.root / 'ckpt' / 'RS5M_ViT-L-14.pt'),
        }
        arch, ckpt = cfg[self.name]
        if not ckpt.exists():
            alts = list(self.root.rglob('*.pt')) + list(self.root.rglob('*.bin'))
            ckpt = alts[0] if alts else ckpt
        self.arch = arch
        self.checkpoint = ckpt
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=str(ckpt))
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            tok = self._tokenizer(['small vehicle'])
            te = self._model.encode_text(tok.to(device))
            self.native_dim = int(te.shape[-1])
        self.modality = 'text_image'
        self.load_status = 'TEXT_IMAGE'

    def _load_dino(self, device: str):
        import torch
        import timm

        self.required_package = 'timm'
        ckpt = self.root / 'dino_vitbase16_pretrain.pth'
        self.checkpoint = ckpt
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        sd = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        self._model = model.to(device).eval()
        self._preprocess = _timm_preprocess(224)
        self.native_dim = model.num_features
        self.modality = 'image'
        self.load_status = 'IMAGE_ONLY'

    def _load_dinov2(self, device: str):
        import torch
        import timm

        self.required_package = 'timm'
        ckpt = self.root / 'dinov2_vitl14_pretrain.pth'
        self.checkpoint = ckpt
        model = timm.create_model(
            'vit_large_patch14_dinov2', pretrained=False, num_classes=0, img_size=518)
        sd = torch.load(ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        model.load_state_dict(sd, strict=False)
        self._model = model.to(device).eval()
        self._preprocess = _timm_preprocess(518)
        self.native_dim = model.num_features
        self.modality = 'image'
        self.load_status = 'IMAGE_ONLY'

    def _load_lae_dino(self, device: str):
        self.modality = 'image'
        self.load_status = 'BLOCKED'
        self.error = 'LAE-DINO is a finetuned detector checkpoint, not a standalone CLIP-like encoder API in-repo'

    def encode_text(self, texts: List[str], device: str = 'cpu') -> np.ndarray:
        if self.modality not in ('text', 'text_image'):
            raise RuntimeError(f'{self.name} is {self.modality}, not text')
        import torch

        tok = self._tokenizer(texts)
        if hasattr(tok, 'to'):
            tok = tok.to(device)
        with torch.no_grad():
            emb = self._model.encode_text(tok)
        out = emb.detach().cpu().numpy().astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)

    def encode_image_paths(self, paths: List[Path], device: str = 'cpu') -> np.ndarray:
        if self.modality not in ('image', 'text_image'):
            raise RuntimeError(f'{self.name} is {self.modality}, not image')
        import torch
        from PIL import Image

        tensors = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            tensors.append(self._preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            if self.name in ('clip', 'open_clip', 'remoteclip', 'georsclip'):
                emb = self._model.encode_image(batch)
            else:
                emb = self._model(batch)
        out = emb.detach().cpu().numpy().astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)


def _timm_preprocess(size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def discover_encoder_roots(pretrained_root: Path) -> Dict[str, Path]:
    mapping = {
        'clip': pretrained_root / 'clip',
        'open_clip': pretrained_root / 'open_clip',
        'remoteclip': pretrained_root / 'remoteclip',
        'georsclip': pretrained_root / 'georsclip',
        'dino': pretrained_root / 'dino',
        'dinov2': pretrained_root / 'dinov2',
        'lae-dino': pretrained_root / 'lae-dino',
    }
    return {k: v for k, v in mapping.items() if v.is_dir()}


def build_adapter(name: str, pretrained_root: Path, device: str = 'cpu') -> EncoderAdapter:
    roots = discover_encoder_roots(pretrained_root)
    if name not in roots:
        return EncoderAdapter(
            name=name, root=pretrained_root / name, modality='unknown',
            load_status='CHECKPOINT_MISSING',
        )
    adapter = EncoderAdapter(name=name, root=roots[name]).load(device=device)
    if adapter.load_status == 'LOAD_FAILED' and device != 'cpu':
        adapter = EncoderAdapter(name=name, root=roots[name]).load(device='cpu')
    return adapter


def inventory_row(adapter: EncoderAdapter) -> dict:
    return dict(
        encoder_name=adapter.name,
        path=str(adapter.root),
        modality=adapter.modality,
        native_dim=adapter.native_dim,
        load_status=adapter.load_status,
        required_package=adapter.required_package,
        checkpoint=str(adapter.checkpoint or ''),
        arch=adapter.arch,
        error=adapter.error[:500] if adapter.error else '',
        supports_text='yes' if adapter.modality in ('text', 'text_image') else 'no',
        supports_image='yes' if adapter.modality in ('image', 'text_image') else 'no',
    )
