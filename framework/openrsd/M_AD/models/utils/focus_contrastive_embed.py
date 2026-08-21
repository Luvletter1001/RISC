from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrientationConditionedContrastiveEmbed(nn.Module):
    """Contrastive classification with optional Fourier-conditioned supports.

    This module mirrors OpenRSD's native support-bank classifier and inserts a
    zero-initialized residual adapter only after the native visual support
    mapping. When the residual is zero, it falls back to the exact baseline
    matmul path to keep checkpoint-equivalence checks strict.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_focus_debug: Optional[Dict[str, Any]] = None

    def get_matching_scores(self, matching_scores: Tensor,
                            support_labels: Tensor, **kwargs) -> Tensor:
        batch, positions, support_count = matching_scores.shape
        support_labels = support_labels[:, None, :].expand(
            batch, positions, support_count)
        scores = matching_scores.clone()
        scores[support_labels < 0] = -10

        num_classes = kwargs["num_classes"]
        num_in_classes = kwargs["num_in_classes"]
        support_shot = kwargs["support_shot"]

        len_in_classes = num_in_classes * support_shot
        cls_scores = torch.full(
            [batch, positions, num_classes],
            float("-inf"),
            device=matching_scores.device)
        in_match_scores = scores[:, :, :len_in_classes].reshape(
            batch, positions, num_in_classes, support_shot)
        cls_scores[:, :, :num_in_classes] = torch.max(
            in_match_scores, dim=-1)[0]

        if support_count > len_in_classes:
            out_match_scores = scores[:, :, len_in_classes:]
            cls_scores[:, :, -1] = torch.max(out_match_scores, dim=-1)[0]
        return cls_scores

    def _match_logits(self, mapped_pred: Tensor,
                      mapped_support: Tensor) -> Tensor:
        w = F.normalize(mapped_support, dim=-1)
        if w.dim() == 2:
            return mapped_pred @ w.transpose(0, 1)
        if w.dim() == 3:
            return mapped_pred @ w.transpose(-1, -2)
        if w.dim() == 4:
            if w.shape[0] != mapped_pred.shape[0]:
                raise ValueError(
                    "batched support batch dimension does not match "
                    f"predictions: {tuple(w.shape)} vs {tuple(mapped_pred.shape)}")
            if w.shape[1] == 1 and mapped_pred.shape[1] != 1:
                w = w.expand(w.shape[0], mapped_pred.shape[1],
                             w.shape[2], w.shape[3])
            if w.shape[1] != mapped_pred.shape[1]:
                raise ValueError(
                    "dense support position dimension does not match "
                    f"predictions: {tuple(w.shape)} vs {tuple(mapped_pred.shape)}")
            return torch.matmul(
                mapped_pred.unsqueeze(-2),
                w.transpose(-1, -2)).squeeze(-2)
        raise ValueError(
            "mapped_support must be [S,D], [B,S,D], or [B,N,S,D], "
            f"got {tuple(mapped_support.shape)}")

    def _baseline_logits(
            self,
            mapped_pred: Tensor,
            mapped_support: Tensor,
            support_labels: Tensor,
            log_scale: Tensor,
            bias: Tensor,
            height: int,
            width: int,
            **kwargs) -> Tensor:
        match_logit = self._match_logits(mapped_pred, mapped_support)
        scaled_logit = match_logit * log_scale.exp() + bias
        cls_logits = self.get_matching_scores(
            scaled_logit, support_labels, **kwargs)
        batch = mapped_pred.shape[0]
        return cls_logits.reshape(
            batch, height, width, cls_logits.shape[-1]).permute(
                0, 3, 1, 2).contiguous()

    def forward(
            self,
            pred_embeds: Tensor,
            support_feats: Tensor,
            support_labels: Tensor,
            visual_fc: nn.Module,
            text_fc: nn.Module,
            log_scale: Tensor,
            bias: Tensor,
            orientation_learner: Optional[nn.Module] = None,
            support_adapter: Optional[nn.Module] = None,
            text_support_adapter: Optional[nn.Module] = None,
            dual_support_fusion: Optional[nn.Module] = None,
            visual_support_feats: Optional[Tensor] = None,
            text_support_feats: Optional[Tensor] = None,
            eqtext_enabled: bool = False,
            dual_fusion_enabled: bool = False,
            head_residual_enabled: bool = True,
            focus_enabled: bool = True,
            return_focus_debug: bool = False,
            **kwargs) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        batch, dim, height, width = pred_embeds.shape
        mapped_pred = pred_embeds.permute(0, 2, 3, 1).reshape(
            batch, height * width, dim)

        align_style = kwargs["align_style"]
        if align_style in ["labelled", "pure_img"]:
            mapped_pred = visual_fc(mapped_pred)
            mapped_support = visual_fc(support_feats)
            mapped_visual_support = (
                visual_fc(visual_support_feats)
                if visual_support_feats is not None else mapped_support)
            mapped_text_support = (
                visual_fc(text_support_feats)
                if text_support_feats is not None else mapped_support)
        else:
            raise Exception(f"Unknown alignment style {align_style}")

        debug: Dict[str, Any] = {"enabled": False}
        if not focus_enabled or orientation_learner is None or support_adapter is None:
            logits = self._baseline_logits(
                mapped_pred, mapped_support, support_labels,
                log_scale, bias, height, width, **kwargs)
            self.last_focus_debug = debug
            if return_focus_debug:
                return logits, debug
            return logits

        orientation = orientation_learner(pred_embeds)
        conditioned_visual = mapped_visual_support
        adapter_debug: Dict[str, Any] = {}
        if support_adapter is not None and head_residual_enabled:
            conditioned_visual, adapter_debug = support_adapter(
                mapped_visual_support,
                orientation["fourier_code"],
                orientation["confidence"],
                support_labels=support_labels,
                class_names=kwargs.get("focus_class_names"))

        conditioned_text = None
        text_debug: Dict[str, Any] = {}
        if eqtext_enabled and text_support_adapter is not None:
            conditioned_text, text_debug = text_support_adapter(
                mapped_text_support,
                orientation["fourier_code"],
                orientation["confidence"],
                support_labels=support_labels,
                class_names=kwargs.get("focus_class_names"))

        fusion_debug: Dict[str, Any] = {}
        if dual_fusion_enabled and dual_support_fusion is not None:
            visual_alpha = getattr(support_adapter, "alpha", None)
            text_alpha = getattr(text_support_adapter, "alpha_t", None)
            conditioned_support, fusion_debug = dual_support_fusion(
                conditioned_visual,
                conditioned_text,
                text_enabled=conditioned_text is not None,
                visual_alpha=visual_alpha,
                text_alpha=text_alpha)
        elif eqtext_enabled and conditioned_text is not None and not head_residual_enabled:
            conditioned_support = conditioned_text
        else:
            conditioned_support = conditioned_visual

        debug = {
            "enabled": True,
            "orientation": {
                "theta": orientation["theta"].detach(),
                "confidence": orientation["confidence"].detach(),
            },
            "adapter": adapter_debug,
            "eqtext": text_debug,
            "dual_fusion": fusion_debug,
        }

        delta_norm = adapter_debug.get("delta_norm")
        text_delta_norm = text_debug.get("text_delta_norm")
        visual_zero = (
            delta_norm is None
            or (hasattr(delta_norm, "detach")
                and torch.max(delta_norm).detach() == 0))
        text_zero = (
            text_delta_norm is None
            or (hasattr(text_delta_norm, "detach")
                and torch.max(text_delta_norm).detach() == 0))
        zero_fusion_fallback = fusion_debug.get(
            "dual_fallback", "") in {"", "zero_alpha", "text_disabled"}
        baseline_support = (
            mapped_visual_support if dual_fusion_enabled else mapped_support)
        if (not torch.is_grad_enabled()
                and visual_zero
                and text_zero
                and zero_fusion_fallback):
            logits = self._baseline_logits(
                mapped_pred, baseline_support, support_labels,
                log_scale, bias, height, width, **kwargs)
        else:
            match_logit = self._match_logits(mapped_pred, conditioned_support)
            scaled_logit = match_logit * log_scale.exp() + bias
            cls_logits = self.get_matching_scores(
                scaled_logit, support_labels, **kwargs)
            logits = cls_logits.reshape(
                batch, height, width, cls_logits.shape[-1]).permute(
                    0, 3, 1, 2).contiguous()

        self.last_focus_debug = debug
        if return_focus_debug:
            return logits, debug
        return logits
