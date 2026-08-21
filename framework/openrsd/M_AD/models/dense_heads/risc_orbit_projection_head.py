from typing import Optional

from mmrotate.registry import MODELS

from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import (
    OpenRotatedRTMDetSepBNHead,
)
from M_AD.models.utils.risc_orbit_projection import RISCOrbitLowRankProjection


@MODELS.register_module()
class RISCOrbitProjectedRTMDetHead(OpenRotatedRTMDetSepBNHead):
    """OpenRSD head with one class-shared low-rank semantic projection."""

    def __init__(self, *args, orbit_projection: Optional[dict] = None,
                 **kwargs) -> None:
        projection_cfg = dict(orbit_projection or {})
        projection_cfg.pop("enable", None)

        focus_cfg = dict(kwargs.get("focus_ovd") or {})
        focus_cfg["enable"] = True
        orientation_cfg = dict(focus_cfg.get("orientation") or {})
        harmonic_orders = tuple(
            orientation_cfg.get("harmonic_orders", (2, 4, 6)))
        focus_cfg["orientation"] = orientation_cfg
        adapter_cfg = dict(focus_cfg.get("adapter") or {})
        adapter_cfg["enable"] = True
        focus_cfg["adapter"] = adapter_cfg
        kwargs["focus_ovd"] = focus_cfg
        kwargs["use_focus_ovd"] = True

        super().__init__(*args, **kwargs)

        self.focus_support_adapter = RISCOrbitLowRankProjection(
            support_dim=self.embed_dims,
            code_dim=2 * len(harmonic_orders),
            **projection_cfg,
        )
        self.focus_ovd_enable = True
        self.focus_head_residual_enabled = True
