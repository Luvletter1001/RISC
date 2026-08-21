#!/usr/bin/env python
"""Fixed success criteria for full repair training (documented in fres_00b)."""

FULL_SUCCESS_CRITERIA = {
    'p0148_final_sv_ratio_drop_min': 0.30,
    'unmatched_sv_ratio_drop_min': 0.30,
    'overall_ap50_drop_max_points': 1.0,
    'small_vehicle_ap50_drop_max_points': 2.0,
    'large_vehicle_flood_max_ratio': 0.50,
    'ship_flood_max_ratio': 0.50,
    'detection_total_collapse_min_ratio': 0.50,
    'notes': [
        'large_vehicle/ship must not become new flood class (final ratio > 0.5 on P0148)',
        'detection_total must not fall below 50% of baseline (avoid empty-output fix)',
    ],
}
