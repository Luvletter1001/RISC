import json

import numpy as np
import pytest
import torch

from projects.OVCapFlow.tools import audit_query_evidence_source as source_audit
from projects.OVCapFlow.tools.audit_query_evidence_source import (
    SOURCE_SCHEMA,
    SourceGateThresholds,
    build_level_centers,
    choose_evidence_indices,
    format_results_tsv_row,
    geometry_reachability,
    make_placebo_logits,
    normalized_points_to_original_pixels,
    require_file_sha256,
    require_unique_image_ids,
    semantic_token_evidence,
    source_gate_decision,
    validate_level_token_count,
    write_report_no_replace,
)


def test_level_centers_are_raster_ordered_and_normalized():
    centers = build_level_centers((2, 2), torch.tensor([[1.0, 1.0]]))
    expected = torch.tensor([[[0.25, 0.25], [0.75, 0.25],
                              [0.25, 0.75], [0.75, 0.75]]])
    assert torch.equal(centers, expected)


def test_evidence_selection_is_stable_masked_and_capped_at_600():
    evidence = torch.arange(0, 605, dtype=torch.float32)[None]
    valid = torch.ones_like(evidence, dtype=torch.bool)
    valid[:, 604] = False
    selected = choose_evidence_indices(evidence, valid, budget=600)
    assert selected.shape == (1, 600)
    assert 604 not in selected[0].tolist()
    assert selected[0, 0].item() == 603
    tied = choose_evidence_indices(
        torch.tensor([[4.0, 4.0, 3.0]]),
        torch.tensor([[True, True, True]]),
        budget=2)
    assert tied.tolist() == [[0, 1]]


def test_semantic_evidence_masks_padding_before_max():
    logits = torch.tensor([[[1.0, 999.0], [2.0, 999.0]]])
    text_mask = torch.tensor([[True, False]])
    assert torch.equal(semantic_token_evidence(logits, text_mask),
                       torch.tensor([[1.0, 2.0]]))


def test_oriented_center_inside_definition_matches_dense400():
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
    centers = torch.tensor([[0.59, 0.50], [0.61, 0.50]])
    reached = geometry_reachability(gt, centers)
    assert reached.tolist() == [True]


def test_normalized_points_use_resized_shape_and_scale_factor():
    points = np.array([[0.5, 0.25]], dtype=np.float32)
    pixels = normalized_points_to_original_pixels(
        points, img_shape=(800, 1000), scale_factor=(0.5, 0.5))
    np.testing.assert_array_equal(
        pixels, np.array([[1000.0, 400.0]], dtype=np.float32))


def test_placebos_are_deterministic_and_break_distinct_associations():
    logits = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    centers = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    a = make_placebo_logits(logits, centers, 'spatial', 2026071601)
    b = make_placebo_logits(logits, centers, 'spatial', 2026071601)
    semantic = make_placebo_logits(logits, centers, 'semantic', 2026071602)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(a[0], logits)
    assert not torch.equal(a[1], centers)
    assert sorted(a[1][0].tolist()) == sorted(centers[0].tolist())
    assert torch.equal(semantic[1], centers)
    assert not torch.equal(semantic[0], logits)
    for token_index in range(logits.shape[2]):
        actual = torch.sort(semantic[0][0, :, token_index]).values
        expected = torch.sort(logits[0, :, token_index]).values
        assert torch.equal(actual, expected)


def test_placebos_shuffle_only_valid_spatial_indices():
    logits = torch.tensor([[[0.0, 10.0], [1.0, 11.0],
                            [999.0, 999.0], [3.0, 13.0]]])
    centers = torch.tensor([[[0.1, 0.1], [0.2, 0.2],
                             [99.0, 99.0], [0.4, 0.4]]])
    valid = torch.tensor([[True, True, False, True]])

    spatial = source_audit._make_placebo_logits_with_valid_mask(
        logits, centers, 'spatial', 2026071601, valid)
    semantic = source_audit._make_placebo_logits_with_valid_mask(
        logits, centers, 'semantic', 2026071602, valid)
    uniform = source_audit._make_placebo_logits_with_valid_mask(
        logits, centers, 'uniform', 2026071602, valid)

    assert torch.equal(spatial[0], logits)
    assert torch.equal(spatial[1][0, 2], centers[0, 2])
    assert sorted(spatial[1][0, valid[0]].tolist()) == \
        sorted(centers[0, valid[0]].tolist())
    assert torch.equal(semantic[1], centers)
    assert torch.equal(semantic[0][0, 2], logits[0, 2])
    for token_index in range(logits.shape[2]):
        actual = torch.sort(
            semantic[0][0, valid[0], token_index]).values
        expected = torch.sort(logits[0, valid[0], token_index]).values
        assert torch.equal(actual, expected)
    assert torch.equal(uniform[0][0, valid[0]], torch.zeros(3, 2))
    assert torch.equal(uniform[0][0, 2], logits[0, 2])

    text_mask = torch.tensor([[True, True]])
    for variant_logits, _ in (spatial, semantic, uniform):
        evidence = semantic_token_evidence(variant_logits, text_mask)
        selected = choose_evidence_indices(evidence, valid, budget=3)
        assert 2 not in selected[0].tolist()


def test_source_gate_boundaries_are_individually_inclusive_and_fail_closed():
    t = SourceGateThresholds()
    common = dict(overall=0.30, spatial_shuffle=0.20,
                  semantic_shuffle=0.20, base14=0.30, novel4=0.20,
                  finite=True, hashes_match=True)

    reach_boundary = {**common, 'overall': 0.25,
                      'spatial_shuffle': 0.20,
                      'semantic_shuffle': 0.20}
    reach_gate = 'reachability_at_least_25pct'
    assert source_gate_decision(reach_boundary, t)['gates'][reach_gate]
    reach_below = {**reach_boundary, 'overall': 0.249999999}
    assert not source_gate_decision(reach_below, t)['gates'][reach_gate]

    prior_e = 0.210711449
    margin_gate = 'beats_prior_e_by_3pp'
    margin_boundary = {**common, 'overall': prior_e + 0.03,
                       'spatial_shuffle': 0.18,
                       'semantic_shuffle': 0.18}
    assert source_gate_decision(margin_boundary, t)['gates'][margin_gate]
    margin_below = {**margin_boundary,
                    'overall': prior_e + 0.03 - 0.000000001}
    assert not source_gate_decision(margin_below, t)['gates'][margin_gate]

    placebo_boundary = {**common, 'spatial_shuffle': 0.25,
                        'semantic_shuffle': 0.25}
    placebo = source_gate_decision(placebo_boundary, t)
    assert placebo['gates']['placebo_sensitivity']
    assert placebo['placebo_subchecks']['spatial_shuffle_loses_5pp']
    assert placebo['placebo_subchecks']['semantic_shuffle_loses_5pp']
    spatial_below = {**placebo_boundary, 'spatial_shuffle': 0.250000001}
    spatial = source_gate_decision(spatial_below, t)
    assert not spatial['gates']['placebo_sensitivity']
    assert not spatial['placebo_subchecks']['spatial_shuffle_loses_5pp']
    semantic_below = {**placebo_boundary, 'semantic_shuffle': 0.250000001}
    semantic = source_gate_decision(semantic_below, t)
    assert not semantic['gates']['placebo_sensitivity']
    assert not semantic['placebo_subchecks']['semantic_shuffle_loses_5pp']

    gap_gate = 'novel_base_gap_at_most_10pp'
    assert source_gate_decision(common, t)['gates'][gap_gate]
    gap_above = {**common, 'base14': 0.300000001}
    assert not source_gate_decision(gap_above, t)['gates'][gap_gate]


def test_report_is_canonical_no_replace_and_schema_locked(tmp_path):
    path = tmp_path / 'source_report.json'
    payload = {
        'schema': SOURCE_SCHEMA,
        'status': 'PASS',
        'decision': {
            'gates': {
                'reachability_at_least_25pct': True,
                'beats_prior_e_by_3pp': True,
                'placebo_sensitivity': True,
                'novel_base_gap_at_most_10pp': True,
                'finite_reproducible_provenance': True,
            },
            'failed_gates': [],
        },
        'frozen_contract': {},
        'provenance': {},
        'counts': {},
        'reachability': {},
        'placebos': {},
        'strata': {},
        'empty_images': {},
        'finite_checks': {},
    }
    first_sha = write_report_no_replace(path, payload)
    expected = b'''{
  "counts": {},
  "decision": {
    "failed_gates": [],
    "gates": {
      "beats_prior_e_by_3pp": true,
      "finite_reproducible_provenance": true,
      "novel_base_gap_at_most_10pp": true,
      "placebo_sensitivity": true,
      "reachability_at_least_25pct": true
    }
  },
  "empty_images": {},
  "finite_checks": {},
  "frozen_contract": {},
  "placebos": {},
  "provenance": {},
  "reachability": {},
  "schema": "ov-capflow-query-evidence-source-v1",
  "status": "PASS",
  "strata": {}
}
'''
    assert path.read_bytes() == expected
    assert first_sha == (
        '176205291029251795b31251d11f512f88ef4510c278c4c7332de3a426377942')
    with pytest.raises(FileExistsError):
        write_report_no_replace(path, payload)

    missing = dict(payload)
    missing.pop('finite_checks')
    with pytest.raises(ValueError, match='schema'):
        write_report_no_replace(tmp_path / 'missing.json', missing)

    altered = {**payload, 'schema': 'ov-capflow-query-evidence-source-v2'}
    with pytest.raises(ValueError, match='schema'):
        write_report_no_replace(tmp_path / 'altered.json', altered)

    nonfinite = {**payload, 'reachability': {'overall': float('nan')}}
    with pytest.raises(ValueError, match='Out of range'):
        write_report_no_replace(tmp_path / 'nonfinite.json', nonfinite)


def test_results_row_uses_external_report_digest():
    payload = {'schema': SOURCE_SCHEMA, 'status': 'PASS',
               'decision': {'gates': {'finite': True}},
               'reachability': {'overall': 0.31}}
    digest = 'a' * 64
    row = format_results_tsv_row(payload, digest)
    assert row.startswith('8-D160-QAF-SOURCE\t')
    assert digest in row


def test_nonfinite_logits_fail_closed():
    logits = torch.tensor([[[float('nan'), 0.0]]])
    with pytest.raises(ValueError, match='finite'):
        semantic_token_evidence(logits, torch.tensor([[True, True]]))


def test_no_valid_prompt_token_fails_closed():
    with pytest.raises(ValueError, match='valid prompt token'):
        semantic_token_evidence(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_wrong_level_token_count_fails_closed():
    with pytest.raises(ValueError, match='level token count'):
        validate_level_token_count((32, 32), observed_token_count=1023)


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    with pytest.raises(ValueError, match='SHA256'):
        require_file_sha256(manifest, '0' * 64)


def test_missing_gate_field_fails_closed():
    metrics = dict(overall=0.31, spatial_shuffle=0.20,
                   semantic_shuffle=0.20, base14=0.30,
                   finite=True, hashes_match=True)
    with pytest.raises(ValueError, match='novel4'):
        source_gate_decision(metrics, SourceGateThresholds())


def test_duplicate_image_id_fails_closed():
    with pytest.raises(ValueError, match='duplicate image id'):
        require_unique_image_ids(['P0001', 'P0001'])


def test_padded_text_columns_never_enter_evidence():
    full = torch.tensor([[[2.0, 1.0, 999.0, 999.0]]])
    runtime_mask = torch.tensor([[True, True, False, False]])
    reduced = semantic_token_evidence(full, runtime_mask)
    assert torch.equal(reduced, torch.tensor([[2.0]]))


def test_padded_spatial_tokens_never_convert_to_pixels():
    evidence = torch.tensor([[1.0, 2.0, 999.0, 3.0]])
    valid = torch.tensor([[True, True, False, True]])
    selected = choose_evidence_indices(evidence, valid, budget=2)
    assert selected.tolist() == [[3, 1]]
    centers = np.array([[0.75, 0.75], [0.25, 0.25]], dtype=np.float32)
    pixels = normalized_points_to_original_pixels(
        centers, img_shape=(1024, 1024), scale_factor=(1.0, 1.0))
    assert pixels.shape == (2, 2)


def test_scientific_fail_still_writes_complete_report(tmp_path):
    metrics = dict(overall=0.20, spatial_shuffle=0.19,
                   semantic_shuffle=0.19, base14=0.20, novel4=0.20,
                   finite=True, hashes_match=True)
    decision = source_gate_decision(metrics, SourceGateThresholds())
    payload = {
        'schema': SOURCE_SCHEMA, 'status': decision['status'],
        'decision': decision, 'frozen_contract': {}, 'provenance': {},
        'counts': {}, 'reachability': metrics, 'placebos': {},
        'strata': {}, 'empty_images': {}, 'finite_checks': {},
    }
    digest = write_report_no_replace(tmp_path / 'source_report.json', payload)
    assert decision['status'] == 'FAIL'
    assert len(digest) == 64
    assert json.loads((tmp_path / 'source_report.json').read_text()) == payload
