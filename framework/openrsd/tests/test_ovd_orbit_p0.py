import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from M_Tools.analysis.ovd_orbit_p0 import (
    OrbitP0Error,
    ScoreCarrier,
    build_manifest,
    collapse_carriers,
    write_receipt,
)
from M_Tools.analysis.ovd_orbit_p0_export import normalize_level_outputs
from M_Tools.analysis.run_ovd_orbit_p0 import main


def carrier(*, view='rot000', scene='scene_a', level=0, row=3,
            scores=(.2, .7, .1)):
    return ScoreCarrier(
        view_id=view,
        scene_id=scene,
        source=(level, row),
        box=np.array([10., 20., 8., 4., .1], dtype=np.float32),
        scores=np.array(scores, dtype=np.float32),
    )


def test_score_carrier_rejects_non_c4_view():
    with pytest.raises(OrbitP0Error, match='C4'):
        carrier(view='rot030')


def test_collapse_preserves_full_score_vector_and_rejects_conflict():
    np.testing.assert_allclose(
        collapse_carriers([carrier(), carrier()])[0].scores, [.2, .7, .1])
    with pytest.raises(OrbitP0Error, match='conflicting'):
        collapse_carriers([carrier(), carrier(scores=(.3, .6, .1))])


def test_manifest_rejects_p0148_scene_and_mixed_vocabulary_hashes():
    with pytest.raises(OrbitP0Error, match='P0148'):
        build_manifest(
            [carrier(scene='P0148__1024__651___0')],
            vocabulary_hash='a',
            prompt_hash='p',
        )


def test_normalize_level_outputs_preserves_each_full_score_vector():
    boxes = torch.tensor([[1., 2., 3., 4., .1], [5., 6., 7., 8., .2]])
    scores = torch.tensor([[.2, .3, .5], [.7, .2, .1]])
    source_scores = scores.clone()

    records = normalize_level_outputs(
        view_id='rot090', scene_id='scene_a', level=2, boxes=boxes, scores=scores)

    assert [record.source for record in records] == [(2, 0), (2, 1)]
    np.testing.assert_allclose(records[1].scores, [.7, .2, .1])
    assert torch.equal(scores, source_scores)


def test_normalize_level_outputs_rejects_invalid_score_matrix():
    with pytest.raises(OrbitP0Error, match='scores'):
        normalize_level_outputs(
            view_id='rot090',
            scene_id='scene_a',
            level=2,
            boxes=torch.ones(1, 5),
            scores=torch.ones(3),
        )


def test_write_receipt_serializes_full_scores_and_digests(tmp_path):
    carriers = collapse_carriers([carrier()])
    manifest = build_manifest(
        carriers,
        vocabulary_hash='vocabulary-hash',
        prompt_hash='prompt-hash',
    )

    receipt = write_receipt(tmp_path / 'receipt', manifest, carriers)

    assert receipt['status'] == 'E0_CONTRACT_READY'
    assert receipt['carrier_count'] == 1
    assert len(receipt['sha256']) == 64
    record = json.loads((tmp_path / 'receipt' / 'carriers.jsonl').read_text())
    assert record['scores'] == pytest.approx([.2, .7, .1])
    assert json.loads((tmp_path / 'receipt' / 'receipt.json').read_text()) == receipt


def test_cli_dry_run_validates_fixture_without_creating_output(tmp_path, capsys):
    fixture = Path(__file__).parent / 'fixtures' / 'ovd_orbit_p0_sample.jsonl'
    output_dir = tmp_path / 'dry_run_output'

    result = main([
        '--input-jsonl', str(fixture),
        '--output-dir', str(output_dir),
        '--vocabulary-hash', 'fixture-vocabulary',
        '--prompt-hash', 'fixture-prompt',
        '--dry-run',
    ])

    assert result == 0
    assert 'E0_CONTRACT_READY' in capsys.readouterr().out
    assert not output_dir.exists()


def test_cli_script_runs_from_openrsd_root(tmp_path):
    fixture = Path(__file__).parent / 'fixtures' / 'ovd_orbit_p0_sample.jsonl'
    root = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONNOUSERSITE='1',
               PYTHONPYCACHEPREFIX=str(tmp_path / 'pycache'))

    result = subprocess.run(
        [
            sys.executable,
            'M_Tools/analysis/run_ovd_orbit_p0.py',
            '--input-jsonl', str(fixture),
            '--output-dir', str(tmp_path / 'dry_run_output'),
            '--vocabulary-hash', 'fixture-vocabulary',
            '--prompt-hash', 'fixture-prompt',
            '--dry-run',
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'E0_CONTRACT_READY' in result.stdout
    assert not (tmp_path / 'dry_run_output').exists()
    with pytest.raises(OrbitP0Error, match='vocabulary'):
        build_manifest(
            [carrier()],
            vocabulary_hash='a',
            prompt_hash='p',
            observed_vocabulary_hashes={'a', 'b'},
        )
