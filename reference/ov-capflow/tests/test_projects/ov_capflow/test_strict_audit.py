from projects.OVCapFlow.tools.audit_strict_inference import scan_calls


def test_strict_scan_reports_forbidden_calls(tmp_path):
    source = tmp_path / 'forbidden.py'
    source.write_text(
        'def choose(values):\n    return values.topk(1)\n',
        encoding='utf-8')

    assert scan_calls(tmp_path) == [{
        'path': str(source),
        'line': 2,
        'call': 'topk',
    }]


def test_runtime_call_recorder_catches_topk_and_restores_torch():
    import torch

    from projects.OVCapFlow.tools.audit_strict_inference import (
        record_forbidden_runtime_calls, )

    original = torch.topk
    with record_forbidden_runtime_calls() as hits:
        values, indices = torch.topk(torch.tensor([1.0, 3.0]), 1)
        assert values.item() == 3.0
        assert indices.item() == 1
    assert hits == [{'call': 'torch.topk', 'count': 1}]
    assert torch.topk is original
