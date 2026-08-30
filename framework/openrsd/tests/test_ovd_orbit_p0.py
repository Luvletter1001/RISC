import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1 import (
    OpenRotatedRTMDetSepBNHead,
)
from M_Tools.analysis.ovd_orbit_p0 import (
    C4_VIEW_IDS,
    OVD_ORBIT_P0_VIEW_IDS,
    OrbitP0Error,
    ScoreCarrier,
    build_manifest,
    collapse_carriers,
    write_receipt,
)
from M_Tools.analysis.ovd_orbit_p0_export import (
    OvdOrbitFullLogitSink,
    normalize_level_outputs,
)
from M_Tools.analysis.run_ovd_orbit_p0 import load_carriers, main


def _bare_ovd_orbit_p0_head():
    head = OpenRotatedRTMDetSepBNHead.__new__(OpenRotatedRTMDetSepBNHead)
    nn.Module.__init__(head)
    return head


def test_head_constructor_defaults_orbit_p0_sink_to_none():
    head = OpenRotatedRTMDetSepBNHead(
        num_classes=2,
        in_channels=8,
        embed_dims=8,
    )

    assert head.ovd_orbit_p0_sink is None


def test_head_accepts_none_orbit_p0_sink():
    head = _bare_ovd_orbit_p0_head()

    head.set_ovd_orbit_p0_sink(None)

    assert head.ovd_orbit_p0_sink is None


def test_head_rejects_record_only_orbit_p0_sink_with_missing_lifecycle():
    class RecordOnlySink:
        def record_level(self, *, level, boxes, scores):
            pass

    head = _bare_ovd_orbit_p0_head()

    with pytest.raises(TypeError) as error:
        head.set_ovd_orbit_p0_sink(RecordOnlySink())

    assert 'begin_image' in str(error.value)
    assert 'end_image' in str(error.value)
    assert 'abort_image' in str(error.value)


def test_head_accepts_complete_orbit_p0_sink_lifecycle():
    class CompleteSink:
        def begin_image(self, *, scene_id, view_id):
            pass

        def record_level(self, *, level, boxes, scores):
            pass

        def end_image(self):
            pass

        def abort_image(self):
            pass

    head = _bare_ovd_orbit_p0_head()
    sink = CompleteSink()

    head.set_ovd_orbit_p0_sink(sink)

    assert head.ovd_orbit_p0_sink is sink


def test_head_rejects_orbit_p0_sink_without_record_level():
    head = _bare_ovd_orbit_p0_head()

    with pytest.raises(TypeError, match='record_level'):
        head.set_ovd_orbit_p0_sink(object())


def test_head_accepts_full_logit_orbit_p0_sink():
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()

    head.set_ovd_orbit_p0_sink(sink)

    assert head.ovd_orbit_p0_sink is sink


def test_head_requires_orbit_metadata_only_with_an_attached_sink():
    head = _bare_ovd_orbit_p0_head()

    head.set_ovd_orbit_p0_sink(None)
    assert head._ovd_orbit_p0_metadata({}) is None

    head.set_ovd_orbit_p0_sink(OvdOrbitFullLogitSink())
    with pytest.raises(OrbitP0Error):
        head._ovd_orbit_p0_metadata({})


@pytest.mark.parametrize('img_meta', [
    {
        'ovd_orbit_p0_scene_id': '',
        'ovd_orbit_p0_view_id': 'rot090',
    },
    {
        'ovd_orbit_p0_scene_id': 'scene_a',
        'ovd_orbit_p0_view_id': None,
    },
])
def test_head_rejects_invalid_orbit_p0_metadata(img_meta):
    head = _bare_ovd_orbit_p0_head()
    head.set_ovd_orbit_p0_sink(OvdOrbitFullLogitSink())

    with pytest.raises(OrbitP0Error):
        head._ovd_orbit_p0_metadata(img_meta)


def test_head_returns_exact_valid_orbit_p0_metadata():
    head = _bare_ovd_orbit_p0_head()
    head.set_ovd_orbit_p0_sink(OvdOrbitFullLogitSink())
    img_meta = {
        'ovd_orbit_p0_scene_id': 'scene_a',
        'ovd_orbit_p0_view_id': 'rot090',
    }

    assert head._ovd_orbit_p0_metadata(img_meta) == ('scene_a', 'rot090')


@pytest.mark.parametrize('view_id', (
    'rot000', 'rot090', 'rot180', 'rot270', 'rot000_a', 'rot000_b'))
def test_attached_head_and_export_sink_accept_orbit_p0_view_id(view_id):
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()
    head.set_ovd_orbit_p0_sink(sink)

    assert head._ovd_orbit_p0_metadata({
        'ovd_orbit_p0_scene_id': 'scene_a',
        'ovd_orbit_p0_view_id': view_id,
    }) == ('scene_a', view_id)
    sink.begin_image(scene_id='scene_a', view_id=view_id)
    sink.end_image()


@pytest.mark.parametrize('view_id', ('id', 'rot030'))
def test_attached_head_and_export_sink_reject_orbit_p0_view_id(view_id):
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()
    head.set_ovd_orbit_p0_sink(sink)
    img_meta = {
        'ovd_orbit_p0_scene_id': 'scene_a',
        'ovd_orbit_p0_view_id': view_id,
    }

    with pytest.raises(OrbitP0Error):
        head._ovd_orbit_p0_metadata(img_meta)
    with pytest.raises(OrbitP0Error):
        sink.begin_image(scene_id='scene_a', view_id=view_id)


def test_legacy_c4_view_contract_remains_public_and_compatible():
    assert C4_VIEW_IDS == frozenset({'rot000', 'rot090', 'rot180', 'rot270'})
    assert OVD_ORBIT_P0_VIEW_IDS == C4_VIEW_IDS | {'rot000_a', 'rot000_b'}
    assert carrier(view='rot000').view_id == 'rot000'


def test_identity_repeat_views_remain_distinct_during_collapse():
    records = collapse_carriers([
        carrier(view='rot000_a'),
        carrier(view='rot000_b'),
    ])

    assert [record.view_id for record in records] == ['rot000_a', 'rot000_b']


class _FakeOrbitP0AngleCoder:
    encode_size = 1

    def decode(self, angle_pred, *, keepdim):
        assert keepdim is True
        return angle_pred + .5


class _FakeOrbitP0BBoxCoder:
    def __init__(self):
        self.calls = []

    def decode(self, priors, bbox_pred, *, max_shape):
        self.calls.append((priors.clone(), bbox_pred.clone(), max_shape))
        return torch.cat((priors + bbox_pred[:, :2], bbox_pred[:, 2:]), dim=-1)


def test_head_capture_helper_records_calibrated_full_scores_and_decoded_boxes():
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()
    bbox_coder = _FakeOrbitP0BBoxCoder()
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = bbox_coder
    head.set_ovd_orbit_p0_sink(sink)
    raw_scores = torch.tensor([[-2., 2.]], dtype=torch.float32)
    calibrated_scores = torch.tensor([[.2, .8]], dtype=torch.float32)
    source_raw_scores = raw_scores.clone()
    source_calibrated_scores = calibrated_scores.clone()

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    head._record_ovd_orbit_p0_level(
        level_idx=1,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
        angle_pred=torch.tensor([[.1]], dtype=torch.float32),
        priors=torch.tensor([[10., 20.]], dtype=torch.float32),
        img_shape=(32, 32),
    )
    sink.end_image()

    records = sink.snapshot()
    assert [record.source for record in records] == [(1, 0)]
    np.testing.assert_allclose(records[0].scores, [-2., 2.])
    np.testing.assert_allclose(records[0].calibrated_scores, [.2, .8])
    np.testing.assert_allclose(records[0].box, [11., 22., 3., 4., .6])
    assert bbox_coder.calls[0][2] == (32, 32)
    assert torch.equal(raw_scores, source_raw_scores)
    assert torch.equal(calibrated_scores, source_calibrated_scores)


def test_head_capture_helper_supports_required_minimal_sink_api():
    class MinimalSink:
        def __init__(self):
            self.active = False
            self.records = []

        def begin_image(self, *, scene_id, view_id):
            self.active = True

        def record_level(self, *, level, boxes, scores):
            assert self.active
            self.records.append((level, boxes.clone(), scores.clone()))

        def end_image(self):
            self.active = False

        def abort_image(self):
            self.active = False

    head = _bare_ovd_orbit_p0_head()
    sink = MinimalSink()
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    head.set_ovd_orbit_p0_sink(sink)
    raw_scores = torch.tensor([[-2., 2.]], dtype=torch.float32)
    calibrated_scores = torch.tensor([[.2, .8]], dtype=torch.float32)

    sink.begin_image(scene_id='scene_a', view_id='rot000')
    head._record_ovd_orbit_p0_level(
        level_idx=1,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
        angle_pred=torch.tensor([[.1]], dtype=torch.float32),
        priors=torch.tensor([[10., 20.]], dtype=torch.float32),
        img_shape=(32, 32),
    )
    sink.end_image()

    assert len(sink.records) == 1
    assert sink.records[0][0] == 1
    torch.testing.assert_close(sink.records[0][2], raw_scores)


def test_ovd_sink_optional_capability_preserves_calibrated_diagnostics():
    sink = OvdOrbitFullLogitSink()
    raw_scores = torch.tensor([[-2., 2.]], dtype=torch.float32)
    calibrated_scores = torch.tensor([[.2, .8]], dtype=torch.float32)

    sink.begin_image(scene_id='scene_a', view_id='rot000')
    sink.record_level_with_calibrated_scores(
        level=1,
        boxes=torch.tensor([[1., 2., 3., 4., .1]], dtype=torch.float32),
        scores=raw_scores,
        calibrated_scores=calibrated_scores,
    )
    sink.end_image()

    record = sink.snapshot()[0]
    np.testing.assert_allclose(record.scores, [-2., 2.])
    np.testing.assert_allclose(record.calibrated_scores, [.2, .8])


def test_head_rejects_noncallable_optional_calibrated_capability():
    class MalformedOptionalSink:
        record_level_with_calibrated_scores = None

        def begin_image(self, *, scene_id, view_id):
            pass

        def record_level(self, *, level, boxes, scores):
            pass

        def end_image(self):
            pass

        def abort_image(self):
            pass

    head = _bare_ovd_orbit_p0_head()

    with pytest.raises(
            TypeError, match='record_level_with_calibrated_scores'):
        head.set_ovd_orbit_p0_sink(MalformedOptionalSink())


def test_head_does_not_suppress_type_error_from_optional_calibrated_capability():
    class FailingOptionalSink:
        def __init__(self):
            self.legacy_record_called = False

        def begin_image(self, *, scene_id, view_id):
            pass

        def record_level(self, *, level, boxes, scores):
            self.legacy_record_called = True

        def record_level_with_calibrated_scores(
                self, *, level, boxes, scores, calibrated_scores):
            raise TypeError('optional capability internal failure')

        def end_image(self):
            pass

        def abort_image(self):
            pass

    head = _bare_ovd_orbit_p0_head()
    sink = FailingOptionalSink()
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    head.set_ovd_orbit_p0_sink(sink)

    with pytest.raises(TypeError, match='optional capability internal failure'):
        head._record_ovd_orbit_p0_level(
            level_idx=1,
            raw_scores=torch.tensor([[-2., 2.]], dtype=torch.float32),
            calibrated_scores=torch.tensor([[.2, .8]], dtype=torch.float32),
            bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
            angle_pred=torch.tensor([[.1]], dtype=torch.float32),
            priors=torch.tensor([[10., 20.]], dtype=torch.float32),
            img_shape=(32, 32),
        )

    assert not sink.legacy_record_called


def test_head_capture_helper_is_a_complete_noop_without_sink():
    class ExplodingDecoder:
        def decode(self, *args, **kwargs):
            raise AssertionError('none-sink capture must not decode')

    head = _bare_ovd_orbit_p0_head()
    head.angle_coder = ExplodingDecoder()
    head.bbox_coder = ExplodingDecoder()
    head.set_ovd_orbit_p0_sink(None)
    raw_scores = torch.tensor([[-2., 2.]], dtype=torch.float32)
    calibrated_scores = torch.tensor([[.2, .8]], dtype=torch.float32)
    source_raw_scores = raw_scores.clone()
    source_calibrated_scores = calibrated_scores.clone()

    head._record_ovd_orbit_p0_level(
        level_idx=1,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
        angle_pred=torch.tensor([[.1]], dtype=torch.float32),
        priors=torch.tensor([[10., 20.]], dtype=torch.float32),
        img_shape=(32, 32),
    )

    assert torch.equal(raw_scores, source_raw_scores)
    assert torch.equal(calibrated_scores, source_calibrated_scores)


def test_single_prediction_without_sink_does_not_add_capture_decode_or_mutate_inputs(
        monkeypatch):
    class CountingAngleCoder:
        encode_size = 1

        def __init__(self):
            self.calls = 0

        def decode(self, angle_pred, *, keepdim):
            self.calls += 1
            assert keepdim is True
            return angle_pred.clone()

    class CountingBBoxCoder:
        def __init__(self):
            self.calls = 0

        def decode(self, priors, bbox_pred, *, max_shape):
            self.calls += 1
            assert max_shape == (32, 32)
            return bbox_pred.clone()

    head = _bare_ovd_orbit_p0_head()
    angle_coder = CountingAngleCoder()
    bbox_coder = CountingBBoxCoder()
    head.set_ovd_orbit_p0_sink(None)
    head.use_sigmoid_cls = True
    head.gaussian_semantic_scale_enable = False
    head.scale_semantic_calibration_enable = False
    head.test_cfg = {'score_thr': 0, 'nms_pre': -1}
    head.angle_coder = angle_coder
    head.bbox_coder = bbox_coder
    head._bbox_post_process = lambda *, results, **kwargs: results
    cls_score = torch.tensor([[[0.]], [[1.]]], dtype=torch.float32)
    bbox_pred = torch.tensor(
        [[[1.]], [[2.]], [[3.]], [[4.]]], dtype=torch.float32)
    angle_pred = torch.tensor([[[.1]]], dtype=torch.float32)
    priors = torch.tensor([[10., 20.]], dtype=torch.float32)
    source_tensors = [
        cls_score.clone(), bbox_pred.clone(), angle_pred.clone(), priors.clone()]

    def fake_filter(scores, score_thr, nms_pre, results):
        return (
            scores[:1, 0],
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            {key: value[:1] for key, value in results.items()},
        )

    monkeypatch.setattr(
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1.filter_scores_and_topk',
        fake_filter)

    results = head._predict_by_feat_single(
        cls_score_list=[cls_score],
        bbox_pred_list=[bbox_pred],
        angle_pred_list=[angle_pred],
        score_factor_list=[None],
        mlvl_priors=[priors],
        img_meta={'img_shape': (32, 32)},
        cfg=None,
        with_nms=False,
    )

    assert angle_coder.calls == 1
    assert bbox_coder.calls == 1
    for actual, expected in zip(
            (cls_score, bbox_pred, angle_pred, priors), source_tensors):
        assert torch.equal(actual, expected)
    torch.testing.assert_close(results.bboxes.tensor,
                               torch.tensor([[1., 2., 3., 4., .1]]))
    torch.testing.assert_close(results.scores, torch.sigmoid(torch.tensor([0.])))
    assert torch.equal(results.labels, torch.tensor([0]))


def _orbit_p0_predict_inputs():
    return (
        [torch.tensor([[[[0.]], [[1.]]]], dtype=torch.float32)],
        [torch.tensor([[[[1.]], [[2.]], [[3.]], [[4.]]]], dtype=torch.float32)],
        [torch.tensor([[[[.1]]]], dtype=torch.float32)],
    )


def _configure_orbit_p0_prediction_stub(head):
    head.prior_generator = SimpleNamespace(
        grid_priors=lambda *args, **kwargs: [
            torch.tensor([[10., 20.]], dtype=torch.float32)
        ])


def test_predict_by_feat_brackets_live_recording_before_filter_and_after_calibration(
        monkeypatch):
    head = _bare_ovd_orbit_p0_head()
    events = []

    class LifecycleSink:
        def __init__(self):
            self.active = False

        def begin_image(self, *, scene_id, view_id):
            assert not self.active
            self.active = True
            events.append(('begin', scene_id, view_id))

        def record_level(self, *, level, boxes, scores, calibrated_scores=None):
            assert self.active
            events.append((
                'record',
                level,
                boxes.clone(),
                scores.clone(),
                None if calibrated_scores is None else calibrated_scores.clone(),
            ))

        def record_level_with_calibrated_scores(
                self, *, level, boxes, scores, calibrated_scores):
            self.record_level(
                level=level,
                boxes=boxes,
                scores=scores,
                calibrated_scores=calibrated_scores)

        def end_image(self):
            assert self.active
            self.active = False
            events.append(('end',))

        def abort_image(self):
            assert self.active
            self.active = False
            events.append(('abort',))

    sink = LifecycleSink()
    head.set_ovd_orbit_p0_sink(sink)
    head.use_sigmoid_cls = True
    head.gaussian_semantic_scale_enable = True
    head.gaussian_semantic_scale_preserve_s3c_guard = False
    head._apply_gaussian_semantic_scale = (
        lambda *, scores, **kwargs: scores * .5)
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    _configure_orbit_p0_prediction_stub(head)
    head._bbox_post_process = lambda *, results, **kwargs: results

    def fake_filter(scores, score_thr, nms_pre, results):
        assert [event[0] for event in events] == ['begin', 'record']
        torch.testing.assert_close(
            events[-1][3], torch.tensor([[0., 1.]]))
        torch.testing.assert_close(
            events[-1][4], torch.sigmoid(torch.tensor([[0., 1.]])) * .5)
        torch.testing.assert_close(
            scores, torch.sigmoid(torch.tensor([[0., 1.]])) * .5)
        return (
            scores[:1, 0],
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            {key: value[:1] for key, value in results.items()},
        )

    monkeypatch.setattr(
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1.filter_scores_and_topk',
        fake_filter)
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    head.predict_by_feat(
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        angle_preds=angle_preds,
        batch_img_metas=[{
            'img_shape': (32, 32),
            'ovd_orbit_p0_scene_id': 'scene_a',
            'ovd_orbit_p0_view_id': 'rot090',
        }],
        cfg={'score_thr': 0, 'nms_pre': -1},
        with_nms=False,
    )

    assert [event[0] for event in events] == ['begin', 'record', 'end']
    assert not sink.active


def test_softmax_head_exports_foreground_raw_logits_aligned_to_calibrated_scores(
        monkeypatch):
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()
    head.set_ovd_orbit_p0_sink(sink)
    head.use_sigmoid_cls = False
    head.gaussian_semantic_scale_enable = False
    head.scale_semantic_calibration_enable = False
    head.test_cfg = {'score_thr': 0, 'nms_pre': -1}
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    head._bbox_post_process = lambda *, results, **kwargs: results
    cls_score = torch.tensor([[[1.]], [[2.]], [[5.]]], dtype=torch.float32)
    bbox_pred = torch.tensor(
        [[[1.]], [[2.]], [[3.]], [[4.]]], dtype=torch.float32)
    angle_pred = torch.tensor([[[.1]]], dtype=torch.float32)
    priors = torch.tensor([[10., 20.]], dtype=torch.float32)

    def fake_filter(scores, score_thr, nms_pre, results):
        return (
            scores[:1, 0],
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            {key: value[:1] for key, value in results.items()},
        )

    monkeypatch.setattr(
        'M_AD.models.dense_heads.Flex_Rrtmdet_head_v3_1.filter_scores_and_topk',
        fake_filter)
    sink.begin_image(scene_id='scene_a', view_id='rot000')
    head._predict_by_feat_single(
        cls_score_list=[cls_score],
        bbox_pred_list=[bbox_pred],
        angle_pred_list=[angle_pred],
        score_factor_list=[None],
        mlvl_priors=[priors],
        img_meta={'img_shape': (32, 32)},
        cfg=None,
        with_nms=False,
    )
    sink.end_image()

    record = sink.snapshot()[0]
    np.testing.assert_allclose(record.scores, [1., 2.])
    np.testing.assert_allclose(
        record.calibrated_scores,
        torch.softmax(torch.tensor([1., 2., 5.]), dim=0)[:2].numpy())
    assert record.scores.shape == record.calibrated_scores.shape == (2,)


def test_predict_by_feat_aborts_when_begin_image_raises_after_activation():
    events = []

    class RaisingAfterBeginSink(OvdOrbitFullLogitSink):
        def begin_image(self, *, scene_id, view_id):
            events.append('begin')
            super().begin_image(scene_id=scene_id, view_id=view_id)
            raise RuntimeError('post-begin failure')

        def abort_image(self):
            events.append('abort')
            super().abort_image()

    head = _bare_ovd_orbit_p0_head()
    sink = RaisingAfterBeginSink()
    head.set_ovd_orbit_p0_sink(sink)
    _configure_orbit_p0_prediction_stub(head)
    head._predict_by_feat_single = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError('prediction must not start after begin failure'))
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    with pytest.raises(RuntimeError, match='post-begin failure'):
        head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_img_metas=[{
                'ovd_orbit_p0_scene_id': 'scene_a',
                'ovd_orbit_p0_view_id': 'rot000',
            }],
            cfg={'score_thr': 0, 'nms_pre': -1},
        )

    assert events == ['begin', 'abort']
    assert sink.snapshot() == ()


def test_predict_by_feat_preserves_begin_error_when_abort_also_raises():
    events = []

    class FailingBeginRecoverySink:
        def begin_image(self, *, scene_id, view_id):
            events.append('begin')
            raise RuntimeError('original begin failure')

        def record_level(self, *, level, boxes, scores, calibrated_scores=None):
            pass

        def end_image(self):
            pass

        def abort_image(self):
            events.append('abort')
            raise ValueError('secondary abort failure')

    head = _bare_ovd_orbit_p0_head()
    head.set_ovd_orbit_p0_sink(FailingBeginRecoverySink())
    _configure_orbit_p0_prediction_stub(head)
    head._predict_by_feat_single = lambda **kwargs: 'single-result'
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    with pytest.raises(RuntimeError, match='original begin failure'):
        head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_img_metas=[{
                'ovd_orbit_p0_scene_id': 'scene_a',
                'ovd_orbit_p0_view_id': 'rot000',
            }],
            cfg={'score_thr': 0, 'nms_pre': -1},
        )

    assert events == ['begin', 'abort']


def test_predict_by_feat_aborts_partial_sink_records_when_single_image_prediction_raises():
    head = _bare_ovd_orbit_p0_head()
    sink = OvdOrbitFullLogitSink()
    head.set_ovd_orbit_p0_sink(sink)
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    _configure_orbit_p0_prediction_stub(head)

    def record_then_raise(**kwargs):
        head._record_ovd_orbit_p0_level(
            level_idx=0,
            raw_scores=torch.tensor([[-2., 2.]], dtype=torch.float32),
            calibrated_scores=torch.tensor([[.2, .8]], dtype=torch.float32),
            bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
            angle_pred=torch.tensor([[.1]], dtype=torch.float32),
            priors=torch.tensor([[10., 20.]], dtype=torch.float32),
            img_shape=(32, 32),
        )
        raise RuntimeError('stub single-image failure')

    head._predict_by_feat_single = record_then_raise
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    with pytest.raises(RuntimeError, match='stub single-image failure'):
        head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_img_metas=[{
                'ovd_orbit_p0_scene_id': 'scene_a',
                'ovd_orbit_p0_view_id': 'rot000',
            }],
            cfg={'score_thr': 0, 'nms_pre': -1},
        )

    assert sink.snapshot() == ()


def test_predict_by_feat_recovers_records_when_end_commits_then_raises():
    events = []

    class RaisingAfterCommitSink(OvdOrbitFullLogitSink):
        def begin_image(self, *, scene_id, view_id):
            events.append('begin')
            super().begin_image(scene_id=scene_id, view_id=view_id)

        def record_level(
                self, *, level, boxes, scores, calibrated_scores=None):
            events.append('record')
            super().record_level(
                level=level,
                boxes=boxes,
                scores=scores,
                calibrated_scores=calibrated_scores)

        def end_image(self):
            events.append('end')
            super().end_image()
            raise RuntimeError('post-commit end failure')

        def abort_image(self):
            events.append('abort')
            super().abort_image()

    head = _bare_ovd_orbit_p0_head()
    sink = RaisingAfterCommitSink()
    head.set_ovd_orbit_p0_sink(sink)
    head.angle_coder = _FakeOrbitP0AngleCoder()
    head.bbox_coder = _FakeOrbitP0BBoxCoder()
    _configure_orbit_p0_prediction_stub(head)

    def record_then_return(**kwargs):
        head._record_ovd_orbit_p0_level(
            level_idx=0,
            raw_scores=torch.tensor([[-2., 2.]], dtype=torch.float32),
            calibrated_scores=torch.tensor([[.2, .8]], dtype=torch.float32),
            bbox_pred=torch.tensor([[1., 2., 3., 4.]], dtype=torch.float32),
            angle_pred=torch.tensor([[.1]], dtype=torch.float32),
            priors=torch.tensor([[10., 20.]], dtype=torch.float32),
            img_shape=(32, 32),
        )
        return 'single-result'

    head._predict_by_feat_single = record_then_return
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    with pytest.raises(RuntimeError, match='post-commit end failure'):
        head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_img_metas=[{
                'ovd_orbit_p0_scene_id': 'scene_a',
                'ovd_orbit_p0_view_id': 'rot000',
            }],
            cfg={'score_thr': 0, 'nms_pre': -1},
        )

    assert events == ['begin', 'record', 'end', 'abort']
    assert sink.snapshot() == ()


def test_predict_by_feat_preserves_end_error_when_abort_also_raises():
    events = []

    class FailingRecoverySink:
        def begin_image(self, *, scene_id, view_id):
            events.append('begin')

        def record_level(self, *, level, boxes, scores):
            pass

        def end_image(self):
            events.append('end')
            raise RuntimeError('original end failure')

        def abort_image(self):
            events.append('abort')
            raise ValueError('secondary abort failure')

    head = _bare_ovd_orbit_p0_head()
    head.set_ovd_orbit_p0_sink(FailingRecoverySink())
    _configure_orbit_p0_prediction_stub(head)
    head._predict_by_feat_single = lambda **kwargs: 'single-result'
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    with pytest.raises(RuntimeError, match='original end failure'):
        head.predict_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_img_metas=[{
                'ovd_orbit_p0_scene_id': 'scene_a',
                'ovd_orbit_p0_view_id': 'rot000',
            }],
            cfg={'score_thr': 0, 'nms_pre': -1},
        )

    assert events == ['begin', 'end', 'abort']


def test_predict_by_feat_with_no_sink_does_not_request_orbit_metadata():
    class OpaqueMetadata:
        def __getitem__(self, key):
            raise AssertionError('none-sink lifecycle must not read metadata')

        def get(self, key, default=None):
            raise AssertionError('none-sink lifecycle must not read metadata')

    head = _bare_ovd_orbit_p0_head()
    head.set_ovd_orbit_p0_sink(None)
    _configure_orbit_p0_prediction_stub(head)
    head._ovd_orbit_p0_metadata = lambda img_meta: (_ for _ in ()).throw(
        AssertionError('none-sink lifecycle must not validate metadata'))
    head._predict_by_feat_single = lambda **kwargs: 'ordinary-result'
    cls_scores, bbox_preds, angle_preds = _orbit_p0_predict_inputs()

    results = head.predict_by_feat(
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        angle_preds=angle_preds,
        batch_img_metas=[OpaqueMetadata()],
        cfg={'score_thr': 0, 'nms_pre': -1},
    )

    assert results == ['ordinary-result']


def carrier(*, view='rot000', scene='scene_a', level=0, row=3,
            scores=(.2, .7, .1), calibrated_scores=None,
            box=(10., 20., 8., 4., .1)):
    return ScoreCarrier(
        view_id=view,
        scene_id=scene,
        source=(level, row),
        box=np.array(box, dtype=np.float32),
        scores=np.array(scores, dtype=np.float32),
        calibrated_scores=(
            None if calibrated_scores is None
            else np.array(calibrated_scores, dtype=np.float32)),
    )


def test_score_carrier_rejects_disallowed_view():
    with pytest.raises(OrbitP0Error, match='P0'):
        carrier(view='rot030')


def test_collapse_preserves_full_score_vector_and_rejects_conflict():
    np.testing.assert_allclose(
        collapse_carriers([carrier(), carrier()])[0].scores, [.2, .7, .1])
    with pytest.raises(OrbitP0Error, match='conflicting') as error:
        collapse_carriers([carrier(), carrier(scores=(.3, .6, .1))])

    assert "('scene_a', 'rot000', (0, 3))" in str(error.value)
    assert 'differing fields: scores' in str(error.value)


def test_collapse_reports_box_only_duplicate_conflict_identity_and_field():
    with pytest.raises(OrbitP0Error, match='conflicting') as error:
        collapse_carriers([
            carrier(),
            carrier(box=(11., 20., 8., 4., .1)),
        ])

    assert "('scene_a', 'rot000', (0, 3))" in str(error.value)
    assert 'differing fields: box' in str(error.value)


def test_collapse_rejects_conflicting_optional_calibrated_scores():
    calibrated = (.2, .7, .1)
    assert len(collapse_carriers([
        carrier(calibrated_scores=calibrated),
        carrier(calibrated_scores=calibrated),
    ])) == 1

    with pytest.raises(OrbitP0Error, match='conflicting') as error:
        collapse_carriers([
            carrier(calibrated_scores=calibrated),
            carrier(calibrated_scores=(.3, .6, .1)),
        ])

    assert "('scene_a', 'rot000', (0, 3))" in str(error.value)
    assert 'differing fields: calibrated_scores' in str(error.value)


def test_manifest_rejects_p0148_scene_and_mixed_vocabulary_hashes():
    with pytest.raises(OrbitP0Error, match='P0148'):
        build_manifest(
            [carrier(scene='P0148__1024__651___0')],
            vocabulary_hash='a',
            prompt_hash='p',
        )
    with pytest.raises(OrbitP0Error, match='vocabulary'):
        build_manifest(
            [carrier()],
            vocabulary_hash='a',
            prompt_hash='p',
            observed_vocabulary_hashes={'a', 'b'},
        )


def test_normalize_level_outputs_preserves_raw_and_optional_calibrated_vectors():
    boxes = torch.tensor([[1., 2., 3., 4., .1], [5., 6., 7., 8., .2]])
    scores = torch.tensor([[-2., .3, 5.], [.7, -1.2, .1]])
    calibrated_scores = torch.tensor([[.2, .3, .5], [.7, .2, .1]])
    source_scores = scores.clone()
    source_calibrated_scores = calibrated_scores.clone()

    records = normalize_level_outputs(
        view_id='rot090',
        scene_id='scene_a',
        level=2,
        boxes=boxes,
        scores=scores,
        calibrated_scores=calibrated_scores)

    assert [record.source for record in records] == [(2, 0), (2, 1)]
    np.testing.assert_allclose(records[1].scores, [.7, -1.2, .1])
    np.testing.assert_allclose(records[1].calibrated_scores, [.7, .2, .1])
    assert not records[1].calibrated_scores.flags.writeable
    assert not np.shares_memory(records[1].calibrated_scores,
                                calibrated_scores.numpy())
    assert torch.equal(scores, source_scores)
    assert torch.equal(calibrated_scores, source_calibrated_scores)


def test_normalize_level_outputs_rejects_invalid_score_matrix():
    with pytest.raises(OrbitP0Error, match='scores'):
        normalize_level_outputs(
            view_id='rot090',
            scene_id='scene_a',
            level=2,
            boxes=torch.ones(1, 5),
            scores=torch.ones(3),
        )


def test_full_logit_sink_rejects_record_without_active_image():
    with pytest.raises(OrbitP0Error, match='active'):
        OvdOrbitFullLogitSink().record_level(
            level=2, boxes=torch.ones(1, 5), scores=torch.ones(1, 3))


def test_full_logit_sink_abort_discards_only_active_image_records():
    sink = OvdOrbitFullLogitSink()
    boxes = torch.tensor([[1., 2., 3., 4., .1]])
    scores = torch.tensor([[.2, .3, .5]])

    sink.begin_image(scene_id='committed', view_id='rot000')
    sink.record_level(level=0, boxes=boxes, scores=scores)
    sink.end_image()
    sink.begin_image(scene_id='partial', view_id='rot090')
    sink.record_level(level=1, boxes=boxes, scores=scores)
    sink.abort_image()

    records = sink.snapshot()
    assert [(record.scene_id, record.source) for record in records] == [
        ('committed', (0, 0))]
    with pytest.raises(OrbitP0Error, match='active'):
        sink.abort_image()


def test_full_logit_sink_abort_can_recover_a_just_committed_image():
    sink = OvdOrbitFullLogitSink()
    sink.begin_image(scene_id='scene_a', view_id='rot000')
    sink.record_level(
        level=0,
        boxes=torch.tensor([[1., 2., 3., 4., .1]]),
        scores=torch.tensor([[.2, .3, .5]]),
    )
    sink.end_image()

    sink.abort_image()

    assert sink.snapshot() == ()


def test_full_logit_sink_requires_keyword_only_public_arguments():
    sink = OvdOrbitFullLogitSink()

    with pytest.raises(TypeError):
        sink.begin_image('scene_a', 'rot090')

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    with pytest.raises(TypeError):
        sink.record_level(2, torch.ones(1, 5), torch.ones(1, 3))
    sink.end_image()


def test_full_logit_sink_preserves_full_scores_with_active_context():
    sink = OvdOrbitFullLogitSink()

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    sink.record_level(
        level=2,
        boxes=torch.tensor([[1., 2., 3., 4., .1]]),
        scores=torch.tensor([[.2, .3, .5]]),
    )
    sink.end_image()

    records = sink.snapshot()
    assert [record.source for record in records] == [(2, 0)]
    np.testing.assert_allclose(records[0].scores, [.2, .3, .5])


def test_full_logit_sink_preserves_optional_calibrated_scores():
    sink = OvdOrbitFullLogitSink()
    raw_scores = torch.tensor([[-2., 2.]], dtype=torch.float32)
    calibrated_scores = torch.tensor([[.2, .8]], dtype=torch.float32)

    sink.begin_image(scene_id='scene_a', view_id='rot000')
    sink.record_level(
        level=2,
        boxes=torch.tensor([[1., 2., 3., 4., .1]]),
        scores=raw_scores,
        calibrated_scores=calibrated_scores,
    )
    sink.end_image()

    record = sink.snapshot()[0]
    np.testing.assert_allclose(record.scores, [-2., 2.])
    np.testing.assert_allclose(record.calibrated_scores, [.2, .8])
    assert not record.calibrated_scores.flags.writeable
    with pytest.raises(OrbitP0Error, match='active'):
        sink.abort_image()


def test_full_logit_sink_snapshot_owns_immutable_detached_arrays():
    sink = OvdOrbitFullLogitSink()
    boxes = torch.tensor([[1., 2., 3., 4., .1]])
    scores = torch.tensor([[.2, .3, .5]])
    expected_box = boxes[0].clone()
    expected_scores = scores[0].clone()

    sink.begin_image(scene_id='scene_a', view_id='rot000')
    sink.record_level(level=2, boxes=boxes, scores=scores)
    sink.end_image()

    record = sink.snapshot()[0]
    assert not record.box.flags.writeable
    assert not record.scores.flags.writeable
    assert not np.shares_memory(record.box, boxes.numpy())
    assert not np.shares_memory(record.scores, scores.numpy())
    with pytest.raises(ValueError):
        record.box[0] = 99.
    with pytest.raises(ValueError):
        record.scores[0] = 99.

    boxes[0, 0] = 99.
    scores[0, 0] = .9

    torch.testing.assert_close(boxes[0, 0], torch.tensor(99.))
    torch.testing.assert_close(scores[0, 0], torch.tensor(.9))
    np.testing.assert_allclose(record.box, expected_box.numpy())
    np.testing.assert_allclose(record.scores, expected_scores.numpy())


def test_full_logit_sink_fails_closed_for_invalid_lifecycle_or_metadata():
    sink = OvdOrbitFullLogitSink()

    with pytest.raises(OrbitP0Error):
        sink.begin_image(scene_id='', view_id='rot090')
    with pytest.raises(OrbitP0Error):
        sink.begin_image(scene_id='scene_a', view_id='rot030')

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    with pytest.raises(OrbitP0Error):
        sink.begin_image(scene_id='scene_b', view_id='rot180')
    with pytest.raises(OrbitP0Error):
        sink.snapshot()
    sink.end_image()

    with pytest.raises(OrbitP0Error):
        sink.end_image()
    sink.abort_image()
    with pytest.raises(OrbitP0Error):
        sink.abort_image()


def test_full_logit_sink_snapshot_collapses_duplicate_level_records():
    sink = OvdOrbitFullLogitSink()
    boxes = torch.tensor([[1., 2., 3., 4., .1]])
    scores = torch.tensor([[.2, .3, .5]])

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    sink.record_level(level=2, boxes=boxes, scores=scores)
    sink.record_level(level=2, boxes=boxes, scores=scores)
    sink.end_image()

    records = sink.snapshot()
    assert len(records) == 1
    assert records[0].source == (2, 0)


def test_full_logit_sink_snapshot_rejects_conflicting_duplicate_records():
    sink = OvdOrbitFullLogitSink()
    boxes = torch.tensor([[1., 2., 3., 4., .1]])

    sink.begin_image(scene_id='scene_a', view_id='rot090')
    sink.record_level(
        level=2, boxes=boxes, scores=torch.tensor([[.2, .3, .5]]))
    sink.record_level(
        level=2, boxes=boxes, scores=torch.tensor([[.3, .2, .5]]))
    sink.end_image()

    with pytest.raises(OrbitP0Error, match='conflicting') as error:
        sink.snapshot()

    assert "('scene_a', 'rot090', (2, 0))" in str(error.value)
    assert 'differing fields: scores' in str(error.value)


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


def test_legacy_jsonl_without_calibrated_scores_roundtrips(tmp_path):
    legacy_jsonl = tmp_path / 'legacy.jsonl'
    legacy_jsonl.write_text(json.dumps({
        'box': [1., 2., 3., 4., .1],
        'scene_id': 'scene_a',
        'scores': [-2., 2.],
        'source': [0, 0],
        'view_id': 'rot000',
    }) + '\n', encoding='utf-8')

    carriers = load_carriers(legacy_jsonl)
    assert carriers[0].calibrated_scores is None
    manifest = build_manifest(
        carriers, vocabulary_hash='vocabulary-hash', prompt_hash='prompt-hash')
    write_receipt(tmp_path / 'receipt', manifest, carriers)
    round_tripped = load_carriers(tmp_path / 'receipt' / 'carriers.jsonl')

    assert round_tripped[0].calibrated_scores is None
    np.testing.assert_allclose(round_tripped[0].scores, [-2., 2.])


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
