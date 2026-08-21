import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from M_Tools.Data1_DOTA2.Step5_3_Prepare_Visual_Text_DeCLIP_support import (
    contains_needed_class,
    crop_axis_aligned_from_poly,
    encode_crop_batch,
    iter_limited_paths,
    l2_normalize_rows,
    select_topk_rows,
    validate_support_schema,
)


class DeclipSupportBuilderTest(unittest.TestCase):

    def test_l2_normalize_rows_keeps_zero_rows_finite(self):
        rows = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

        normalized = l2_normalize_rows(rows)

        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertTrue(np.allclose(normalized[0], [0.6, 0.8]))
        self.assertTrue(np.allclose(normalized[1], [0.0, 0.0]))

    def test_crop_axis_aligned_from_poly_clamps_to_image_bounds(self):
        image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        poly = np.asarray([-3, -2, 4, -2, 4, 3, -3, 3], dtype=np.float32)

        crop = crop_axis_aligned_from_poly(image, poly, pad=1)

        self.assertEqual(crop.shape, (5, 6, 3))
        self.assertEqual(int(crop[0, 0, 0]), int(image[0, 0, 0]))

    def test_select_topk_rows_orders_by_confidence_and_repeats_short_classes(self):
        rows = np.asarray([[1, 1], [2, 2]], dtype=np.float32)
        scores = np.asarray([0.2, 0.9], dtype=np.float32)

        selected, selected_scores = select_topk_rows(rows, scores, topk=5)

        self.assertEqual(selected.shape, (5, 2))
        self.assertEqual(selected_scores.shape, (5,))
        self.assertTrue(np.allclose(selected[0], [2, 2]))
        self.assertTrue(np.allclose(selected[1], [1, 1]))
        self.assertTrue(np.allclose(selected[2], [2, 2]))

    def test_validate_support_schema_rejects_missing_required_key(self):
        support = {
            'small-vehicle': {
                'texts': ['a small vehicle in an aerial image'],
                'text_embeds': np.zeros((1, 768), dtype=np.float32),
            }
        }

        with self.assertRaisesRegex(KeyError, 'visual_embeds'):
            validate_support_schema(support)

    def test_iter_limited_paths_applies_max_files_after_sorting(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ('b.txt', 'a.txt', 'c.txt'):
                (root / name).write_text('', encoding='utf-8')

            paths = iter_limited_paths(root, '*.txt', max_files=2)

            self.assertEqual([path.name for path in paths], ['a.txt', 'b.txt'])

    def test_encode_crop_batch_can_use_global_image_features(self):
        class DummyModel:
            def __init__(self):
                self.dense_calls = 0
                self.image_calls = 0

            def encode_dense(self, *args, **kwargs):
                self.dense_calls += 1
                raise AssertionError('dense path should not be used')

            def encode_image(self, batch):
                self.image_calls += 1
                return torch.ones((batch.shape[0], 4), device=batch.device)

        model = DummyModel()
        crops = [np.ones((8, 8, 3), dtype=np.uint8) * 127]

        feats = encode_crop_batch(
            model,
            crops,
            device='cpu',
            image_size=16,
            mode='csa',
            image_encode='global',
        )

        self.assertEqual(model.image_calls, 1)
        self.assertEqual(model.dense_calls, 0)
        self.assertEqual(feats.shape, (1, 4))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_contains_needed_class_skips_filled_classes(self):
        items = [
            ('ship', np.zeros(8, dtype=np.float32)),
            ('plane', np.zeros(8, dtype=np.float32)),
        ]
        counts = {'ship': 5, 'plane': 0}

        self.assertTrue(
            contains_needed_class(items, counts, max_candidates_per_class=5))

        counts['plane'] = 5
        self.assertFalse(
            contains_needed_class(items, counts, max_candidates_per_class=5))


if __name__ == '__main__':
    unittest.main()
