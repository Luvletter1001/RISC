from mmengine import Config


ALL_CLASSES = (
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank',
    'swimming-pool', 'tennis-court')
NOVEL = ('airport', 'container-crane', 'helipad', 'helicopter')


def test_prompt_class_modes_preserve_canonical_order():
    from projects.OVCapFlow.tools.eval_prompt_subset import (
        select_prompt_classes, )

    assert select_prompt_classes(ALL_CLASSES, NOVEL, 'all') == ALL_CLASSES
    assert select_prompt_classes(ALL_CLASSES, NOVEL, 'novel') == NOVEL
    assert select_prompt_classes(ALL_CLASSES, NOVEL, 'base') == tuple(
        name for name in ALL_CLASSES if name not in NOVEL)


def test_configure_prompt_subset_preserves_full_annotation_parser():
    from projects.OVCapFlow.tools.eval_prompt_subset import (
        configure_prompt_subset, )

    cfg = Config(dict(
        train_dataloader=dict(dataset=dict(
            metainfo=dict(classes=ALL_CLASSES))),
        val_dataloader=dict(dataset=dict(
            metainfo=dict(classes=ALL_CLASSES))),
        test_dataloader=dict(
            num_workers=4,
            persistent_workers=True,
            dataset=dict(
                metainfo=dict(classes=ALL_CLASSES),
                filter_cfg=dict(filter_empty_gt=False),
                test_mode=True)),
        test_evaluator=dict(type='DOTAMetric')))
    base = tuple(name for name in ALL_CLASSES if name not in NOVEL)
    configure_prompt_subset(cfg, base)

    assert tuple(cfg.test_dataloader.dataset.metainfo.classes) == ALL_CLASSES
    assert cfg.test_dataloader.num_workers == 0
    assert cfg.test_dataloader.persistent_workers is False
    assert cfg.test_dataloader.dataset.filter_cfg.filter_empty_gt is False
    assert cfg.test_dataloader.dataset.test_mode is True
    assert tuple(cfg.train_dataloader.dataset.metainfo.classes) == ALL_CLASSES
    assert tuple(cfg.val_dataloader.dataset.metainfo.classes) == ALL_CLASSES


def test_subset_ground_truth_is_filtered_and_remapped():
    import torch
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    from projects.OVCapFlow.tools.eval_prompt_subset import (
        metric_sample_dicts, remap_sample_ground_truth, )

    sample = DetDataSample(metainfo=dict(img_id='x'))
    sample.gt_instances = InstanceData(
        labels=torch.tensor([0, 1, 4, 7, 8]),
        bboxes=torch.arange(25, dtype=torch.float32).reshape(5, 5))
    sample.ignored_instances = InstanceData(
        labels=torch.tensor([1, 8]),
        bboxes=torch.arange(10, dtype=torch.float32).reshape(2, 5))
    remapped = remap_sample_ground_truth(sample, ALL_CLASSES, NOVEL)

    assert remapped.gt_instances.labels.tolist() == [0, 1, 3, 2]
    assert remapped.gt_instances.bboxes[:, 0].tolist() == [0, 10, 15, 20]
    assert remapped.ignored_instances.labels.tolist() == [2]
    assert sample.gt_instances.labels.tolist() == [0, 1, 4, 7, 8]
    metric_samples = metric_sample_dicts([remapped])
    assert isinstance(metric_samples[0], dict)
    assert metric_samples[0]['gt_instances']['labels'].tolist() == [0, 1, 3, 2]
