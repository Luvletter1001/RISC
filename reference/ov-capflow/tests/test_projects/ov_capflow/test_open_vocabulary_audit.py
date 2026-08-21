import torch
import torch.nn as nn


class _TinyOpenVocabularyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_projection = nn.Linear(4, 4)

    def get_tokens_positive_and_prompts(self, entities,
                                        custom_entities=True,
                                        enhanced_text_prompts=None):
        assert custom_entities
        prompt = '. '.join(entities) + '. '
        positive_map = {}
        cursor = 0
        for label, entity in enumerate(entities, start=1):
            positive_map[label] = list(range(cursor, cursor + len(entity)))
            cursor += len(entity) + 2
        return positive_map, prompt, None, list(entities)


def test_prompt_variants_preserve_shapes_and_save_protocol():
    from projects.OVCapFlow.tools.audit_open_vocabulary import (
        audit_prompt_variants, )

    model = _TinyOpenVocabularyModel()
    variants = {
        'canonical': ['airport', 'plane'],
        'reordered': ['plane', 'airport'],
        'synonym_expanded': ['airport airfield', 'plane aircraft'],
        'one_new_class': ['airport', 'plane', 'wind-turbine'],
    }
    report = audit_prompt_variants(
        model,
        variants,
        run_variant=lambda entities: [600, 600],
        expected_queries=600)

    assert report['pass'] is True
    assert report['state_shapes_unchanged'] is True
    assert report['state_shapes_before'] == report['state_shapes_after']
    assert list(report['variants']) == list(variants)
    assert report['variants']['reordered']['entity_order'] == [
        'plane', 'airport']
    assert report['variants']['one_new_class']['actual_prompt'].endswith('. ')
    assert len(report['variants']['one_new_class']['token_spans']) == 3
    assert report['variants']['canonical']['positive_map']['1']
    assert report['variants']['canonical']['result_counts'] == [600, 600]


def test_fixed_class_classifier_and_forbidden_checkpoint_keys_are_rejected():
    from projects.OVCapFlow.tools.audit_open_vocabulary import (
        find_fixed_classification_layers, find_forbidden_checkpoint_keys, )

    class BadModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.final_classifier = nn.Linear(8, 18)
            self.geometry = nn.Linear(8, 18)

    hits = find_fixed_classification_layers(BadModel(), class_count=18)
    assert hits == ['final_classifier']
    assert find_forbidden_checkpoint_keys([
        'backbone.stage.0.weight',
        'teacher.backbone.weight',
        'bbox_head.roi_head.weight',
        'pseudo_label_generator.bias',
    ]) == [
        'teacher.backbone.weight',
        'bbox_head.roi_head.weight',
        'pseudo_label_generator.bias',
    ]


def test_parameter_shape_audit_fails_on_prompt_side_mutation():
    from projects.OVCapFlow.tools.audit_open_vocabulary import (
        audit_prompt_variants, )

    model = _TinyOpenVocabularyModel()

    def mutate(_entities):
        model.new_classifier = nn.Linear(4, 18)
        return [600]

    report = audit_prompt_variants(
        model, {'canonical': ['plane']}, mutate, expected_queries=600)
    assert report['pass'] is False
    assert report['state_shapes_unchanged'] is False


def test_prompt_replacement_uses_existing_metainfo_fields():
    from mmdet.structures import DetDataSample

    from projects.OVCapFlow.tools.audit_open_vocabulary import (
        set_prompt_metainfo, )

    sample = DetDataSample(
        metainfo=dict(text=['old'], custom_entities=True))
    set_prompt_metainfo(sample, ['plane', 'airport'])
    assert sample.text == ['plane', 'airport']
    assert sample.custom_entities is True


def test_decoder_coordinate_rescale_preserves_query_order():
    from projects.OVCapFlow.tools.audit_decoder_alignment import (
        align_decoder_predictions, export_decoder_boxes, )

    normalized = torch.tensor([
        [0.10, 0.20, 0.30, 0.40, 0.25],
        [0.80, 0.70, 0.20, 0.10, 0.75],
    ])
    meta = dict(img_shape=(100, 200), scale_factor=(2.0, 4.0))
    exported = export_decoder_boxes(
        normalized, meta, angle_factor=torch.pi, rescale=True)
    report = align_decoder_predictions(
        normalized, exported, meta, angle_factor=torch.pi, expected_queries=2)
    assert report['pass'] is True
    assert report['same_query_order'] is True
    assert report['max_abs_error'] == 0.0

    reordered = exported.flip(0)
    bad = align_decoder_predictions(
        normalized, reordered, meta, angle_factor=torch.pi,
        expected_queries=2)
    assert bad['pass'] is False
    assert bad['same_query_order'] is False
