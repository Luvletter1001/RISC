import torch
from types import SimpleNamespace
from torch import nn

from projects.OVCapFlow.ov_capflow.ov_capflow import OVCapFlow
from projects.OVCapFlow.ov_capflow.query_initializer import (
    FixedRotatedQueryInitializer, )


def _build_pre_decoder_only_model() -> OVCapFlow:
    model = OVCapFlow.__new__(OVCapFlow)
    nn.Module.__init__(model)
    model.num_queries = 4
    model.query_init_mode = 'fixed'
    model.query_initializer = FixedRotatedQueryInitializer(
        num_queries=4, embed_dims=8)
    model.eval()
    return model


class _StaticClassBranch(nn.Module):

    max_text_len = 2

    def __init__(self, scores):
        super().__init__()
        self.register_buffer('scores', torch.as_tensor(scores).float())

    def forward(self, memory, memory_text, text_token_mask):
        del memory_text, text_token_mask
        return self.scores.unsqueeze(0).expand(memory.shape[0], -1, -1)


class _ZeroRegBranch(nn.Module):

    def forward(self, memory):
        return memory.new_zeros(memory.shape[0], memory.shape[1], 5)


def _build_encoder_proposal_model(training=False) -> OVCapFlow:
    model = OVCapFlow.__new__(OVCapFlow)
    nn.Module.__init__(model)
    model.num_queries = 2
    model.train_query_groups = 1
    model.query_init_mode = 'encoder_proposal'
    model.query_initializer = FixedRotatedQueryInitializer(
        num_queries=2, embed_dims=8)
    model.__dict__['decoder'] = SimpleNamespace(num_layers=1)
    model.__dict__['bbox_head'] = SimpleNamespace(
        cls_branches=[None, _StaticClassBranch([
            [0.1, 0.2],
            [2.0, 0.1],
            [0.3, 0.1],
            [0.4, 0.1],
            [0.2, 3.0],
            [0.5, 0.1],
        ])],
        reg_branches=[None, _ZeroRegBranch()])
    proposal_logits = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)

    def proposals(memory, memory_mask, spatial_shapes):
        del memory_mask, spatial_shapes
        return memory, proposal_logits.expand(memory.shape[0], -1, -1)

    model.gen_encoder_output_proposals = proposals
    if training:
        model.train()

        def dn_generator(batch_data_samples):
            batch_size = len(batch_data_samples)
            return (
                torch.zeros(batch_size, 1, 8),
                torch.zeros(batch_size, 1, 5),
                torch.zeros(3, 3, dtype=torch.bool),
                {'num_denoising_queries': 1},
            )

        model.dn_query_generator = dn_generator
    else:
        model.eval()
    return model


def test_pre_decoder_uses_fixed_queries_without_encoder_proposals():
    model = _build_pre_decoder_only_model()
    memory = torch.randn(2, 10, 8)
    memory_text = torch.randn(2, 3, 8)
    text_token_mask = torch.ones(2, 3, dtype=torch.bool)

    decoder_inputs, head_inputs = model.pre_decoder(
        memory=memory,
        memory_mask=torch.zeros(2, 10, dtype=torch.bool),
        spatial_shapes=torch.tensor([[2, 5]]),
        memory_text=memory_text,
        text_token_mask=text_token_mask,
        batch_data_samples=None)

    assert decoder_inputs['query'].shape == (2, 4, 8)
    assert decoder_inputs['reference_points'].shape == (2, 4, 5)
    assert decoder_inputs['dn_mask'] is None
    assert 'enc_outputs_class' not in head_inputs
    assert 'enc_outputs_coord' not in head_inputs
    assert head_inputs['memory_text'] is memory_text
    assert head_inputs['text_token_mask'] is text_token_mask


def test_pre_decoder_query_initialization_does_not_change_with_memory():
    model = _build_pre_decoder_only_model()
    common = dict(
        memory_mask=torch.zeros(1, 6, dtype=torch.bool),
        spatial_shapes=torch.tensor([[2, 3]]),
        memory_text=torch.randn(1, 2, 8),
        text_token_mask=torch.ones(1, 2, dtype=torch.bool),
        batch_data_samples=None)

    decoder_a, _ = model.pre_decoder(memory=torch.zeros(1, 6, 8), **common)
    decoder_b, _ = model.pre_decoder(memory=torch.randn(1, 6, 8), **common)

    torch.testing.assert_close(decoder_a['query'], decoder_b['query'])
    torch.testing.assert_close(decoder_a['reference_points'],
                               decoder_b['reference_points'])


def test_encoder_proposal_mode_selects_references_from_encoder_scores():
    model = _build_encoder_proposal_model()
    memory = torch.randn(1, 6, 8)
    decoder_inputs, head_inputs = model.pre_decoder(
        memory=memory,
        memory_mask=torch.zeros(1, 6, dtype=torch.bool),
        spatial_shapes=torch.tensor([[2, 3]]),
        memory_text=torch.randn(1, 3, 8),
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        batch_data_samples=None)

    expected_logits = torch.arange(30, dtype=torch.float32).reshape(
        1, 6, 5)[:, [4, 1]]
    torch.testing.assert_close(
        decoder_inputs['reference_points'], expected_logits.sigmoid())
    assert decoder_inputs['query'].shape == (1, 2, 8)
    assert 'enc_outputs_class' not in head_inputs
    assert 'enc_outputs_coord' not in head_inputs


def test_encoder_proposal_mode_exposes_topk_encoder_targets_in_training():
    model = _build_encoder_proposal_model(training=True)
    decoder_inputs, head_inputs = model.pre_decoder(
        memory=torch.randn(1, 6, 8),
        memory_mask=torch.zeros(1, 6, dtype=torch.bool),
        spatial_shapes=torch.tensor([[2, 3]]),
        memory_text=torch.randn(1, 3, 8),
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        batch_data_samples=[SimpleNamespace()])

    assert decoder_inputs['query'].shape == (1, 3, 8)
    assert decoder_inputs['reference_points'].shape == (1, 3, 5)
    assert head_inputs['enc_outputs_class'].shape == (1, 2, 2)
    assert head_inputs['enc_outputs_coord'].shape == (1, 2, 5)
    assert head_inputs['dn_meta']['num_matching_query_groups'] == 1
    assert head_inputs['dn_meta']['num_matching_queries_per_group'] == 2


def test_query_embedding_alias_is_not_registered_twice():
    model = _build_pre_decoder_only_model()
    assert model.query_embedding is model.query_initializer.query_embedding
    assert 'query_embedding' not in model._modules


def test_forward_decoder_exports_only_available_calibration_state():
    model = OVCapFlow.__new__(OVCapFlow)
    nn.Module.__init__(model)
    model.__dict__['decoder'] = SimpleNamespace(
        last_null_logits=torch.randn(2, 4),
        layers=[SimpleNamespace(
            last_capacity=None,
            last_predicted_count=None,
            last_semantic_gate=torch.randn(2, 4, 8))])
    output = model._append_calibration_state({'hidden_states': torch.empty(0)})
    assert output['null_logits'].shape == (2, 4)
    assert 'capacity' not in output
    assert output['semantic_gate'].shape == (2, 4, 8)
