import torch

from projects.OVCapFlow.ov_capflow.ov_capflow_layers import (
    OVCapFlowDecoder, OVCapFlowDecoderLayer,
    apply_matching_query_interventions, fuse_matching_query_suffix)
from projects.OVCapFlow.ov_capflow.semantic_capacity import (
    ContinuousDensityCapacity,
    SemanticEvidenceFusion,
    density_capacity_losses,
)


def test_zero_gate_starts_as_exact_parent_query():
    fusion = SemanticEvidenceFusion(embed_dims=8, adapter_init='identity')
    native = torch.randn(2, 4, 8)
    parent = torch.randn(2, 4, 8)

    fused, gate = fusion(native, parent)

    assert torch.equal(fused, parent)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_orthogonal_adapter_is_orthogonal_at_initialization():
    fusion = SemanticEvidenceFusion(embed_dims=8, adapter_init='orthogonal')
    weight = fusion.evidence_adapter.weight.detach()

    torch.testing.assert_close(
        weight @ weight.transpose(0, 1), torch.eye(8), atol=1e-5, rtol=1e-5)


def test_signed_gate_moves_parent_along_native_residual():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    native = torch.tensor([[[3.0, 1.0, -1.0, 2.0]]])
    parent = torch.tensor([[[1.0, 2.0, 1.0, -2.0]]])

    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    positive, positive_gate = fusion(native, parent)
    expected_positive = parent + positive_gate * (native - parent)
    torch.testing.assert_close(positive, expected_positive)

    with torch.no_grad():
        fusion.gate.bias.fill_(-1.0)
    negative, negative_gate = fusion(native, parent)
    expected_negative = parent + negative_gate * (native - parent)
    torch.testing.assert_close(negative, expected_negative)


def test_zero_capacity_preserves_parent_even_with_open_gate():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    native = torch.randn(1, 3, 4)
    parent = torch.randn(1, 3, 4)

    fused, gate = fusion(
        native, parent, capacity=torch.zeros(1, 3))

    assert torch.equal(fused, parent)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_density_capacity_is_continuous_and_bounded():
    module = ContinuousDensityCapacity(embed_dims=8)
    query = torch.randn(2, 5, 8)
    memory = torch.randn(2, 7, 8)

    capacity, predicted_count = module(query, memory)

    assert capacity.shape == (2, 5)
    assert predicted_count.shape == (2, )
    assert torch.all((capacity > 0) & (capacity < 1))
    assert torch.all((predicted_count > 0) & (predicted_count < 5))


def test_padded_memory_does_not_change_predicted_count():
    module = ContinuousDensityCapacity(embed_dims=4)
    query = torch.randn(1, 3, 4)
    memory_a = torch.randn(1, 5, 4)
    memory_b = memory_a.clone()
    memory_b[:, 2:] = 1000
    mask = torch.tensor([[False, False, True, True, True]])

    _, count_a = module(query, memory_a, memory_mask=mask)
    _, count_b = module(query, memory_b, memory_mask=mask)

    torch.testing.assert_close(count_a, count_b)


def test_density_capacity_losses_backpropagate():
    module = ContinuousDensityCapacity(embed_dims=4)
    query = torch.randn(2, 6, 4)
    memory = torch.randn(2, 5, 4)
    target_count = torch.tensor([2.0, 4.0])
    capacity, predicted_count = module(query, memory)

    losses = density_capacity_losses(
        capacity=capacity,
        predicted_count=predicted_count,
        target_count=target_count)
    sum(losses.values()).backward()

    assert set(losses) == {'loss_capacity_mass', 'loss_density_count'}
    assert module.capacity_head.weight.grad is not None
    assert module.density_head.weight.grad is not None


def test_matching_suffix_fusion_leaves_denoising_prefix_unchanged():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    native = torch.randn(1, 5, 4)
    transported = torch.randn(1, 5, 4)
    capacity = torch.ones(1, 3)

    output, gate = fuse_matching_query_suffix(
        native_query=native,
        transported_query=transported,
        capacity=capacity,
        matching_query_count=3,
        fusion=fusion)
    expected_matching, _ = fusion(
        native[:, -3:], transported[:, -3:], capacity=capacity)

    torch.testing.assert_close(output[:, :2], transported[:, :2])
    torch.testing.assert_close(output[:, -3:], expected_matching)
    assert gate.shape == (1, 3, 4)


def test_zero_initialized_matching_suffix_preserves_complete_parent():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    native = torch.randn(1, 5, 4)
    parent = torch.randn(1, 5, 4)

    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=parent,
        matching_query_count=3,
        fusion=fusion,
        capacity=None)

    assert torch.equal(output, parent)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_decoder_constructs_with_semantic_fusion_config():
    decoder = OVCapFlowDecoder(
        num_layers=1,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=8, num_heads=2, num_levels=1, dropout=0.0),
            ffn_cfg=dict(embed_dims=8, feedforward_channels=16, ffn_drop=0.0),
            semantic_fusion_cfg=dict(adapter_init='orthogonal')),
        post_norm_cfg=None)

    assert isinstance(decoder.layers[0], OVCapFlowDecoderLayer)
    assert decoder.layers[0].semantic_fusion.adapter_init == 'orthogonal'


def test_disabled_semantic_fusion_returns_transported_query():
    native = torch.randn(1, 5, 4)
    transported = torch.randn(1, 5, 4)
    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=transported,
        matching_query_count=3,
        fusion=None,
        capacity=None)
    torch.testing.assert_close(output, transported)
    assert gate is None


def test_fusion_without_density_uses_no_capacity_mask():
    fusion = SemanticEvidenceFusion(embed_dims=4, adapter_init='identity')
    with torch.no_grad():
        fusion.gate.bias.fill_(1.0)
    native = torch.randn(1, 3, 4)
    transported = torch.randn(1, 3, 4)
    output, gate = apply_matching_query_interventions(
        native_query=native,
        transported_query=transported,
        matching_query_count=3,
        fusion=fusion,
        capacity=None)
    expected, expected_gate = fusion(native, transported, capacity=None)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(gate, expected_gate)


def test_decoder_layer_omits_disabled_modules():
    layer = OVCapFlowDecoderLayer(
        enable_semantic_fusion=False,
        enable_density_capacity=False,
        self_attn_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
        cross_attn_text_cfg=dict(embed_dims=8, num_heads=2, dropout=0.0),
        cross_attn_cfg=dict(
            embed_dims=8, num_heads=2, num_levels=1, dropout=0.0),
        ffn_cfg=dict(embed_dims=8, feedforward_channels=16, ffn_drop=0.0))
    assert layer.semantic_fusion is None
    assert layer.density_capacity is None
