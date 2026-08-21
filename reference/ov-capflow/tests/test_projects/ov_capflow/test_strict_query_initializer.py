import torch

from projects.OVCapFlow.ov_capflow.query_initializer import (
    FixedRotatedQueryInitializer, )


def test_fixed_initializer_returns_all_queries_and_five_dimensional_refs():
    module = FixedRotatedQueryInitializer(num_queries=9, embed_dims=16)

    query, references = module(batch_size=2)

    assert query.shape == (2, 9, 16)
    assert references.shape == (2, 9, 5)
    assert torch.all((references > 0) & (references < 1))


def test_fixed_initializer_is_independent_of_encoder_memory():
    module = FixedRotatedQueryInitializer(num_queries=4, embed_dims=8)

    query_a, references_a = module(batch_size=1)
    query_b, references_b = module(batch_size=1)

    torch.testing.assert_close(query_a, query_b)
    torch.testing.assert_close(references_a, references_b)


def test_fixed_initializer_parameters_receive_gradients():
    module = FixedRotatedQueryInitializer(num_queries=4, embed_dims=8)

    query, references = module(batch_size=2)
    (query.sum() + references.sum()).backward()

    assert module.query_embedding.weight.grad is not None
    assert module.reference_embedding.weight.grad is not None
