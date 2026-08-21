import math
from typing import Tuple

import torch
from torch import Tensor, nn


class FixedRotatedQueryInitializer(nn.Module):
    """Learned content queries with learned five-dimensional references.

    The module deliberately has no image-memory input. This makes the strict
    matching-query set independent of encoder proposal scores and prevents an
    accidental reintroduction of proposal top-k during refactors.
    """

    def __init__(self, num_queries: int, embed_dims: int) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError('num_queries must be positive')
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive')

        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.query_embedding = nn.Embedding(num_queries, embed_dims)
        self.reference_embedding = nn.Embedding(num_queries, 5)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_embedding.weight, mean=0.0, std=0.02)

        side = math.ceil(math.sqrt(self.num_queries))
        query_ids = torch.arange(self.num_queries, dtype=torch.float32)
        center_x = (query_ids.remainder(side) + 0.5) / side
        center_y = (torch.floor(query_ids / side) + 0.5) / side
        initial_size = torch.full_like(center_x, 1.0 / side)
        initial_angle = torch.full_like(center_x, 0.5)
        references = torch.stack(
            [center_x, center_y, initial_size, initial_size, initial_angle],
            dim=-1)
        references = references.clamp(min=1e-4, max=1 - 1e-4)

        with torch.no_grad():
            self.reference_embedding.weight.copy_(torch.logit(references))

    def forward(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')

        query = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1)
        references = self.reference_embedding.weight.sigmoid().unsqueeze(
            0).expand(batch_size, -1, -1)
        return query, references
