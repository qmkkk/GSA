# SPDX-License-Identifier: Apache-2.0
"""A dataset-free example of index reuse and private result reuse."""

import torch
from gsa import QuerySession

torch.manual_seed(7)
with torch.inference_mode():
    support = torch.randn(4096, 3, device="cuda")
    queries = support[:256].clone()
    indices = torch.empty(256, 16, dtype=torch.int32, device="cuda")
    smaller = torch.empty(64, 8, dtype=torch.int32, device="cuda")
    distance = torch.empty(64, 3, device="cuda")
    nearest = torch.empty(64, 3, dtype=torch.int32, device="cuda")
    with QuerySession(threshold=32) as session:
        session.ball_into(queries, support, 0.4, 16, indices)
        session.ball_into(queries[:64].clone(), support.clone(), 0.4, 8, smaller)
        session.nn_into(queries[:64], support[:1024], distance, nearest)
        torch.cuda.synchronize()
        assert torch.equal(smaller, indices[:64, :8])
        assert session.stats["ball_hits"] == 1
        assert session.stats["tree_reuses"] == 1
        print(session.stats)
