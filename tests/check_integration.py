# SPDX-License-Identifier: Apache-2.0
"""Check fixed-graph equivalence, exception cleanup, and public buffer contracts."""

import torch
from gsa import QuerySession, fixed_graph_forward
from gsa import reference as ops

torch.manual_seed(123)
torch.set_num_threads(1)
data = {"pos": torch.randn(1, 4096, 3, device="cuda")}


def graph(inputs):
    root = inputs["pos"]
    sampled = root[:, :1024].contiguous()
    outputs = []
    for support, queries, k in [
        (root, sampled, 16),
        (sampled, sampled, 32),
        (sampled, sampled[:, :256], 8),
        (sampled[:, :256], sampled[:, :64], 16),
    ]:
        result = torch.zeros(1, queries.shape[1], k, device="cuda", dtype=torch.int32)
        ops.ball_query_wrapper(
            1, support.shape[1], queries.shape[1], 0.5, k, queries, support, result
        )
        outputs.append(result)
    for support in [sampled, sampled[:, :256], sampled[:, :64]]:
        q = root[:, :79]
        distance = torch.empty(1, 79, 3, device="cuda")
        ids = torch.empty(1, 79, 3, device="cuda", dtype=torch.int32)
        ops.three_nn_wrapper(1, 79, support.shape[1], q, support, distance, ids)
        outputs.extend([distance, ids])
    return outputs


with torch.inference_mode():
    expected = graph(data)
    original_ball, original_nn = ops.ball_query_wrapper, ops.three_nn_wrapper
    for policy in ["prefix", "scale"]:
        stats = {}
        actual = fixed_graph_forward(
            graph, data, ops=ops, policy=policy, threshold=32, stats=stats
        )
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert stats["ball_hits"] == 1
        assert (
            ops.ball_query_wrapper is original_ball
            and ops.three_nn_wrapper is original_nn
        )

    def raising_graph(inputs):
        graph(inputs)
        raise RuntimeError("test cleanup")

    try:
        fixed_graph_forward(raising_graph, data, ops=ops, threshold=32)
    except RuntimeError as error:
        assert str(error) == "test cleanup"
    else:
        raise AssertionError("Expected model exception")
    assert (
        ops.ball_query_wrapper is original_ball and ops.three_nn_wrapper is original_nn
    )

    x = data["pos"][0]
    q = x[:8]
    with QuerySession() as session:
        invalid = [
            lambda: session.ball_into(q, x, 0.5, 8, torch.empty(8, 8, device="cuda")),
            lambda: session.nn_into(
                q,
                x,
                torch.empty(8, 3, device="cuda", dtype=torch.float64),
                torch.empty(8, 3, device="cuda", dtype=torch.int32),
            ),
            lambda: session.ball_into(
                q, x, -0.5, 8, torch.empty(8, 8, device="cuda", dtype=torch.int32)
            ),
            lambda: session.ball_into(
                q,
                x,
                0.5,
                8,
                torch.empty(8, 16, device="cuda", dtype=torch.int32)[:, ::2],
            ),
        ]
        for operation in invalid:
            try:
                operation()
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid output or radius accepted")
    torch.cuda.synchronize()
print(
    "PASS: prefix/scale graph equivalence, cache hit, exception cleanup, four invalid-buffer/radius checks"
)
