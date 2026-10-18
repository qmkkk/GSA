# SPDX-License-Identifier: Apache-2.0
"""Fixed hierarchical graph adapter. See docs/integration.md for admission rules.

The caller configures its model and sampling policy before entering this adapter.
ops exposes the two PointNet++ wrapper signatures. Calls must be sequential.
"""

import torch
from .runtime import Tree


def forward(
    model,
    data,
    *,
    ops,
    model_forward=None,
    policy="prefix",
    cache=True,
    threshold=1024,
    leaf=16,
    verify=True,
    stats=None
):
    assert policy in ["flash", "prefix", "scale"]
    rt = ops
    if model_forward is None:
        model_forward = lambda m, d: m(d)
    if policy == "flash":
        return model_forward(model, data)
    origball = rt.ball_query_wrapper
    orignn = rt.three_nn_wrapper
    state = {"full": None}
    trees = {}
    memo = {}
    n0 = data["pos"].shape[1]
    counters = {
        "builds": 0,
        "ball_calls": 0,
        "ball_hits": 0,
        "nn_calls": 0,
        "cache_bytes": 0,
        "fallbacks": 0,
    }

    def get_tree(x):
        n = x.shape[1]
        nested = policy == "prefix" and n != n0
        key = "prefix" if nested else n
        if nested and verify:
            assert torch.equal(x, state["full"][:, :n])
        if key not in trees:
            trees[key] = Tree(state["full"][0] if nested else x[0], nested, leaf)
            counters["builds"] += 1
        return trees[key]

    def ball(b, n, m, r, ns, q, x, idx):
        assert b == 1 and ns <= 32
        counters["ball_calls"] += 1
        if state["full"] is None:
            assert n == n0
            state["full"] = q
        key = (n, r)
        reusable = n != n0
        if cache and reusable and key in memo:
            cached, oldq, oldx = memo[key]
            if verify:
                assert torch.equal(q, oldq[:, :m]) and torch.equal(x, oldx)
            idx.copy_(cached[:, :m, :ns])
            counters["ball_hits"] += 1
            return
        cap = 32 if cache and reusable else ns
        out = torch.zeros((b, m, cap), device=q.device, dtype=torch.int32)
        if n <= threshold:
            origball(b, n, m, r, cap, q, x, out)
        else:
            get_tree(x).ball(q[0], n, r, cap, out[0])
        idx.copy_(out[:, :, :ns])
        if cache and reusable and m == n:
            memo[key] = (out, q, x)
            counters["cache_bytes"] += out.numel() * out.element_size()
        if n == n0 and n in trees:
            if verify:
                counters["fallbacks"] += int(trees[n].fallback.item())
            trees.pop(n).close()

    def nn(b, n, m, q, x, d, idx):
        assert b == 1
        counters["nn_calls"] += 1
        if m <= threshold:
            orignn(b, n, m, q, x, d, idx)
        else:
            get_tree(x).query(q[0], m, d[0], idx[0])

    rt.ball_query_wrapper = ball
    rt.three_nn_wrapper = nn
    try:
        return model_forward(model, data)
    finally:
        rt.ball_query_wrapper = origball
        rt.three_nn_wrapper = orignn
        for tree in trees.values():
            if verify:
                counters["fallbacks"] += int(tree.fallback.item())
            tree.close()
        if stats is not None:
            stats.update(counters)
