# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import argparse
import gsa.session as session

parser = argparse.ArgumentParser(
    description="94 independent FP32 oracle and state-transition checks"
)
parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
args = parser.parse_args()
reg = {"correctness_seed": 20261111}
load = lambda: session
import datetime, hashlib, json
import numpy as np

session = load()
torch = session.torch
torch.set_num_threads(4)
rng = np.random.default_rng(reg["correctness_seed"])
rows = []


def distance(q, x):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = q[:, None, :] - x[None, :, :]
        sq = delta * delta
        return (sq[:, :, 0] + sq[:, :, 1]) + sq[:, :, 2]


def ball_oracle(q, x, r, k):
    result = np.zeros((len(q), k), np.int32)
    d = distance(q, x)
    with np.errstate(over="ignore", under="ignore"):
        r2 = np.float32(r) * np.float32(r)
    for i, mask in enumerate(d < r2):
        ids = np.flatnonzero(mask)[:k]
        if len(ids):
            result[i, :] = ids[0]
            result[i, : len(ids)] = ids
    return result


def ball(s, x, q, r, k, label):
    out = torch.empty((len(q), k), device="cuda", dtype=torch.int32)
    s.ball_into(q, x, r, k, out)
    a = out.cpu().numpy()
    expected = ball_oracle(q.cpu().numpy(), x.cpu().numpy(), r, k)
    assert np.array_equal(a, expected), label
    rows.append(
        {
            "op": "ball",
            "label": label,
            "n": len(x),
            "q": len(q),
            "r": r,
            "k": k,
            "ids_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "stats": dict(s.stats),
        }
    )
    return out


def nn(s, x, q, label):
    d = torch.empty((len(q), 3), device="cuda", dtype=torch.float32)
    ids = torch.empty((len(q), 3), device="cuda", dtype=torch.int32)
    s.nn_into(q, x, d, ids)
    distances = distance(q.cpu().numpy(), x.cpu().numpy())
    ed = np.full((len(q), 3), np.inf, np.float32)
    ei = np.zeros((len(q), 3), np.int32)
    for i, row in enumerate(distances):
        valid = np.flatnonzero(np.isfinite(row))
        order = valid[np.argsort(row[valid], kind="stable")[:3]]
        ed[i, : len(order)] = row[order]
        ei[i, : len(order)] = order
    assert np.array_equal(d.cpu().numpy(), ed) and np.array_equal(
        ids.cpu().numpy(), ei
    ), label
    rows.append(
        {"op": "nn", "label": label, "n": len(x), "q": len(q), "stats": dict(s.stats)}
    )


with torch.inference_mode():
    for n in [17, 257, 4099]:
        original = rng.normal(size=(n, 3)).astype(np.float32)
        original[1] = original[0]
        x = torch.from_numpy(original).cuda()
        q = x[:13].clone()
        with session.QuerySession(threshold=32) as s:
            first = ball(s, x, q, 1.0, 8, "initial")
            hits = s.stats["ball_hits"]
            ball(
                s,
                x.clone(),
                q[:7].clone(),
                1.0,
                1,
                "value_equal_clones_and_query_prefix",
            )
            assert s.stats["ball_hits"] == hits + 1
            first.fill_(-123)
            hits = s.stats["ball_hits"]
            ball(s, x, q, 1.0, 32, "private_cached_rows_after_caller_output_edit")
            assert s.stats["ball_hits"] == hits + 1
            ball(s, x, q, 0.0, 32, "strict_zero_radius")
            ball(s, x, q, 4.0, 8, "changed_radius")
            for m in [n, max(3, n // 4), max(3, n // 16), max(3, n - 3)]:
                nn(s, x[:m].contiguous(), q, "prefix_and_independent_support")
            x.add_(0.37)
            ball(s, x, q, 1.0, 8, "inplace_support_update")
            nn(s, x, q, "nn_after_support_update")
            x.copy_(x.flip(0))
            ball(s, x, q, 1.0, 8, "support_permutation")
            nn(s, x, q, "nn_after_permutation")
            q.add_(0.19)
            ball(s, x, q, 1.0, 8, "inplace_query_update")
            old = torch.from_numpy(original.copy()).cuda()
            ball(
                s, old, old[:13].clone(), 1.0, 8, "return_to_original_coordinate_values"
            )
            ball(s, old, old[:0], 1.0, 8, "empty_query")
            nn(s, old, old[:0], "empty_nn_query")
            s.close(collect=True)
            assert s.stats["fallbacks"] == 0
    # Explicit ID-changing sequence: each operation has a different correct answer.
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device="cuda")
    q = torch.zeros((1, 3), device="cuda")
    with session.QuerySession(threshold=32) as s:
        assert ball(s, x, q, 0.1, 1, "id_change_start").item() == 0
        x.copy_(x[[1, 0, 2]])
        assert ball(s, x, q, 0.1, 1, "id_change_after_permutation").item() == 1
        x[1, 0] = 3.0
        x[2, 0] = 0.0
        assert ball(s, x, q, 0.1, 1, "id_change_after_inplace_mutation").item() == 2
    # FP32 scale and ties use the same full exact distance arithmetic.
    for scale in [0.0, 1e-25, 1.0, 1e20]:
        a = (rng.normal(size=(97, 3)) * scale).astype(np.float32)
        x = torch.from_numpy(a).cuda()
        q = x[:9].clone()
        with session.QuerySession(threshold=32) as s:
            for r in [
                0.0,
                float(np.float32(scale)),
                float(np.nextafter(np.float32(scale), np.float32(np.inf))),
            ]:
                for k in [1, 8, 32]:
                    ball(s, x, q, r, k, "scale_radius_K_contract")
            nn(s, x, q, "scaled_nn")
            s.close(collect=True)
            assert s.stats["fallbacks"] == 0
assert len(rows) == 94
dest = args.output
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(
    json.dumps(
        {
            "status": "PASS",
            "at": datetime.datetime.now().isoformat(),
            "calls": len(rows),
            "records": rows,
            "scope": "Independent FP32 oracle plus sequential state-transition and caller-buffer checks. CUDA execution uses the library compiled from this repository.",
        },
        indent=2,
        allow_nan=False,
    )
)
print("QUERY_SESSION_ORACLE_PASS", len(rows), flush=True)
