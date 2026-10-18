# SPDX-License-Identifier: Apache-2.0
"""Recompute published numerical results from the included measurement records (CPU only)."""

from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gm(values):
    return math.exp(st.mean(math.log(v) for v in values))


def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def metric(matrix):
    n = len(matrix)
    iou = []
    for c in range(n):
        intersection = matrix[c][c]
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - intersection
        iou.append(intersection / union if union else 0.0)
    return st.mean(iou) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/recomputed.csv")
    args = parser.parse_args()
    data = args.data
    manifest = read(data / "manifest.json")
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    full = [read(p) for p in sorted((data / "full").rglob("*.json"))]
    assert len(full) == 760
    assert sum(r["multi_voxel_passes"] * 3 for r in full) == 43146
    output = []
    for group in read(data / "summaries/full.json")["groups"]:
        rows = [
            r
            for r in full
            if (r["dataset"], r["model"]) == (group["dataset"], group["model"])
        ]
        assert len(rows) == group["scenes"]
        for r in rows:
            assert r["all_pass_logits_bitexact"]
            assert len({v["prediction_sha256"] for v in r["methods"].values()}) == 1
            assert all(
                v["confusion"] == r["methods"]["flash"]["confusion"]
                for v in r["methods"].values()
            )
        for method in ["flash", "scale", "prefix"]:
            seconds = (
                sum(t["wall_ms"] for r in rows for t in r["methods"][method]["timings"])
                / 1000
            )
            equal(seconds, group["methods"][method]["forward_seconds"])
            cm = rows[0]["methods"][method]["confusion"]
            summed = [
                [
                    sum(r["methods"][method]["confusion"][i][j] for r in rows)
                    for j in range(len(cm))
                ]
                for i in range(len(cm))
            ]
            assert summed == group["methods"][method]["confusion"]
            equal(metric(summed), group["methods"][method]["mIoU"])
        for baseline in ["flash", "scale"]:
            ratios = [
                sum(t["wall_ms"] for t in r["methods"][baseline]["timings"])
                / sum(t["wall_ms"] for t in r["methods"]["prefix"]["timings"])
                for r in rows
            ]
            value = gm(ratios)
            equal(
                value,
                group["comparisons"][baseline]["forward_seconds"][
                    "scene_geomean_speedup"
                ],
            )
            output.append(
                [
                    "full",
                    group["dataset"],
                    group["model"],
                    "prefix_vs_" + baseline,
                    len(rows),
                    value,
                ]
            )

    strong = [read(p) for p in sorted((data / "strong_baselines").glob("*.json"))]
    assert len(strong) == 96
    for group in read(data / "summaries/strong_baselines.json")["groups"]:
        rows = [
            r
            for r in strong
            if (r["dataset"], r["model"]) == (group["dataset"], group["model"])
        ]
        # Pair repetitions within an input, then combine two seeds within each scene.
        for comparison, expected in group["comparisons"].items():
            candidate, baseline = comparison.split("_vs_")
            by_scene = collections.defaultdict(list)
            for r in rows:
                ratio = st.median(
                    v["wall_ms"] for v in r["methods"][baseline]
                ) / st.median(v["wall_ms"] for v in r["methods"][candidate])
                by_scene[r["scene"]].append(ratio)
            for scene, values in by_scene.items():
                equal(gm(values), expected["scene_speedups"][scene])
            value = gm(gm(v) for v in by_scene.values())
            equal(value, expected["speedup"])
            output.append(
                [
                    "strong",
                    group["dataset"],
                    group["model"],
                    comparison,
                    len(rows),
                    value,
                ]
            )

    sessions = [
        read(p)
        for p in sorted((data / "query_session").glob("*.json"))
        if p.name != "correctness.json"
    ]
    assert len(sessions) == 96
    assert read(data / "query_session/correctness.json")["calls"] == 94
    for group in read(data / "summaries/query_session.json")["groups"]:
        rows = [
            r
            for r in sessions
            if (r["dataset"], r["model"]) == (group["dataset"], group["model"])
        ]
        assert all(r["all_logits_bitexact"] for r in rows)
        for comparison, expected in group["comparisons"].items():
            candidate, baseline = comparison.split("_vs_")
            by_scene = collections.defaultdict(list)
            for r in rows:
                by_scene[r["scene"]].append(
                    st.median(v["wall_ms"] for v in r["methods"][baseline])
                    / st.median(v["wall_ms"] for v in r["methods"][candidate])
                )
            equal(gm(gm(v) for v in by_scene.values()), expected["speedup"])

    ablations = [read(p) for p in sorted((data / "ablations").glob("*.json"))]
    assert len(ablations) == 24
    for group in read(data / "summaries/ablations.json")["groups"]:
        rows = [
            r
            for r in ablations
            if (r["dataset"], r["model"]) == (group["dataset"], group["model"])
        ]
        for method, expected in group["variants"].items():
            value = gm(
                st.median(t["wall_ms"] for t in r["methods"]["flash"])
                / st.median(t["wall_ms"] for t in r["methods"][method])
                for r in rows
            )
            equal(value, expected["speed_vs_flash"])

    first = [read(p) for p in sorted((data / "resources/first").glob("*.json"))]
    warm = [read(p) for p in sorted((data / "resources/warm").glob("*.json"))]
    assert len(first) == 108 and len(warm) == 36
    assert (
        sum(len(r["measurements"]) + len(r["profiles"]) for r in first + warm) == 5220
    )
    for row in read(data / "summaries/resources.json")["resource_rows"]:
        raw = [
            r
            for r in first
            if (r["scene"], r["model"], r["method"])
            == (row["scene"], row["model"], row["method"])
        ]
        warmed = [
            r for r in warm if (r["scene"], r["model"]) == (row["scene"], row["model"])
        ]
        assert len(raw) == len(warmed) == 3
        equal(
            st.median(r["measurements"][0]["wall_ms"] for r in raw),
            row["first_ms_median"],
        )
        equal(
            st.median(
                st.median(
                    m["wall_ms"]
                    for m in r["measurements"]
                    if m["method"] == row["method"] and m["phase"] == "measured"
                )
                for r in warmed
            ),
            row["warm_ms_median_of_block_medians"],
        )
        equal(
            max(
                m["memory"]["used_peak_upper_bound_bytes"]
                for r in raw
                for m in r["measurements"]
            )
            / 2**20,
            row["peak_used_upper_bound_MiB"],
        )
    scale = read(data / "summaries/scale.json")
    assert len(scale["rows"]) == 760
    for band in scale["bins"]:
        rows = [
            r
            for r in scale["rows"]
            if (r["dataset"], r["model"]) == (band["dataset"], band["model"])
            and band["voxel_n_min"] <= r["voxel_n"] <= band["voxel_n_max"]
        ]
        assert len(rows) == band["n"]
        equal(gm(r["forward_speedup"] for r in rows), band["forward"]["gm"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cohort", "dataset", "model", "comparison", "cases", "geomean_speedup"]
        )
        writer.writerows(output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "full_cases": 760,
                "strong_cases": 96,
                "session_cases": 96,
                "ablation_cases": 24,
                "resource_processes": 144,
                "resource_output_checks": 5220,
                "scale_cases": 760,
                "sha256_files": len(manifest["sha256"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
