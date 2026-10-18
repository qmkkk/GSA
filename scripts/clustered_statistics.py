# SPDX-License-Identifier: Apache-2.0
"""Resample physical spaces while retaining the registered per-scan estimand."""

import re
import numpy as np


def physical_key(dataset, scene):
    if dataset == "scannet":
        m = re.match(r"^(scene[0-9]{4})_[0-9]{2}(?:_|$)", scene)
        if not m:
            raise ValueError(("Invalid ScanNet scene identity", scene))
        return m.group(1)
    if dataset == "s3dis":
        return scene
    raise ValueError(dataset)


def group_plan(dataset, scenes, samples=10000, seed=20261072):
    labels = [physical_key(dataset, s) for s in scenes]
    groups = sorted(set(labels))
    indices = [np.flatnonzero(np.array(labels) == g) for g in groups]
    counts = np.array([len(ix) for ix in indices])
    draws = np.random.default_rng(seed).integers(0, len(groups), (samples, len(groups)))
    return {"names": groups, "indices": indices, "counts": counts, "draws": draws}


def bootstrap_scan_mean(values, plan):
    values = np.asarray(values)
    sums = np.array([values[ix].sum() for ix in plan["indices"]])
    draws = plan["draws"]
    return sums[draws].sum(1) / plan["counts"][draws].sum(1)


def physical_mean(values, plan):
    return float(np.mean([np.asarray(values)[ix].mean() for ix in plan["indices"]]))
