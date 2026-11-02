#!/usr/bin/env python3
"""Flatten an ann-benchmarks HDF5 file into the .annb layout read by annvil.

The harness stays free of an HDF5 dependency by consuming this format instead. The
layout is documented in include/annvil/dataset.hpp.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np

MAGIC = b"ANNB0001"
HEADER_SIZE = 64
METRICS = {"euclidean": 0, "angular": 1}


def convert(source: Path, destination: Path) -> None:
    with h5py.File(source, "r") as f:
        metric = f.attrs["distance"]
        if isinstance(metric, bytes):
            metric = metric.decode()
        if metric not in METRICS:
            raise SystemExit(f"{source}: unsupported distance {metric!r}")

        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        distances = np.asarray(f["distances"], dtype=np.float32)

    if train.shape[1] != test.shape[1]:
        raise SystemExit(f"{source}: train and test dimensions differ")
    if neighbors.shape != distances.shape or neighbors.shape[0] != test.shape[0]:
        raise SystemExit(f"{source}: ground truth does not line up with the test set")

    header = struct.pack(
        "<8sIIQQQ",
        MAGIC,
        METRICS[metric],
        train.shape[1],
        train.shape[0],
        test.shape[0],
        neighbors.shape[1],
    ).ljust(HEADER_SIZE, b"\0")

    temporary = destination.with_suffix(".annb.tmp")
    with temporary.open("wb") as out:
        out.write(header)
        for array in (train, test, neighbors, distances):
            array.astype(array.dtype.newbyteorder("<")).tofile(out)
    temporary.replace(destination)
    print(
        f"{destination.name}: {train.shape[0]} train, {test.shape[0]} test, "
        f"dim {train.shape[1]}, {metric}, {neighbors.shape[1]} true neighbours per query"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
