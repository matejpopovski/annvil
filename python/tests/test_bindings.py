import threading

import annvil
import numpy as np
import pytest

DIM = 24


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    centres = rng.normal(0.0, 3.0, size=(16, DIM)).astype(np.float32)
    picks = rng.integers(0, len(centres), size=3000)
    return centres[picks] + rng.normal(0.0, 1.0, size=(3000, DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def index(data):
    ix = annvil.Index(dim=DIM, capacity=len(data), M=12, ef_construction=100)
    ix.add(data, threads=2)
    return ix


def exact_labels(data, queries, k):
    distances = ((queries[:, None, :] - data[None, :, :]) ** 2).sum(axis=2)
    return np.argsort(distances, axis=1)[:, :k]


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth, strict=True)])


def test_kernel_is_reported():
    assert annvil.KERNEL in {"neon", "sse2", "avx2", "scalar"}


def test_search_matches_brute_force(index, data):
    queries = data[:100] + 0.01
    labels, distances = index.search(queries, k=10, ef=128)
    assert labels.shape == (100, 10)
    assert labels.dtype == np.uint64
    assert recall(labels, exact_labels(data, queries, 10)) > 0.98
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_single_vector_query_returns_one_row(index, data):
    labels, _ = index.search(data[7], k=1, ef=64)
    assert labels.shape == (1, 1)
    assert labels[0, 0] == 7


def test_float64_input_is_accepted(index, data):
    labels, _ = index.search(data[:3].astype(np.float64), k=1, ef=64)
    assert labels[:, 0].tolist() == [0, 1, 2]


def test_short_results_are_padded():
    ix = annvil.Index(dim=4, capacity=2)
    ix.add(np.eye(4, dtype=np.float32)[:2])
    labels, distances = ix.search(np.zeros(4, dtype=np.float32), k=5, ef=10)
    assert labels[0, 2] == np.iinfo(np.uint64).max
    assert np.isinf(distances[0, 2])


def test_custom_labels():
    ix = annvil.Index(dim=2, capacity=2)
    ix.add(
        np.array([[0, 0], [1, 1]], dtype=np.float32),
        labels=np.array([100, 200], dtype=np.uint64),
    )
    labels, _ = ix.search(np.array([0.9, 0.9], dtype=np.float32), k=1)
    assert labels[0, 0] == 200


def test_wrong_dimension_raises(index):
    with pytest.raises(ValueError):
        index.search(np.zeros(DIM + 1, dtype=np.float32))


def test_nan_is_rejected(index):
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = np.nan
    with pytest.raises(ValueError):
        index.search(query)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        annvil.Index(dim=4, capacity=4, metric="cosine-ish")


def test_angular_metric_ignores_scale(data):
    ix = annvil.Index(dim=DIM, capacity=500, metric="angular", M=8, ef_construction=64)
    ix.add(data[:500])
    assert ix.metric == "angular"
    labels, distances = ix.search(data[3] * 50.0, k=1, ef=64)
    assert labels[0, 0] == 3
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_save_and_mmap_round_trip(index, data, tmp_path):
    path = tmp_path / "x.annvil"
    index.save(str(path))
    mapped = annvil.Index.open(str(path), verify=True)
    assert mapped.read_only
    assert len(mapped) == len(index)
    a, _ = index.search(data[:20], k=5, ef=64)
    b, _ = mapped.search(data[:20], k=5, ef=64)
    assert np.array_equal(a, b)
    with pytest.raises(RuntimeError):
        mapped.add(data[:1])


def test_opening_garbage_raises(tmp_path):
    path = tmp_path / "bad"
    path.write_bytes(b"x" * 8192)
    with pytest.raises(RuntimeError):
        annvil.Index.open(str(path))


def test_pq_search(data):
    ix = annvil.Index(dim=DIM, capacity=len(data), M=12, ef_construction=100)
    ix.add(data, threads=2)
    ix.train_pq(m=8, threads=2)
    assert ix.has_pq
    assert ix.memory["pq"] > 0
    queries = data[:100] + 0.01
    labels, _ = ix.search_pq(queries, k=10, ef=200, rerank=100)
    assert recall(labels, exact_labels(data, queries, 10)) > 0.9


def test_verify_and_reorder(data):
    ix = annvil.Index(dim=DIM, capacity=800, M=8, ef_construction=64)
    ix.add(data[:800])
    ix.reorder()
    ok, errors = ix.verify()
    assert ok, errors


def test_python_threads_search_in_parallel(index, data):
    expected, _ = index.search(data[:50], k=5, ef=64)
    results = {}

    def worker(slot):
        results[slot] = index.search(data[:50], k=5, ef=64)[0]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for labels in results.values():
        assert np.array_equal(labels, expected)
