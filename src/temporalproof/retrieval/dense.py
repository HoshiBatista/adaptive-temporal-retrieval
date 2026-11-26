"""FAISS exact cosine retrieval and immutable corpus-embedding caches."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.data.timelyqa import DatasetError, Document
from temporalproof.retrieval.bm25 import Hit


def normalized_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("zero-norm embedding")
    return np.ascontiguousarray(matrix / norms)


class DenseIndex:
    def __init__(self, ids: np.ndarray, embeddings: np.ndarray):
        import faiss

        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
            raise ValueError("embedding IDs must be a one-dimensional nonnegative integer array")
        if len(ids) != len(embeddings) or len(set(map(int, ids))) != len(ids) or not len(ids):
            raise ValueError("embedding IDs must be nonempty, unique and aligned")
        order = np.argsort(ids)
        self.ids = np.asarray(ids, dtype=np.int64)[order]
        vectors = normalized_matrix(embeddings)[order]
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> list[Hit]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        vector = normalized_matrix(np.asarray(query).reshape(1, -1))
        if vector.shape[1] != self.index.d:
            raise ValueError("query embedding dimension mismatch")
        # Full exact search lets ties at the top-k boundary use ALL document IDs.
        scores, positions = self.index.search(vector, len(self.ids))
        ids = self.ids[positions[0]]
        order = np.lexsort((ids, -scores[0]))[:top_k]
        return [Hit(int(ids[i]), float(scores[0, i])) for i in order]


def corpus_cache(documents: list[Document], encoder, directory: Path) -> tuple[DenseIndex, dict]:
    """Cache keys include every ID/text and encoder implementation/configuration."""
    ordered = sorted(documents, key=lambda doc: doc.global_doc_id)
    digest = hashlib.sha256()
    for doc in ordered:
        digest.update(json.dumps([doc.global_doc_id, doc.text], ensure_ascii=False).encode())
        digest.update(b"\n")
    profile = encoder.profile()
    identity = {
        key: value
        for key, value in profile.items()
        if key
        not in {
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "inputs_at_max_length",
            "length_count_scope",
        }
    }
    identity.update(
        corpus_text_ids_sha256=digest.hexdigest(),
        encoder_implementation_sha256=sha256_file(Path(__file__).with_name("encoder.py")),
        cache_implementation_sha256=sha256_file(Path(__file__)),
        faiss_version=importlib.metadata.version("faiss-cpu"),
        environment_lock_sha256=sha256_file(Path(__file__).resolve().parents[3] / "uv.lock"),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = directory / key
    manifest_path = cache / "manifest.json"
    if cache.exists():
        if not manifest_path.is_file():
            raise DatasetError("embedding cache is incomplete; choose a new cache directory")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise DatasetError("embedding cache identity mismatch")
        for name in ["ids.npy", "embeddings.npy"]:
            if sha256_file(cache / name) != manifest["files_sha256"][name]:
                raise DatasetError("embedding cache checksum mismatch")
        ids = np.load(cache / "ids.npy", allow_pickle=False)
        vectors = np.load(cache / "embeddings.npy", allow_pickle=False)
        metadata = {**manifest, "cache_hit": True}
    else:
        started = perf_counter()
        vectors = encoder.encode([doc.text for doc in ordered], kind="passage")
        ids = np.array([doc.global_doc_id for doc in ordered], dtype=np.int64)
        cache.mkdir(parents=True, exist_ok=False)
        np.save(cache / "ids.npy", ids, allow_pickle=False)
        np.save(cache / "embeddings.npy", vectors, allow_pickle=False)
        manifest = {
            "identity": identity,
            "embedding_build_ms": (perf_counter() - started) * 1000,
            "input_count_at_max_length": encoder.inputs_at_max_length,
            "files_sha256": {
                name: sha256_file(cache / name) for name in ["ids.npy", "embeddings.npy"]
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        metadata = {**manifest, "cache_hit": False}
    started = perf_counter()
    index = DenseIndex(ids, vectors)
    metadata.update(
        cache_key=key,
        faiss_index="IndexFlatIP",
        faiss_build_ms=(perf_counter() - started) * 1000,
        dimension=index.index.d,
        document_count=index.index.ntotal,
        cache_manifest_sha256=sha256_file(manifest_path),
    )
    return index, metadata
