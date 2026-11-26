"""Pinned MiniLM/Korean cross-encoders; offline CUDA inference only."""

import argparse
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.data.timelyqa import DatasetError
from temporalproof.retrieval.bm25 import Hit

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
DEFAULT_MODEL = Path("models/mmarco-minilm") / REVISION
ASSETS = {
    "model.safetensors": (
        470592698,
        "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4",
    ),
    "tokenizer.json": (
        17082660,
        "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
    ),
    "sentencepiece.bpe.model": (
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "config.json": (891, "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77"),
    "tokenizer_config.json": (
        435,
        "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e",
    ),
    "special_tokens_map.json": (
        239,
        "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    directory: Path
    assets: dict[str, tuple[int, str]]
    position_count: int
    korean_scope: str


MINILM = ModelSpec(
    MODEL_ID,
    REVISION,
    DEFAULT_MODEL,
    ASSETS,
    514,
    "transfer; absent from documented mMARCO supervised languages",
)
KOREAN_MODEL_ID = "dragonkue/bge-reranker-v2-m3-ko"
KOREAN_REVISION = "2aca5884ecac490192af9ebd86836d9073d826cd"
KOREAN = ModelSpec(
    KOREAN_MODEL_ID,
    KOREAN_REVISION,
    Path("models/bge-reranker-v2-m3-ko") / KOREAN_REVISION,
    {
        "model.safetensors": (
            2271071852,
            "aba51b69d97bfff4cd2c1d8eb6c57764e350231850f712b75ae426270f265a43",
        ),
        "tokenizer.json": (
            17098251,
            "b74659c780d49afad7a7b9799868f75cbd3014fb6c34956e85a793028d38094a",
        ),
        "sentencepiece.bpe.model": (
            5069051,
            "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        ),
        "config.json": (808, "a1d675becc0aa53d579723d7842dc25bea0797fecdc5004f55b9bdb43394d69a"),
        "tokenizer_config.json": (
            1173,
            "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",
        ),
        "special_tokens_map.json": (
            964,
            "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
        ),
    },
    8194,
    "Korean optimization declared by author; complete fine-tuning data undisclosed; "
    "benchmark contamination not independently excluded",
)
MODELS = {spec.model_id: spec for spec in (MINILM, KOREAN)}


def model_spec(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise DatasetError("unsupported reranker model; use a pinned registered model") from None


def verify_model(directory: Path, spec: ModelSpec | None = None) -> None:
    for name, (size, digest) in (spec.assets if spec is not None else ASSETS).items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise DatasetError("reranker asset missing or checksum mismatch; fetch explicitly")


def fetch_model(directory: Path, spec: ModelSpec | None = None) -> None:
    assets = spec.assets if spec is not None else ASSETS
    source = spec or MINILM
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in assets.items():
        target = directory / name
        if target.exists():
            if target.stat().st_size != size or sha256_file(target) != digest:
                raise DatasetError("existing reranker asset differs; original was not replaced")
            continue
        temporary = None
        try:
            with urlopen(
                f"https://huggingface.co/{source.model_id}/resolve/{source.revision}/{name}",
                timeout=60,
            ) as response:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise DatasetError("reranker download exceeds pinned size")
                        stream.write(chunk)
            if temporary.stat().st_size != size or sha256_file(temporary) != digest:
                raise DatasetError("reranker download checksum mismatch")
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Verified reranker asset: {name} ({size} bytes)", flush=True)


def rank_scores(doc_ids: Sequence[int], scores: Sequence[float]) -> list[Hit]:
    values = np.asarray(scores, dtype=np.float64)
    if len(set(doc_ids)) != len(doc_ids) or values.shape != (len(doc_ids),):
        raise ValueError("reranker requires unique aligned candidate IDs and scalar scores")
    if not np.isfinite(values).all():
        raise ValueError("reranker scores must be finite")
    return sorted(
        [Hit(doc_id, float(score)) for doc_id, score in zip(doc_ids, values, strict=True)],
        key=lambda hit: (-hit.score, hit.doc_id),
    )


class CrossEncoder:
    """Single raw relevance logit, higher is better; no timestamps or qrels as inputs."""

    def __init__(
        self,
        directory: Path,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        spec: ModelSpec = MINILM,
    ):
        verify_model(directory, spec)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise DatasetError("CUDA unavailable; refusing an unreported CPU fallback")
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("invalid batch size or maximum length")
        torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.spec = spec
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        unexpected = set(loading.get("unexpected_keys", []))
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or unexpected - {"roberta.embeddings.position_ids"}
        ):
            raise DatasetError("unexpected reranker parameter mismatch")
        if unexpected:
            from safetensors import safe_open

            with safe_open(directory / "model.safetensors", framework="np") as tensors:
                positions = tensors.get_tensor("roberta.embeddings.position_ids")
            if not np.array_equal(positions, np.arange(spec.position_count).reshape(1, -1)):
                raise DatasetError("legacy reranker position buffer differs")
        if self.model.config.num_labels != 1:
            raise DatasetError("reranker must expose one relevance logit")
        if self.model.config.max_position_embeddings != spec.position_count:
            raise DatasetError("reranker position capacity differs from pinned specification")
        self.loading_info = {
            "missing_keys": [],
            "unexpected_keys": sorted(unexpected),
            "legacy_position_buffer_checked": bool(unexpected),
        }
        self.model = self.model.to("cuda").eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.pairs_at_max_length = 0
        self.pair_count = 0
        self.call_timings: list[dict] = []

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        torch = self.torch
        results = []
        tokenization_ms = 0.0
        forward_ms = 0.0
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = list(passages[start : start + self.batch_size])
                started = perf_counter()
                tokens = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenization_ms += (perf_counter() - started) * 1000
                self.pairs_at_max_length += int(
                    (tokens["attention_mask"].sum(1) == self.max_length).sum()
                )
                self.pair_count += len(batch)
                started = perf_counter()
                tokens = {key: value.to("cuda") for key, value in tokens.items()}
                logits = self.model(**tokens).logits
                if logits.shape != (len(batch), 1):
                    raise DatasetError("unexpected relevance score shape")
                results.append(logits[:, 0].cpu().numpy())
                torch.cuda.synchronize()
                forward_ms += (perf_counter() - started) * 1000
        torch.cuda.synchronize()
        scores = np.concatenate(results) if results else np.empty(0, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise DatasetError("nonfinite reranker scores")
        self.call_timings.append(
            {
                "pair_count": len(passages),
                "cpu_tokenization_ms": tokenization_ms,
                "synchronized_forward_and_transfers_ms": forward_ms,
            }
        )
        return scores

    def profile(self) -> dict:
        torch = self.torch
        return {
            "model_id": self.spec.model_id,
            "revision": self.spec.revision,
            "license": "Apache-2.0 (model card)",
            "asset_sha256": {name: item[1] for name, item in self.spec.assets.items()},
            "dtype": "float32",
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "truncation": "longest_first",
            "padding": "dynamic_per_batch",
            "score": "single_raw_logit; higher_is_better; identity_activation",
            "input": "query/text pairs; no prefixes, titles, timestamps or qrels",
            "attention": "eager",
            "tf32": False,
            "seed": 1729,
            "gpu": torch.cuda.get_device_name(),
            "cuda_runtime": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "pair_count_including_warmup": self.pair_count,
            "pairs_at_max_length_including_warmup": self.pairs_at_max_length,
            "length_count_scope": "potential truncation; not exact truncated count",
            "loading_info": self.loading_info,
            "korean_scope": self.spec.korean_scope,
            "resident_models": "reranker only; frozen candidates; encoder not loaded",
            "call_timings_including_warmup": self.call_timings,
            "component_timing_scope": (
                "wall clock; forward includes CUDA work, CPU dispatch and transfers; "
                "not kernel-only GPU time; warmup calls precede measured calls"
            ),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=sorted(MODELS), default=MODEL_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--download", action="store_true", help="explicitly fetch pinned model assets"
    )
    args = parser.parse_args(argv)
    spec = model_spec(args.model_id)
    directory = args.output or spec.directory
    if args.download:
        fetch_model(directory, spec)
    verify_model(directory, spec)
    print(f"Pinned reranker verified: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
