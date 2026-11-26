"""Pinned E5 assets and offline GPU inference; downloads are an explicit command."""

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.data.timelyqa import DatasetError

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_MODEL = Path("models/multilingual-e5-small") / REVISION
ASSETS = {
    "model.safetensors": (
        470641600,
        "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    ),
    "tokenizer.json": (
        17082730,
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    ),
    "sentencepiece.bpe.model": (
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "config.json": (655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    "tokenizer_config.json": (
        443,
        "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
    ),
    "special_tokens_map.json": (
        167,
        "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    ),
}


def verify_model(directory: Path) -> None:
    for name, (size, digest) in ASSETS.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise DatasetError(
                "encoder asset missing or checksum mismatch; fetch pinned assets explicitly"
            )


def fetch_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in ASSETS.items():
        target = directory / name
        if target.exists():
            if target.stat().st_size != size or sha256_file(target) != digest:
                raise DatasetError("existing encoder asset differs; original file was not replaced")
            continue
        temporary = None
        try:
            with urlopen(
                f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}", timeout=60
            ) as response:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    received = 0
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > size:
                            raise DatasetError("encoder download exceeds pinned asset size")
                        stream.write(chunk)
            if temporary.stat().st_size != size or sha256_file(temporary) != digest:
                raise DatasetError("encoder download checksum mismatch")
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Verified encoder asset: {name} ({size} bytes)", flush=True)


class E5Encoder:
    """FP32 mean pooling + L2 normalization, following the author's model card."""

    def __init__(self, directory: Path, *, batch_size: int = 16, max_length: int = 512):
        verify_model(directory)
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise DatasetError("CUDA GPU unavailable; refusing an unreported CPU fallback")
        if batch_size < 1 or not 1 <= max_length <= 512:
            raise ValueError("invalid encoder batch size or maximum length")
        torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        self.model, loading = AutoModel.from_pretrained(
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
            or unexpected - {"embeddings.position_ids"}
        ):
            raise DatasetError("unexpected encoder parameter mismatch")
        if unexpected:
            from safetensors import safe_open

            with safe_open(directory / "model.safetensors", framework="np") as tensors:
                positions = tensors.get_tensor("embeddings.position_ids")
            if not np.array_equal(positions, np.arange(512).reshape(1, -1)):
                raise DatasetError("legacy position buffer differs from generated position IDs")
        self.loading_info = {
            "missing_keys": [],
            "unexpected_keys": sorted(unexpected),
            "legacy_position_buffer_checked": bool(unexpected),
        }
        self.model = self.model.to("cuda").eval()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.inputs_at_max_length = 0

    def encode(self, texts: Sequence[str], *, kind: str) -> np.ndarray:
        if kind not in {"query", "passage"}:
            raise ValueError("E5 input kind must be query or passage")
        torch = self.torch
        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [f"{kind}: {text}" for text in texts[start : start + self.batch_size]]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                self.inputs_at_max_length += int(
                    (tokens["attention_mask"].sum(1) == self.max_length).sum()
                )
                tokens = {key: value.to("cuda") for key, value in tokens.items()}
                states = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).bool()
                pooled = states.masked_fill(~mask, 0).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                results.append(pooled.cpu().numpy().astype(np.float32))
                if kind == "passage" and (start // self.batch_size) % 100 == 0:
                    print(
                        f"Encoded {min(start + self.batch_size, len(texts))}/{len(texts)} passages",
                        flush=True,
                    )
        torch.cuda.synchronize()
        return np.concatenate(results) if results else np.empty((0, 384), dtype=np.float32)

    def profile(self) -> dict[str, object]:
        torch = self.torch
        return {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "license": "MIT",
            "asset_sha256": {key: value[1] for key, value in ASSETS.items()},
            "dtype": "float32",
            "dimension": 384,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "pooling": "attention_mask_mean_then_l2",
            "attention": "eager",
            "tf32": False,
            "gpu": torch.cuda.get_device_name(),
            "cuda_runtime": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "inputs_at_max_length": self.inputs_at_max_length,
            "length_count_scope": (
                "potential truncation, includes all calls; not exact truncated count"
            ),
            "seed": 1729,
            "loading_info": self.loading_info,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--download", action="store_true", help="explicitly fetch ~493 MB of pinned assets"
    )
    args = parser.parse_args(argv)
    if args.download:
        fetch_model(args.output)
    verify_model(args.output)
    print(f"Pinned encoder verified: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
