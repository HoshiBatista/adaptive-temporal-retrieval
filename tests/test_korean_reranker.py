"""Pinned-model selection tests require neither CUDA nor Transformers imports."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.reranking import RerankingConfig
from temporalproof.retrieval import reranker


@pytest.mark.parametrize("model", [reranker.MINILM, reranker.KOREAN])
def test_registered_model_config_selects_exact_revision_and_default_asset_directory(model):
    config = json.loads(Path("configs/experiments/m3_company_policies.json").read_text())
    config.update(model_id=model.model_id, revision=model.revision)
    validated = RerankingConfig.model_validate(config)
    selected = reranker.model_spec(validated.model_id)
    assert selected is model and selected.directory.name == validated.revision
    config["revision"] = "a" * 40
    with pytest.raises(ValidationError, match="pinned specification"):
        RerankingConfig.model_validate(config)


def test_unsupported_model_and_mixed_model_revision_are_rejected():
    with pytest.raises(DatasetError, match="unsupported"):
        reranker.model_spec("unregistered/remote-code-model")
    config = json.loads(Path("configs/experiments/m3_korean_company_policies.json").read_text())
    config["revision"] = reranker.MINILM.revision
    with pytest.raises(ValidationError, match="pinned specification"):
        RerankingConfig.model_validate(config)


def test_korean_model_manifest_and_protocol_are_pinned():
    manifest = json.loads(Path("configs/models/bge_reranker_v2_m3_ko.json").read_text())
    spec = reranker.KOREAN
    assert manifest["model_id"] == spec.model_id and manifest["revision"] == spec.revision
    assert manifest["assets"] == {name: list(asset) for name, asset in spec.assets.items()}
    config = RerankingConfig.model_validate_json(
        Path("configs/experiments/m3_korean_company_policies.json").read_text()
    )
    old = RerankingConfig.model_validate_json(
        Path("configs/experiments/m3_company_policies.json").read_text()
    )
    assert config.baseline.model_dump(exclude={"name"}) == old.baseline.model_dump(exclude={"name"})
    assert config.candidate_manifest_sha256 == old.candidate_manifest_sha256
    assert config.max_length == old.max_length == 512
    assert config.batch_size == old.batch_size == 16


def test_selected_model_assets_are_verified_before_any_gpu_or_network_use(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    with pytest.raises(DatasetError, match="asset missing"):
        reranker.CrossEncoder(tmp_path, spec=reranker.KOREAN)


def test_explicit_download_uses_selected_model_revision_and_checks_bytes(tmp_path, monkeypatch):
    payload = b"independently authored synthetic model"
    spec = replace(
        reranker.KOREAN,
        assets={"model.safetensors": (len(payload), hashlib.sha256(payload).hexdigest())},
    )
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(reranker, "urlopen", download)
    reranker.fetch_model(tmp_path, spec)
    reranker.verify_model(tmp_path, spec)
    assert urls == [
        f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/model.safetensors"
    ]
    reranker.fetch_model(tmp_path, spec)
    assert len(urls) == 1
    (tmp_path / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(DatasetError, match="not replaced"):
        reranker.fetch_model(tmp_path, spec)


def test_explicit_bad_download_is_removed_and_never_promoted(tmp_path, monkeypatch):
    spec = replace(
        reranker.KOREAN, assets={"model.safetensors": (4, hashlib.sha256(b"good").hexdigest())}
    )
    monkeypatch.setattr(reranker, "urlopen", lambda *a, **kw: io.BytesIO(b"bad!"))
    with pytest.raises(DatasetError, match="checksum"):
        reranker.fetch_model(tmp_path, spec)
    assert list(tmp_path.iterdir()) == []
