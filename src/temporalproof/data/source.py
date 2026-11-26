"""Pinned, size-bounded download, explicit opt-in only."""

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from temporalproof.data.timelyqa import DatasetError, QueryRecord, iter_records

REVISION = "5213a0c1743f23fd52a258005c3887aaf7e8cd0f"
ARCHIVE_SHA256 = "a006481ffdf723d4b15554da2c621ebdfefd59b78ad37c58a633b3b71de247ad"
ARCHIVE_URL = f"https://raw.githubusercontent.com/kaist-dmlab/TimelyRAG/{REVISION}/datasets.zip"
MEMBER = "timelyrag_company_policies_dataset_v2.jsonl"
DEFAULT_ARCHIVE = Path("data/raw/timelyqa") / REVISION / "datasets.zip"
MAX_ARCHIVE_BYTES = 8_000_000
MAX_MEMBER_BYTES = 25_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise DatasetError("archive exceeds size limit")
    if sha256_file(path) != ARCHIVE_SHA256:
        raise DatasetError("archive checksum mismatch; original input was not modified")


def load_archive_records(path: Path) -> list[QueryRecord]:
    """Load only the pinned member; never extract or download anything."""
    verify_archive(path)
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(MEMBER)
        if member.file_size > MAX_MEMBER_BYTES:
            raise DatasetError("JSONL member exceeds size limit")
        with archive.open(member) as stream, io.TextIOWrapper(stream, encoding="utf-8") as text:
            return list(iter_records(text))


def download_archive(destination: Path) -> Path:
    """Verify cached inputs; never replace a mismatching existing file."""
    if destination.exists():
        verify_archive(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with urlopen(ARCHIVE_URL, timeout=30) as response:
            content = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(content) > MAX_ARCHIVE_BYTES:
            raise DatasetError("download exceeds size limit")
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise DatasetError("download checksum mismatch")
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        # Hard-link publication refuses to overwrite a concurrently created input.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
