"""Strict adapter for the observed TimelyQABench v2 Company Policies schema.

No event times, validity intervals, versions, or time zones are inferred.
"""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self, TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Identifier = Annotated[int, Field(strict=True, ge=0)]
Text = Annotated[str, Field(strict=True, min_length=1)]
TimestampText = Annotated[str, Field(strict=True)]


class DatasetError(ValueError):
    """Invalid input; messages deliberately exclude source text."""


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse reference-format dates; missing/malformed strings remain unknown.

    Raw timestamp strings survive in the input models. The audit reports malformed
    values separately from absent/empty values; it never repairs them silently.
    """
    if value is None or value == "":
        return None
    try:
        result = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if result.strftime("%Y-%m-%d %H:%M:%S") != value:
        return None
    return result


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_doc_id: Identifier
    text: Text
    doc_insert_time: TimestampText | None = None
    title: Text | None = None


class QueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: Identifier
    category: Text
    query_ko: Text
    answer: Annotated[str, Field(strict=True)]
    query_insert_time: TimestampText | None = None
    docs_ko: list[Document] = Field(min_length=1)
    gold_docs_final: list[Identifier] = Field(min_length=1)
    gold_docs_final_global: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        local = self.gold_docs_final
        global_ids = self.gold_docs_final_global
        if len(set(local)) != len(local) or len(set(global_ids)) != len(global_ids):
            raise ValueError("duplicate relevance labels")
        if len({doc.global_doc_id for doc in self.docs_ko}) != len(self.docs_ko):
            raise ValueError("duplicate document IDs within record")
        if any(index >= len(self.docs_ko) for index in local):
            raise ValueError("local gold index outside docs_ko")
        if {self.docs_ko[index].global_doc_id for index in local} != set(global_ids):
            raise ValueError("local and global relevance labels disagree")
        return self


def iter_records(stream: TextIO) -> Iterator[QueryRecord]:
    """Read JSONL with sanitized line/field errors; never print raw inputs."""
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            yield QueryRecord.model_validate_json(line)
        except ValidationError as exc:
            allowed = set(QueryRecord.model_fields) | set(Document.model_fields)
            locations = [
                ".".join(
                    str(part) if isinstance(part, int) or part in allowed else "<unknown_field>"
                    for part in error["loc"]
                )
                or "record"
                for error in exc.errors()[:8]
            ]
            raise DatasetError(
                f"line {line_number}: invalid schema at {', '.join(locations)}"
            ) from None


def load_records(path: Path) -> list[QueryRecord]:
    with path.open(encoding="utf-8") as stream:
        return list(iter_records(stream))
