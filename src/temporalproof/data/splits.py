"""Deterministic connected query groups, without using relevance labels."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.data.source import ARCHIVE_SHA256, DEFAULT_ARCHIVE, load_archive_records
from temporalproof.data.timelyqa import DatasetError, Identifier, QueryRecord

POLICY = "query_template_char3_jaccard90_shared_exact_passages_v1"
DEFAULT_SPLIT = Path("data/splits/company_policies_v1.json")
Partition = Literal["train", "validation", "test"]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def query_template(text: str) -> str:
    return re.sub(r"\d+", "<number>", normalize(text))


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy: Literal[POLICY] = POLICY
    seed: Identifier
    allocation: Literal["sha256_mod10000_7000_1500_1500_v1"] = "sha256_mod10000_7000_1500_1500_v1"
    groups: dict[Partition, list[list[Identifier]]]
    audit: dict[str, int]

    @model_validator(mode="after")
    def check_partitions(self) -> Self:
        if set(self.groups) != {"train", "validation", "test"}:
            raise ValueError("all three partitions are required")
        flattened = [
            query for groups in self.groups.values() for group in groups for query in group
        ]
        if not flattened or len(set(flattened)) != len(flattened):
            raise ValueError("query IDs must be nonempty and disjoint")
        if any(not group for groups in self.groups.values() for group in groups):
            raise ValueError("empty query group")
        return self

    def query_ids(self, partition: Partition) -> list[int]:
        return sorted(query for group in self.groups[partition] for query in group)


def build_groups(records: Sequence[QueryRecord]) -> tuple[list[list[int]], dict[str, int]]:
    """Conservatively connect duplicates/templates, near queries and shared passages.

    Similarity uses character-trigram set Jaccard >= 0.9 on number-masked query
    templates. This does not establish semantic equivalence or true document families.
    All ten associated passages participate, regardless of which are gold.
    """
    rows = sorted(records, key=lambda record: record.index)
    if not rows or len({record.index for record in rows}) != len(rows):
        raise DatasetError("query record IDs must be nonempty and unique")
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        a, b = root(left), root(right)
        parent[max(a, b)] = min(a, b)

    seen_queries: dict[str, int] = {}
    seen_templates: dict[str, int] = {}
    seen_passages: dict[str, int] = {}
    postings: dict[str, list[int]] = defaultdict(list)
    signatures: list[set[str]] = []
    templates: list[str] = []
    audit = Counter(exact_query_matches=0, template_only_matches=0, near_query_pairs=0)
    passage_pairs: set[tuple[int, int]] = set()
    for i, record in enumerate(rows):
        text = normalize(record.query_ko)
        template = query_template(record.query_ko)
        if text in seen_queries:
            join(i, seen_queries[text])
            audit["exact_query_matches"] += 1
        else:
            seen_queries[text] = i
        if template in seen_templates:
            previous = seen_templates[template]
            join(i, previous)
            audit["template_only_matches"] += text != normalize(rows[previous].query_ko)
        else:
            seen_templates[template] = i
        signature = {template[n : n + 3] for n in range(max(1, len(template) - 2))}
        overlaps = Counter(j for gram in signature for j in postings[gram])
        for j, intersection in overlaps.items():
            union = len(signature) + len(signatures[j]) - intersection
            if template != templates[j] and intersection * 10 >= union * 9:
                join(i, j)
                audit["near_query_pairs"] += 1
        for gram in signature:
            postings[gram].append(i)
        signatures.append(signature)
        templates.append(template)
        for doc in record.docs_ko:
            digest = hashlib.sha256(normalize(doc.text).encode()).hexdigest()
            if digest in seen_passages:
                previous = seen_passages[digest]
                if previous != i:
                    join(i, previous)
                    passage_pairs.add((previous, i))
            else:
                seen_passages[digest] = i
    components: dict[int, list[int]] = defaultdict(list)
    for i, record in enumerate(rows):
        components[root(i)].append(record.index)
    groups = sorted(components.values())
    audit.update(
        record_count=len(rows),
        group_count=len(groups),
        largest_group=max(map(len, groups)),
        cross_record_shared_passage_pairs=len(passage_pairs),
    )
    return groups, dict(sorted(audit.items()))


def create_split(
    records: Sequence[QueryRecord], dataset_sha256: str, seed: int = 1729
) -> SplitManifest:
    groups, audit = build_groups(records)
    partitions: dict[Partition, list[list[int]]] = {"train": [], "validation": [], "test": []}
    for group in groups:
        key = f"{seed}:" + ",".join(map(str, group))
        bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10000
        partition = "train" if bucket < 7000 else "validation" if bucket < 8500 else "test"
        partitions[partition].append(group)
    return SplitManifest(dataset_sha256=dataset_sha256, seed=seed, groups=partitions, audit=audit)


def verify_split(
    manifest: SplitManifest, records: Sequence[QueryRecord], dataset_sha256: str
) -> None:
    """Recompute grouping/allocation; reject stale IDs, overlap or corpus mismatch."""
    expected = create_split(records, dataset_sha256, manifest.seed)
    if manifest != expected:
        raise DatasetError("split does not match dataset, grouping policy or seed")


def select_queries(
    manifest: SplitManifest, partition: Partition, limit: int, seed: int
) -> list[int]:
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("query limit must be positive")
    ids = manifest.query_ids(partition)
    if limit > len(ids):
        raise DatasetError("query limit exceeds available partition; refusing silent truncation")
    return sorted(
        ids, key=lambda query: (hashlib.sha256(f"{seed}:query:{query}".encode()).hexdigest(), query)
    )[:limit]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        records = load_archive_records(args.archive)
        manifest = create_split(records, ARCHIVE_SHA256, args.seed)
        content = json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n"
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != content:
                raise DatasetError("existing split differs; choose a new versioned output path")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
    except (OSError, ValueError, KeyError) as exc:
        message = str(exc) if isinstance(exc, DatasetError) else "input or split validation failed"
        print(f"split failed: {message}", file=sys.stderr)
        return 1
    print(f"Split: {args.output}")
    print(
        json.dumps({key: len(manifest.query_ids(key)) for key in manifest.groups}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
