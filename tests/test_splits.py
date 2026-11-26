import json

import pytest
from pydantic import ValidationError

from temporalproof.data.splits import (
    SplitManifest,
    build_groups,
    create_split,
    main,
    select_queries,
    verify_split,
)
from temporalproof.data.timelyqa import DatasetError, QueryRecord


def record(index, query, passage=None):
    return QueryRecord.model_validate(
        {
            "index": index,
            "category": "synthetic",
            "query_ko": query,
            "answer": "",
            "query_insert_time": "2024-01-01 00:00:00",
            "docs_ko": [{"global_doc_id": index, "text": passage or f"passage {index}"}],
            "gold_docs_final": [0],
            "gold_docs_final_global": [index],
        }
    )


def test_query_templates_and_shared_passages_connect_transitively():
    rows = [
        record(0, "policy in 2023"),
        record(1, "policy in 2024", "shared clause"),
        record(2, "completely different request", " SHARED   clause "),
        record(3, "astronomy explanation"),
    ]
    groups, audit = build_groups(rows)
    assert groups == [[0, 1, 2], [3]]
    assert audit["template_only_matches"] == 1
    assert audit["cross_record_shared_passage_pairs"] == 1


def test_near_query_templates_join_without_relevance_features():
    left = "extended synthetic travel policy reimbursement requirement " * 3
    rows = [record(0, left + "a"), record(1, left + "b")]
    groups, audit = build_groups(rows)
    assert groups == [[0, 1]]
    assert audit["near_query_pairs"] == 1
    rows[0].answer = "changed answer"
    rows[0].query_insert_time = "2030-01-01 00:00:00"
    assert build_groups(rows) == (groups, audit)


def test_split_is_order_independent_and_stable_after_serialization():
    rows = [record(0, "apple"), record(1, "volcano"), record(2, "galaxy")]
    manifest = create_split(rows, "a" * 64)
    assert manifest == create_split(list(reversed(rows)), "a" * 64)
    restored = SplitManifest.model_validate_json(manifest.model_dump_json())
    verify_split(restored, rows, "a" * 64)
    assert sorted(query for key in manifest.groups for query in manifest.query_ids(key)) == [
        0,
        1,
        2,
    ]


def test_verify_split_rejects_changed_corpus_and_assignment():
    rows = [record(0, "apple"), record(1, "volcano")]
    manifest = create_split(rows, "a" * 64)
    with pytest.raises(DatasetError, match="split"):
        verify_split(manifest, rows, "b" * 64)
    source = next(key for key, groups in manifest.groups.items() if groups)
    target = next(key for key in manifest.groups if key != source)
    manifest.groups[target].append(manifest.groups[source].pop())
    with pytest.raises(DatasetError, match="split"):
        verify_split(manifest, rows, "a" * 64)


def test_manifest_rejects_cross_partition_query_overlap():
    manifest = create_split([record(0, "apple")], "a" * 64).model_dump()
    manifest["groups"]["validation"] = [[0]]
    manifest["groups"]["test"] = [[0]]
    with pytest.raises(ValidationError):
        SplitManifest.model_validate_json(json.dumps(manifest))


def test_subset_selection_is_deterministic_partition_scoped_and_not_silently_shortened():
    manifest = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={"train": [[0]], "validation": [[1], [2], [3]], "test": [[4]]},
        audit={},
    )
    selected = select_queries(manifest, "validation", 2, 1729)
    assert selected == select_queries(manifest, "validation", 2, 1729)
    assert set(selected) <= {1, 2, 3}
    with pytest.raises(DatasetError):
        select_queries(manifest, "validation", 4, 1729)


def test_duplicate_record_ids_fail_before_grouping():
    with pytest.raises(DatasetError):
        build_groups([record(0, "apple"), record(0, "pear")])


def test_grouping_does_not_depend_on_which_passage_is_gold():
    rows = [record(0, "apple"), record(1, "volcano")]
    rows[0].docs_ko.append(rows[1].docs_ko[0].model_copy(update={"global_doc_id": 99}))
    before = build_groups(rows)
    rows[0].gold_docs_final = [1]
    rows[0].gold_docs_final_global = [99]
    assert build_groups(rows) == before


def test_split_cli_is_offline_reproducible_and_refuses_replacement(tmp_path, monkeypatch):
    rows = [record(0, "apple"), record(1, "volcano")]
    monkeypatch.setattr("temporalproof.data.splits.load_archive_records", lambda path: rows)
    output = tmp_path / "split.json"
    args = ["--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == original
    assert main([*args, "--seed", "1730"]) == 1
    assert output.read_bytes() == original
