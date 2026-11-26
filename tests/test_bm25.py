import pytest

from temporalproof.data.timelyqa import Document
from temporalproof.retrieval.bm25 import BM25Retriever


def documents():
    return [
        Document(global_doc_id=30, text="identical policy text"),
        Document(global_doc_id=10, text="identical policy text"),
        Document(global_doc_id=20, text="astronomy galaxy"),
    ]


def test_equal_scores_use_document_id_and_ignore_corpus_input_order():
    first = BM25Retriever(documents()).search("unknown-token", 100)
    second = BM25Retriever(list(reversed(documents()))).search("unknown-token", 100)
    assert first == second
    assert [hit.doc_id for hit in first] == [10, 20, 30]
    assert all(hit.score == 0.0 for hit in first)


def test_search_returns_lexical_hit_without_gold_labels():
    hits = BM25Retriever(documents()).search("galaxy", 1)
    assert hits[0].doc_id == 20


def test_whitespace_tokenization_preserves_case_and_punctuation():
    index = BM25Retriever(documents())
    assert index.search("GALAXY", 3) == index.search("galaxy!", 3)
    assert index.search("   ", 10) == []


def test_duplicate_ids_and_empty_corpora_fail():
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever([documents()[0], documents()[0]])
    with pytest.raises(ValueError):
        BM25Retriever([])
    with pytest.raises(ValueError, match="token"):
        BM25Retriever([Document(global_doc_id=0, text=" ")])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_fails(k):
    with pytest.raises(ValueError):
        BM25Retriever(documents()).search("policy", k)
