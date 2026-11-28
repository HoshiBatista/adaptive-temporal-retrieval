from datetime import date

from temporalproof.evaluation.exact_start import match_ranks, promote, query_day
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.query_time import analyze_query_time


def test_query_day_requires_day_precision_and_policy_role():
    assert query_day(analyze_query_time("2031년 1월 1일부터 시행되는 정책은?")) == date(2031, 1, 1)
    for query in (
        "2031년 1월 1일 현재 사용 가능한 휴가는?",
        "2031년 1월부터 시행되는 정책은?",
        "2031년 1월 1일 12시부터 시행되는 정책은?",
        "2031년 1월 1일 입사한 직원의 휴가는?",
    ):
        assert query_day(analyze_query_time(query)) is None


def test_promote_swaps_only_topmost_match_and_keeps_scores():
    hits = [Hit(i, 100 - i) for i in range(1, 24)]
    target = date(2031, 1, 1)
    dates = {4: target, 6: target, 9: date(2031, 1, 2)}
    matches = match_ranks(hits, target, dates)
    assert matches == [4, 6]
    assert promote(hits, matches) == hits[:2] + [hits[3], hits[2]] + hits[4:]
    assert promote(hits, [20]) == hits[:18] + [hits[19], hits[18]] + hits[20:]
    for unchanged in ([1, 4], [21], match_ranks(hits, None, dates)):
        assert promote(hits, unchanged) == hits
