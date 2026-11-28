"""Extract explicit rule-start statements from passage text."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from temporalproof.retrieval.query_time import DateMention, analyze_query_time

RULE = r"(?:규정|정책|방침|개정안|내규|지침|제도|법령)"
VERB = r"(?:시행|적용|발효)"
START = r"(?:부터|부로|자로)"
SUBJECT = re.compile(rf"{RULE}\s*(?:은|는|이|가)\s*$")
HEADER = re.compile(rf"{VERB}\s*(?:일자|일|날짜)\s*[:=]\s*$")
ACTIVE = r"(?:합니다|한다|됩니다|된다|되었습니다|되었다|하고|하며|되고|되며)"
POST = re.compile(rf"^\s*{START}\s*{VERB}(?:되는|된|하는)\s+[^,.?!;\n]{{0,36}}?{RULE}")
DECLARATION = re.compile(rf"^\s*{START}\s*{VERB}{ACTIVE}")
BARE = re.compile(rf"^\s*{START}\s*(?:시행|발효)(?:{ACTIVE})?\s*$")
BOUNDARY = re.compile(r"[.!?;\n]")
RANGE = re.compile(r"까지|~")
DATE_SEPARATOR = re.compile(r"\s*[-–—]\s*")
CONDITIONAL = re.compile(r"경우|가정|예정|계획|여부|검토|수\s*있|없다|없습니다")
Kind = Literal["start_statement", "range_or_qualified", "other_date"]


@dataclass(frozen=True)
class DocumentDate:
    mention: DateMention
    kind: Kind


@dataclass(frozen=True)
class DocumentTime:
    dates: tuple[DocumentDate, ...]
    start: datetime | None
    reason: str


def analyze_document_time(text: str) -> DocumentTime:
    """Return a day-resolution anchor for a unique explicit start statement."""
    parsed = analyze_query_time(text)
    normalized = parsed.normalized_query
    dates = []
    for index, mention in enumerate(parsed.mentions):
        preceding = normalized[: mention.start]
        following = normalized[mention.end :]
        before = BOUNDARY.split(preceding)[-1]
        boundary = BOUNDARY.search(following)
        after = following[: boundary.start()] if boundary else following
        next_date = parsed.mentions[index + 1] if index + 1 < len(parsed.mentions) else None
        adjacent = after[: next_date.start - mention.end] if next_date else after
        cue = (
            POST.match(adjacent)
            or (SUBJECT.search(before) and DECLARATION.match(adjacent))
            or HEADER.search(before)
            or (not before.strip() and BARE.fullmatch(adjacent))
        )
        question = boundary is not None and boundary.group() == "?"
        qa = bool(re.match(r"\s*(?:q|a|질문|답변)\s*:", before, re.IGNORECASE))
        # A range can contain non-start dates too. Keep its endpoints as mentions.
        between_dates = next_date is not None and DATE_SEPARATOR.fullmatch(
            normalized[mention.end : next_date.start]
        )
        range_bound = (
            between_dates or RANGE.search(adjacent[:16]) or mention.relation_marker == "까지"
        )
        qualified = cue and (
            question
            or qa
            or CONDITIONAL.search(before + after)
            or re.match(r"\s*(?:이전|이후)", adjacent)
        )
        kind: Kind = "other_date"
        if range_bound or qualified:
            kind = "range_or_qualified"
        elif cue:
            kind = "start_statement"
        dates.append(DocumentDate(mention, kind))

    candidates = [d.mention for d in dates if d.kind == "start_statement"]
    if any(d.kind == "range_or_qualified" for d in dates):
        reason = "range_or_qualified"
    elif not candidates:
        reason = "other_dates_only" if dates else "no_dates"
    elif any(m.status != "explicit" for m in candidates):
        reason = "unknown_start"
    elif any(m.precision != "day" for m in candidates):
        reason = "non_day_start"
    elif len({m.lower for m in candidates}) != 1:
        reason = "multiple_start_dates"
    else:
        return DocumentTime(tuple(dates), candidates[0].lower, "unique_day_start")
    return DocumentTime(tuple(dates), None, reason)
