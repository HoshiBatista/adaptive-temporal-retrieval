"""Text-only temporal mentions and conservative role cues; no ground-truth validity."""

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RULE_VERSION = "ko_query_date_roles_v1"
Precision = Literal["year", "month", "day", "hour", "minute", "second"]
Role = Literal["employment", "policy", "as_of", "event"]

# Longest non-overlapping explicit spans win, including invalid calendar mentions.
YEAR = r"(?<!\d)(?P<year>\d{4})"
KO_MONTH = YEAR + r"\s*년\s*(?P<month>\d{1,2})\s*월"
KO_DAY = KO_MONTH + r"\s*(?P<day>\d{1,2})\s*일"
FORMATS = [
    (
        "ko_datetime",
        re.compile(KO_DAY + r"\s*(?P<hour>\d{1,2})\s*시(?:\s*(?P<minute>\d{1,2})\s*분)?"),
    ),
    (
        "iso_datetime",
        re.compile(
            YEAR
            + (
                r"[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"
                r"[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            )
        ),
    ),
    ("ko_day", re.compile(KO_DAY)),
    ("iso_day", re.compile(YEAR + r"[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})(?!\d)")),
    ("ko_month", re.compile(KO_MONTH)),
    ("iso_month", re.compile(YEAR + r"[-/.](?P<month>\d{1,2})(?!\d)")),
    ("ko_year", re.compile(YEAR + r"\s*년")),
    ("yearless_day", re.compile(r"(?<!\d)(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일")),
    ("yearless_month", re.compile(r"(?<!\d)(?P<month>\d{1,2})\s*월")),
]
EMPLOYMENT = re.compile(r"입사|채용|입학|취업|(?:정규직|계약직)(?:으로)?\s*전환|근무(?:를)?\s*시작")
EVENT = re.compile(
    r"휴직(?:을|하|한|할|했|신청|\s*신청)|휴가.{0,6}(?:사용|신청)|복직(?:하|한|할|했|시|일)|퇴사(?:하|한|할|일)|퇴직(?:하|한|할|일)"
)
POLICY_VERB = re.compile(r"적용|시행|발효|개정|변경")
POLICY_NOUN = re.compile(r"정책|규정|기준|방침|개정안")
ASOF = re.compile(r"현재|기준|시점|당시")
COMPARISON = re.compile(r"비교|차이|전후|변경\s*전|변경\s*후|이전과|이후와")
RELATION = re.compile(r"^\s*(이후|이전|부터|까지|부로|자로)")


class DateMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text: str
    format: str
    precision: Precision
    status: Literal["explicit", "missing_year", "invalid_calendar", "unsupported_timezone"]
    lower: datetime | None
    upper_exclusive: datetime | None
    role_cues: tuple[Role, ...]
    role_conflict: bool
    relation_marker: str | None

    @model_validator(mode="after")
    def bounds(self):
        if self.end <= self.start:
            raise ValueError("invalid mention span")
        if self.status == "explicit":
            if (
                self.lower is None
                or self.upper_exclusive is None
                or self.lower >= self.upper_exclusive
            ):
                raise ValueError("explicit mention requires increasing resolution bounds")
        elif self.lower is not None or self.upper_exclusive is not None:
            raise ValueError("unknown/invalid mentions cannot acquire date bounds")
        return self


class QueryTimeSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_version: Literal[RULE_VERSION] = RULE_VERSION
    normalized_query: str
    offset_scope: Literal["NFKC_normalized_query"] = "NFKC_normalized_query"
    mentions: tuple[DateMention, ...]
    action: Literal["use_explicit_point", "keep_fast_ranking"]
    identified_target_index: int | None
    selected_index: int | None
    selected_time: datetime | None
    reason: str

    @model_validator(mode="after")
    def consistent_selection(self):
        previous_end = 0
        for mention in self.mentions:
            if (
                mention.start < previous_end
                or mention.end > len(self.normalized_query)
                or self.normalized_query[mention.start : mention.end] != mention.text
            ):
                raise ValueError("mentions must be nonoverlapping normalized query spans")
            previous_end = mention.end
        for index in [self.selected_index, self.identified_target_index]:
            if index is not None and not 0 <= index < len(self.mentions):
                raise ValueError("selected mention index outside query")
        if self.action == "keep_fast_ranking":
            if self.selected_index is not None or self.selected_time is not None:
                raise ValueError("abstention cannot carry a selected point")
        else:
            if self.selected_index is None:
                raise ValueError("explicit selection requires a mention")
            mention = self.mentions[self.selected_index]
            if (
                mention.status != "explicit"
                or mention.precision in {"year", "month"}
                or mention.role_conflict
                or not {"as_of", "policy"} & set(mention.role_cues)
                or self.selected_time != mention.lower
                or self.identified_target_index != self.selected_index
            ):
                raise ValueError("selected point must follow the explicit target contract")
        return self


def extent(fields: dict) -> tuple[Precision, str, datetime | None, datetime | None]:
    values = {k: int(v) for k, v in fields.items() if v is not None}
    precision = next(k for k in ["second", "minute", "hour", "day", "month", "year"] if k in values)
    if "year" not in values:
        return precision, "missing_year", None, None
    try:
        lower = datetime(
            values["year"],
            values.get("month", 1),
            values.get("day", 1),
            values.get("hour", 0),
            values.get("minute", 0),
            values.get("second", 0),
        )
        if precision == "year":
            upper = datetime(lower.year + 1, 1, 1)
        elif precision == "month":
            upper = datetime(lower.year + int(lower.month == 12), lower.month % 12 + 1, 1)
        else:
            upper = lower + timedelta(
                **{
                    {"day": "days", "hour": "hours", "minute": "minutes", "second": "seconds"}[
                        precision
                    ]: 1
                }
            )
    except (ValueError, OverflowError):
        return precision, "invalid_calendar", None, None
    return precision, "explicit", lower, upper


def role_cues(before: str, after: str) -> tuple[tuple[Role, ...], bool]:
    cues = []
    employment = EMPLOYMENT.search(after)
    event = EVENT.search(after)
    asof = ASOF.search(after)
    if (employment is not None and employment.start() <= 16) or re.search(
        r"(?:입사|채용|전환)(?:일|날짜|시점)?(?:은|는|이|가|:)?\s*$", before
    ):
        cues.append("employment")
    if (asof is not None and asof.start() <= 6) or re.search(
        r"(?:기준|현재|시점|당시)(?:일|날짜|은|는|이|가|:)?\s*$", before
    ):
        cues.append("as_of")
    if event is not None and event.start() <= 20:
        cues.append("event")
    verb = POLICY_VERB.search(after)
    noun = POLICY_NOUN.search(after) or POLICY_NOUN.search(before)
    intervening_actor = any(
        m is not None and m.start() < (verb.start() if verb else 0) for m in [employment, event]
    )
    if "as_of" not in cues and (
        (verb is not None and verb.start() <= 20 and noun is not None and not intervening_actor)
        or re.search(r"(?:적용|시행|발효|개정)(?:일|날짜|시점)?(?:은|는|이|가|:)?\s*$", before)
    ):
        cues.append("policy")
    conflict = ("employment" in cues and bool({"as_of", "policy"} & set(cues))) or (
        "policy" in cues and "event" in cues
    )
    return tuple(cues), conflict


def analyze_query_time(
    query: str, *, before_characters: int = 16, after_characters: int = 32
) -> QueryTimeSelection:
    """Calendar resolution bounds are uncertainty extents, never document validity intervals.

    No insertion-time backfill for missing years. Employment/event-only dates are not
    promoted to query targets. Coarse/ambiguous target dates trigger a fast-path bypass.
    """
    if before_characters < 1 or after_characters < 1:
        raise ValueError("positive context windows required")
    text = unicodedata.normalize("NFKC", query)
    matches = [
        (m.start(), m.end(), priority, name, m.groupdict())
        for priority, (name, pattern) in enumerate(FORMATS)
        for m in pattern.finditer(text)
    ]
    spans = []
    for start, end, priority, name, fields in sorted(
        matches, key=lambda v: (v[0], -(v[1] - v[0]), v[2])
    ):
        if any(start < b and end > a for a, b, *_ in spans):
            continue
        spans.append((start, end, priority, name, fields))
    mentions = []
    for index, (start, end, _, name, fields) in enumerate(spans):
        left = spans[index - 1][1] if index else 0
        right = spans[index + 1][0] if index + 1 < len(spans) else len(text)
        before = text[max(left, start - before_characters) : start]
        after = text[end : min(right, end + after_characters)]
        before = re.split(r"[,.;?!]", before)[-1]
        after = re.split(r"[,.;?!]", after)[0]
        roles, conflict = role_cues(before, after)
        precision, status, lower, upper = extent(fields)
        if precision in {"hour", "minute", "second"} and re.match(
            r"^\s*(?:Z|[+-]\d{2}:?\d{2}(?!\d))", text[end:]
        ):
            status, lower, upper = "unsupported_timezone", None, None
        marker = RELATION.search(after)
        mentions.append(
            DateMention(
                start=start,
                end=end,
                text=text[start:end],
                format=name,
                precision=precision,
                status=status,
                lower=lower,
                upper_exclusive=upper,
                role_cues=roles,
                role_conflict=conflict,
                relation_marker=marker.group(1) if marker else None,
            )
        )

    def result(reason, identified=None, selected=None):
        return QueryTimeSelection(
            normalized_query=text,
            mentions=tuple(mentions),
            reason=reason,
            action="use_explicit_point" if selected is not None else "keep_fast_ranking",
            identified_target_index=identified,
            selected_index=selected,
            selected_time=mentions[selected].lower if selected is not None else None,
        )

    if COMPARISON.search(text):
        return result("comparison_request")
    target = [i for i, m in enumerate(mentions) if "as_of" in m.role_cues]
    role = "as_of"
    if not target:
        target = [i for i, m in enumerate(mentions) if "policy" in m.role_cues]
        role = "policy"
    if not target:
        return result("no_target_role_cue")
    if any(mentions[i].role_conflict for i in target):
        return result("conflicting_role_cues")
    if any(mentions[i].status != "explicit" for i in target):
        return result("unknown_or_invalid_target", target[0] if len(target) == 1 else None)
    values = {(mentions[i].lower, mentions[i].upper_exclusive) for i in target}
    if len(values) != 1:
        return result("multiple_target_values")
    chosen = target[0]
    if mentions[chosen].precision in {"year", "month"}:
        return result("coarse_target_precision", chosen)
    if mentions[chosen].relation_marker in {"이전", "까지"} or (
        role == "as_of" and mentions[chosen].relation_marker in {"이후", "부터"}
    ):
        return result("non_point_target_relation", chosen)
    return result(f"unique_{role}_precise_target", chosen, chosen)
