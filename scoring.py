"""Deterministic field scoring; unknown reference values are not graded."""
import json
import re
import unicodedata
from datetime import date

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

FIELDS = {
    "Species name": "Spezies Label",
    "Collection date": "Sammeldatum",
    "Collector's name": "Sammler",
    "Country/State": "Geographische_Zuordnung",
    "Location": "Fundort Label",
    "Region": "Naturraum",
    "Notes": "Bemerkung_zur_Pflanze",
}
MONTHS = {
    name: index for index, name in enumerate(
        ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
         "september", "oktober", "november", "dezember"], start=1
    )
}


def normalize_date(value: str) -> str:
    """Normalize supported German/ISO dates without inventing missing components."""
    match = re.fullmatch(r"(?:(\d{1,2})\.\s*)?([a-zä]+)\s+(\d{4})", value)
    if match and match[2] in MONTHS:
        day, month, year = match[1], MONTHS[match[2]], int(match[3])
        try:
            return date(year, month, int(day)).isoformat() if day else f"{year:04d}-{month:02d}"
        except ValueError:
            return value
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            return value
    return value


def normalize(field: str, value: str) -> str:
    value = " ".join(unicodedata.normalize("NFC", value).casefold().split())
    if field == "Species name":
        value = re.sub(r"\s*×\s*", " x ", value)
    if field == "Collection date":
        value = normalize_date(value)
    return value.strip()


def parse_answer(text: str) -> dict[str, str]:
    """Require exactly the requested string-valued fields, without JSON repairs."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    answer = json.loads(text, object_pairs_hook=unique_pairs)
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError("Response must be a JSON object with exactly the seven requested fields")
    if not all(isinstance(value, str) for value in answer.values()):
        raise ValueError("Every field must contain a string (use an empty string for unknowns)")
    return answer


@scorer(metrics={key: [mean()] for key in ["valid_json", *FIELDS]})
def label_fields():
    async def score(state: TaskState, target: Target) -> Score:
        reference = json.loads(target.text)
        error = None
        try:
            answer = parse_answer(state.output.completion)
        except (ValueError, TypeError) as exc:
            answer = None
            error = str(exc)
        values = {"valid_json": int(answer is not None)}
        excluded = []
        for field in FIELDS:
            expected = normalize(field, reference[field])
            if not expected:
                values[field] = float("nan")  # Inspect excludes NaN from aggregate metrics.
                excluded.append(field)
            else:
                values[field] = int(answer is not None and normalize(field, answer[field]) == expected)
        return Score(
            value=values,
            answer=state.output.completion,
            explanation=error or "Exact field matches after case, whitespace, hybrid-symbol and date normalization.",
            metadata={"unscored_fields": excluded, "blank_reference_policy": "unknown"},
        )

    return score
