"""Deterministic, field-specific scoring against explicit or provisional references."""
import json
import re
import unicodedata
from datetime import date

from inspect_ai.scorer import Score, Target, mean, scorer

# Original catalogue columns. Keep these separate from the model's output schema.
FIELDS = {
    "Species name": "Spezies Label",
    "Collection date": "Sammeldatum",
    "Collector's name": "Sammler",
    "Country/State": "Geographische_Zuordnung",
    "Location": "Fundort Label",
    "Region": "Naturraum",
    "Notes": "Bemerkung_zur_Pflanze",
}
OUTPUT_FIELDS = ("Species name", "Species author", "Collection date", "Collector's name",
                 "Country", "State", "Location", "Region", "Notes")
FACT_FIELDS = {"Location", "Notes"}
MONTHS = {name: i for i, name in enumerate(
    ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
     "september", "oktober", "november", "dezember"], 1)}
STATUSES = {"present", "absent", "unknown", "unreadable", "uncertain"}
METRICS = ["valid_json", *OUTPUT_FIELDS, "field_accuracy", "specimen_exact",
           "date_year", "date_month", "date_day", "omitted_facts", "unsupported_additions",
           "Location_precision", "Location_recall", "Notes_precision", "Notes_recall"]


def normalize_date(value):
    match = re.fullmatch(r"(?:(\d{1,2})\.\s*)?([a-zä]+)\s+(\d{4})", value)
    if match and match[2] in MONTHS:
        day, month, year = match[1], MONTHS[match[2]], int(match[3])
        try:
            return date(year, month, int(day)).isoformat() if day else f"{year:04d}-{month:02d}"
        except ValueError:
            return value
    match = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", value)
    if match:
        year, month, day = match.groups()
        try:
            parsed = date(int(year), int(month), int(day or 1))
            return parsed.isoformat() if day else parsed.isoformat()[:7]
        except ValueError:
            pass
    return value


def normalize(field, value):
    value = " ".join(unicodedata.normalize("NFC", value).casefold().split())
    if field == "Species name":
        value = re.sub(r"\s*×\s*", " x ", value)
    if field == "Collection date":
        value = normalize_date(value)
    if field in FACT_FIELDS:
        # Facts ignore word order and punctuation, but retain every word and number.
        return " ".join(sorted(re.findall(r"\w+", value)))
    return value.strip()


def parse_fields(text):
    """Salvage valid fields from a JSON object; never repair invalid JSON."""
    class ParsedObject(dict):
        pass

    def pairs(items):
        result = ParsedObject()
        result.duplicates = set()
        for key, value in items:
            if key in result:
                result.duplicates.add(key)
            result[key] = value
        return result

    try:
        raw = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        return {}, False
    if not isinstance(raw, dict):
        return {}, False
    usable = {}
    for field in OUTPUT_FIELDS:
        value = raw.get(field)
        valid = (isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
                 if field in FACT_FIELDS else isinstance(value, str))
        if valid and field not in raw.duplicates:
            usable[field] = value
    return usable, set(raw) == set(OUTPUT_FIELDS) and len(usable) == len(OUTPUT_FIELDS) and not raw.duplicates


def parse_answer(text):
    answer, valid = parse_fields(text)
    if not valid:
        raise ValueError("Expected exactly nine fields: strings, with Location and Notes as lists of nonempty strings")
    return answer


def validate_reference(fields):
    """Validate before inference; a missing annotation must never mean absence."""
    if not isinstance(fields, dict) or set(fields) != set(OUTPUT_FIELDS):
        raise ValueError("Reference must annotate all nine output fields")
    for field, item in fields.items():
        if not isinstance(item, dict) or item.get("status") not in STATUSES:
            raise ValueError(f"Invalid status for {field}")
        allowed = {"status", "source"}
        if item["status"] == "present":
            if field in FACT_FIELDS:
                allowed.add("facts")
                facts = item.get("facts")
                if not isinstance(facts, list) or not facts:
                    raise ValueError(f"Present {field} requires nonempty facts")
            else:
                allowed.update({"value", "alternatives"})
                facts = [item]
            seen = set()
            for fact in facts:
                if not isinstance(fact, dict):
                    raise ValueError(f"Invalid fact for {field}")
                if field in FACT_FIELDS and set(fact) - {"value", "alternatives"}:
                    raise ValueError(f"Unexpected fact annotation for {field}")
                variants = [fact.get("value"), *fact.get("alternatives", [])] if isinstance(fact.get("alternatives", []), list) else []
                if not variants or any(not isinstance(v, str) or not normalize(field, v) for v in variants):
                    raise ValueError(f"Present {field} requires nonblank values and string alternatives")
                # Different facts must not share an accepted form.
                forms = {normalize(field, v) for v in variants}
                if forms & seen:
                    raise ValueError(f"Overlapping fact alternatives in {field}")
                seen.update(forms)
        if set(item) - allowed:
            raise ValueError(f"Unexpected annotation keys for {field}")
    return fields


def legacy_reference(row):
    """Provisional CSV conversion, not a claim of human-verified label truth."""
    values = {field: row[column] for field, column in FIELDS.items()}
    species = values.pop("Species name").strip()
    # Only split a simple binomial/hybrid followed by a capitalized authority.
    # Complex taxonomic strings stay intact and need explicit annotations.
    match = re.fullmatch(r"([A-Z][a-z]+\s+[a-z-]+(?:\s+[x×]\s+(?:[A-Z][a-z]+\s+)?[a-z-]+)?)\s+([A-Z].*)", species)
    values["Species name"] = match[1] if match else species
    values["Species author"] = match[2] if match else ""
    country, separator, state = values.pop("Country/State").partition(":")
    values.update(Country=country.strip(), State=state.strip() if separator else "")
    result = {}
    for field in OUTPUT_FIELDS:
        value = values[field].strip()
        item = {"status": "present" if value else "unknown", "source": "catalogue"}
        if value:
            if field in FACT_FIELDS:
                # Catalogue delimiters supply provisional facts, not semantic inference.
                parts = [part.strip() for part in re.split(r"[/:;,]", value) if part.strip()]
                unique = {normalize(field, part): part for part in parts}
                item["facts"] = [{"value": part} for part in unique.values()]
            else:
                item["value"] = value
        result[field] = item
    return validate_reference(result)


def accepted(field, annotation):
    return {normalize(field, value) for value in [annotation["value"], *annotation.get("alternatives", [])]}


def grade(answer, valid, reference):
    values = {metric: float("nan") for metric in METRICS}
    values["valid_json"] = int(valid)
    details = {}
    omissions = additions = 0
    for field, item in reference.items():
        status = item["status"]
        if status not in {"present", "absent"}:
            details[field] = {"status": status}
            continue
        usable = field in answer
        prediction = answer.get(field, [] if field in FACT_FIELDS else "")
        if field in FACT_FIELDS:
            facts = item.get("facts", [])
            remaining = list(range(len(facts)))
            extra = []
            for fact in prediction:
                hit = next((i for i in remaining if normalize(field, fact) in accepted(field, facts[i])), None)
                if hit is None:
                    extra.append(fact)
                else:
                    remaining.remove(hit)
            matched = len(facts) - len(remaining)
            values[field] = int(usable and not remaining and not extra)
            values[field + "_recall"] = matched / len(facts) if facts else float("nan")
            values[field + "_precision"] = matched / len(prediction) if prediction else float("nan")
            missing = [facts[i]["value"] for i in remaining]
        else:
            correct = (not prediction.strip() if status == "absent"
                       else normalize(field, prediction) in accepted(field, item))
            values[field] = int(usable and correct)
            missing = [item["value"]] if status == "present" and not values[field] else []
            extra = [prediction] if prediction.strip() and not values[field] else []
        omissions += len(missing)
        additions += len(extra)
        details[field] = {"status": status, "omitted": missing, "unsupported": extra,
                          "usable_output": usable, "source": item.get("source", "golden")}
    known = [values[field] for field in OUTPUT_FIELDS if reference[field]["status"] in {"present", "absent"}]
    if known:
        values["field_accuracy"] = sum(known) / len(known)
        values["omitted_facts"] = omissions
        values["unsupported_additions"] = additions
    # A fully correct specimen cannot be established with unknown reference fields.
    if len(known) == len(OUTPUT_FIELDS):
        values["specimen_exact"] = int(all(known))
    annotation = reference["Collection date"]
    if annotation["status"] == "present":
        expected = normalize("Collection date", annotation["value"])
        predicted = normalize("Collection date", answer.get("Collection date", ""))
        pattern = r"\d{4}(?:-\d{2}(?:-\d{2})?)?"
        if re.fullmatch(pattern, expected):
            parts = predicted.split("-") if re.fullmatch(pattern, predicted) else []
            for index, part in enumerate(expected.split("-")):
                values[["date_year", "date_month", "date_day"][index]] = int(index < len(parts) and parts[index] == part)
    return values, details


@scorer(metrics={key: [mean()] for key in METRICS})
def label_fields():
    async def score(state, target):
        target_data = json.loads(target.text)
        # Old saved targets can still be inspected, but need the new output schema to rescore.
        if "schema_version" not in target_data:
            target_data = {"schema_version": 2, "fields": legacy_reference({column: target_data[field] for field, column in FIELDS.items()})}
        if target_data["schema_version"] != 2:
            raise ValueError("Unsupported reference schema version")
        reference = validate_reference(target_data["fields"])
        answer, valid = parse_fields(state.output.completion)
        values, details = grade(answer, valid, reference)
        return Score(value=values, answer=state.output.completion,
                     explanation="Field-specific grading v2; schema compliance is scored separately. See metadata for omissions and additions.",
                     metadata={"scoring_version": 2, "fields": details,
                               "unscored_fields": [f for f in OUTPUT_FIELDS if reference[f]["status"] not in {"present", "absent"}]})
    return score
