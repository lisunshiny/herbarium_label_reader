"""Deterministic, field-specific scoring against explicit or provisional references."""
import json
import re
import unicodedata
from datetime import date
from functools import lru_cache

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
SCORING_VERSION = 5
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
        value = re.sub(r"\b(?:ssp|subsp)\.?\s+", "subsp. ", value)
        value = re.sub(r"\b(?:v|var)\.\s+", "var. ", value)
    if field == "Species author":
        # Abbreviation punctuation/spacing, not author spelling or identity.
        value = re.sub(r"[.\s]+", "", value)
    if field == "Collection date":
        value = normalize_date(value)
    if field in FACT_FIELDS:
        # List order is irrelevant; word order within a fact preserves relationships.
        # Preserve numeric signs and decimals rather than turning -5 into 5.
        return " ".join(re.findall(r"[+-]?\d+(?:[.,]\d+)?|[^\W\d_]+", value))
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
        allowed = {"status", "source", "raw_value", "reason"}
        if field in FACT_FIELDS:
            allowed.add("complete")
            if "complete" in item and type(item["complete"]) is not bool:
                raise ValueError(f"complete must be boolean for {field}")
            if item.get("complete") is False and item["status"] == "absent":
                raise ValueError(f"Absent {field} cannot have an incomplete reference")
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


def split_species(value):
    """Parse common written name forms without taxonomic synonym resolution.

    Return (None, None) on unrecognized syntax instead of constructing a bad target.
    Authorities at each rank are retained in their original order.
    """
    word = r"[^\W\d_]+(?:-[^\W\d_]+)*"
    value = " ".join(value.split()).replace("×", " x ")
    value = " ".join(value.split())
    base = re.match(rf"^({word}\s+(?:x\s+)?{word})(?=\s|$)", value)
    if not base:
        return None, None
    name, tail = base[1], value[base.end():].strip()
    # Hybrid formulas with an abbreviated or repeated genus, no parent authority inference.
    while re.match(r"^[xX]\s+", tail):
        parent = re.match(rf"^[xX]\s+({word})(?:\s+({word}))?", tail)
        if not parent:
            return None, None
        first = parent[1]
        if first[0].isupper():
            if not parent[2] or not parent[2][0].islower():
                return None, None
            consumed = parent.end()
            name += " x " + first + " " + parent[2]
        else:
            consumed = parent.start(2) if parent[2] else parent.end()
            name += " x " + first
        tail = tail[consumed:].strip()
    ranks = re.compile(rf"(?<!\w)(subsp\.?|ssp\.?|var\.?|v\.|forma|f\.)\s+({word})(?=\s|$)", re.I)
    authors = []
    cursor = 0
    for rank in ranks.finditer(tail):
        # 'f.' in an authority can mean filius; don't reinterpret a following author.
        if rank[1].lower() == "f." and not rank[2][0].islower():
            return None, None
        prefix = tail[cursor:rank.start()].strip()
        if prefix:
            authors.append(prefix)
        name += " " + rank[1] + " " + rank[2]
        cursor = rank.end()
    remainder = tail[cursor:].strip()
    if remainder:
        authors.append(remainder)
    author = " ".join(authors)
    # Dangling ranks, additional hybrids, or scope qualifiers require human review.
    if re.search(r"(?<!\w)(?:subsp|ssp|var|v|f)\.(?=\s|$)|\b(?:x|s\.\s*str|sensu)\b", author, re.I):
        return None, None
    if author and not (author[0].isupper() or author.startswith("(")):
        return None, None
    if author.count("(") != author.count(")"):
        return None, None
    return name, author


def prepare_reference(target):
    """Upgrade provisional saved targets, without touching reviewed annotations."""
    reference = {field: dict(item) for field, item in target["fields"].items()}
    if target.get("reference_version", 2) < 3:
        species, author = reference["Species name"], reference["Species author"]
        if all(item.get("source") == "catalogue" for item in (species, author)):
            raw = " ".join(item.get("value", "") for item in (species, author)).strip()
            name, authority = split_species(raw)
            for field, value in [("Species name", name), ("Species author", authority)]:
                item = {"source": "catalogue", "status": "present" if value else "unknown"}
                if value:
                    item["value"] = value
                elif raw and name is None:
                    item.update(raw_value=raw, reason="Ambiguous species/author syntax; needs review")
                reference[field] = item
    notes = reference["Notes"]
    if notes.get("source") == "catalogue" and notes["status"] == "present":
        notes.setdefault("complete", False)
    return validate_reference(reference)


def legacy_reference(row):
    """Provisional CSV conversion, not a claim of human-verified label truth."""
    values = {field: row[column] for field, column in FIELDS.items()}
    species = values.pop("Species name").strip()
    name, author = split_species(species)
    values["Species name"] = name or ""
    values["Species author"] = author or ""
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
                if field == "Notes":
                    item["complete"] = False
            else:
                item["value"] = value
        if field in {"Species name", "Species author"} and species and name is None:
            item.update(raw_value=species, reason="Ambiguous species/author syntax; needs review")
        result[field] = item
    return validate_reference(result)


def accepted(field, annotation):
    return {normalize(field, value) for value in [annotation["value"], *annotation.get("alternatives", [])]}


# Optional descriptive context, never a wildcard for arbitrary place names.
LOCATION_CONTEXT = frozenset("feld wald wegrand waldwegrand wiese hangwiese trockenhang ödland odland ruderal sand geschiebelehm lehm ton kies feuchter feuchte feuchten trockene trockener trocken sandiger sandige feuchtsandiger acker erlenbruch mehrfach zahlreich".split())


def location_without_metadata(value):
    # MTB sheet/quadrant notation; do not remove arbitrary numbers or distances.
    value = re.sub(r"(?<!\w)(?:MTB\s*)?\d{2}\s?\d{2}/[1-4]{1,3}(?!\d)", " ", value, flags=re.I)
    value = re.sub(r"(?<!\w)\d+(?:[.,]\d+)?\s*m\s*(?:ü\.?\s*(?:NN|NHN)|a\.?\s*s\.?\s*l\.?)(?!\w)", " ", value, flags=re.I)
    return value.strip(" ,;:")


def match_facts(field, predictions, facts, *, tolerate_context=False):
    """Match whole ordered phrases across item/delimiter boundaries.

    A clause must be fully explained by matched facts or remain unmatched. This
    prevents credit for 'flowering' inside 'not flowering'. Each reference fact
    may be used once; duplicates and unconsumed clauses remain additions.
    """
    if tolerate_context:
        predictions = [clean for value in predictions if (clean := location_without_metadata(value))]
        facts = [{**fact, 'value': location_without_metadata(fact['value']),
                  'alternatives': [location_without_metadata(v) for v in fact.get('alternatives', [])]} for fact in facts]
    atoms = []
    for prediction in predictions:
        start = len(atoms)
        # Preserve separators inside numbers (decimal commas, grid references).
        for part in re.split(r"[;:\n]|(?<!\d)[,/]|[,/](?!\d)", prediction):
            tokens = normalize(field, part).split()
            if tokens:
                atoms.append(tuple(tokens))
        if len(atoms) == start:
            # Nonempty punctuation-only output is not a correct empty answer.
            atoms.append((prediction.strip(),))
    stream = tuple(token for atom in atoms for token in atom)
    boundaries = {0}
    for atom in atoms:
        boundaries.add(max(boundaries) + len(atom))
    edges = sorted(boundaries)
    next_boundary = dict(zip(edges, edges[1:]))
    variants = [sorted({tuple(form.split()) for form in accepted(field, fact)}) for fact in facts]

    @lru_cache(maxsize=None)
    def solve(position, used):
        if position == len(stream):
            return (0, 0, (), ())  # matched facts, matched tokens, indices, unmatched clauses
        best = None
        if position in next_boundary:
            end = next_boundary[position]
            rest = solve(end, used)
            if rest is not None:
                best = (rest[0], rest[1], rest[2], (" ".join(stream[position:end]), *rest[3]))
        if tolerate_context and stream[position] in LOCATION_CONTEXT:
            rest = solve(position + 1, used)
            if rest is not None and (best is None or (rest[0], rest[1], -len(rest[3])) > (best[0], best[1], -len(best[3]))):
                best = rest
        for index, forms in enumerate(variants):
            if used & (1 << index):
                continue
            for form in forms:
                end = position + len(form)
                if stream[position:end] != form:
                    continue
                rest = solve(end, used | (1 << index))
                if rest is None:
                    continue
                candidate = (rest[0] + 1, rest[1] + len(form), (index, *rest[2]), rest[3])
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        return best

    result = solve(0, 0)
    matched = set(result[2])
    return [i for i in range(len(facts)) if i not in matched], list(result[3])


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
            remaining, extra = match_facts(field, prediction, facts, tolerate_context=field == "Location")
            matched = len(facts) - len(remaining)
            complete = item.get("complete", True)
            values[field] = int(usable and not remaining and not extra) if complete else float("nan")
            values[field + "_recall"] = matched / len(facts) if facts else float("nan")
            values[field + "_precision"] = matched / (matched + len(extra)) if (matched or extra) and complete else float("nan")
            missing = [facts[i]["value"] for i in remaining]
        else:
            correct = (not prediction.strip() if status == "absent"
                       else normalize(field, prediction) in accepted(field, item))
            values[field] = int(usable and correct)
            missing = [item["value"]] if status == "present" and not values[field] else []
            extra = [prediction] if prediction.strip() and not values[field] else []
        unverified = extra if field in FACT_FIELDS and not item.get("complete", True) else []
        if unverified:
            extra = []
        omissions += len(missing)
        additions += len(extra)
        details[field] = {"status": status, "omitted": missing, "unsupported": extra,
                          "usable_output": usable, "source": item.get("source", "golden"),
                          "reference_complete": item.get("complete", True), "unverified": unverified}
    known = [values[field] for field in OUTPUT_FIELDS if reference[field]["status"] in {"present", "absent"}
             and reference[field].get("complete", True)]
    if known:
        values["field_accuracy"] = sum(known) / len(known)
    if any(item["status"] in {"present", "absent"} for item in reference.values()):
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
        reference = prepare_reference(target_data)
        answer, valid = parse_fields(state.output.completion)
        values, details = grade(answer, valid, reference)
        return Score(value=values, answer=state.output.completion,
                     explanation="Field-specific grading v5; schema compliance is scored separately. See metadata for omissions and additions.",
                     metadata={"scoring_version": SCORING_VERSION, "fields": details,
                               "unscored_fields": [f for f in OUTPUT_FIELDS if reference[f]["status"] not in {"present", "absent"} or not reference[f].get("complete", True)]})
    return score
