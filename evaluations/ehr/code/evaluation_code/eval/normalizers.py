"""Text normalization and ISO 8601 duration parsing utilities."""

from __future__ import annotations

import re


# Common medical abbreviations for bidirectional matching
MEDICAL_ABBREVIATIONS = {
    "cxr": "chest x-ray",
    "cxrs": "chest x-rays",
    "iv": "intravenous",
    "po": "oral",
    "o2": "oxygen",
    "ed": "emergency department",
    "icu": "intensive care unit",
    "hr": "hour",
    "hrs": "hours",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription",
    "hx": "history",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, expand medical abbreviations."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", expansion, text)
    return text


# ISO 8601 duration regex: PnYnMnDTnHnMnS
_ISO_DURATION_RE = re.compile(
    r"^P"
    r"(?:(\d+)Y)?"
    r"(?:(\d+)M)?"
    r"(?:(\d+)W)?"
    r"(?:(\d+)D)?"
    r"(?:T"
    r"(?:(\d+)H)?"
    r"(?:(\d+)M)?"
    r"(?:(\d+(?:\.\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)

# Free-text duration patterns
_FREETEXT_DURATION_PATTERNS = [
    (re.compile(r"(\d+)\s*(?:day|days)\b", re.IGNORECASE), lambda m: int(m.group(1)) * 86400),
    (re.compile(r"(\d+)\s*(?:hour|hours|hr|hrs)\b", re.IGNORECASE), lambda m: int(m.group(1)) * 3600),
    (re.compile(r"(\d+)\s*(?:week|weeks)\b", re.IGNORECASE), lambda m: int(m.group(1)) * 604800),
    (re.compile(r"(\d+)\s*(?:minute|minutes|min|mins)\b", re.IGNORECASE), lambda m: int(m.group(1)) * 60),
]


def parse_iso_duration_to_seconds(value: str) -> int | None:
    """Parse an ISO 8601 duration or free-text duration to total seconds.

    Returns None if the value cannot be parsed.
    """
    value = value.strip()

    # Try ISO 8601 format
    match = _ISO_DURATION_RE.match(value)
    if match:
        years, months, weeks, days, hours, minutes, seconds = match.groups()
        total = 0
        if years:
            total += int(years) * 365 * 86400
        if months:
            total += int(months) * 30 * 86400
        if weeks:
            total += int(weeks) * 7 * 86400
        if days:
            total += int(days) * 86400
        if hours:
            total += int(hours) * 3600
        if minutes:
            total += int(minutes) * 60
        if seconds:
            total += int(float(seconds))
        return total

    # Try free-text patterns
    for pattern, extractor in _FREETEXT_DURATION_PATTERNS:
        match = pattern.search(value)
        if match:
            return extractor(match)

    return None


def check_disqualifiers(text: str, disqualifying_patterns: list[str]) -> str | None:
    """Check if any disqualifying pattern matches the text.

    Checks against both the raw lowercased text and the abbreviation-expanded
    form so that patterns like "ICU only" match regardless of expansion.

    Returns the matched pattern if disqualified, None otherwise.
    """
    raw_lower = text.lower().strip()
    raw_lower = re.sub(r"\s+", " ", raw_lower)
    expanded = normalize_text(text)

    for pattern in disqualifying_patterns:
        # Each pattern is pipe-delimited alternatives
        alternatives = [alt.strip().lower() for alt in pattern.split("|")]
        for alt in alternatives:
            escaped = re.escape(alt)
            if re.search(escaped, raw_lower) or re.search(escaped, expanded):
                return pattern
            # Also check the expanded form of the alternative
            alt_expanded = normalize_text(alt)
            if re.search(re.escape(alt_expanded), expanded):
                return pattern
    return None
