import re

# Known deliberate-withholding phrasing (SPEC.md section 2: "Confidential" /
# "cannot say" / "cannot disclose" is a business decision, not a data-quality
# defect -- never imputed, defaulted, or sent to an LLM to guess a value).
#
# Keyword/substring search, not a whole-cell exact match: a fixed list of
# exact phrases would miss real variation ("Cannot Disclose.", "client
# declined to disclose", "not disclosed at this time"), and for this branch
# those phrasings are the only thing standing between a would-be-withheld
# cell and the LLM fallback -- which can only be asked to decline, not
# structurally forced to. Casting a wider deterministic net here is a much
# stronger guarantee than relying on prompt compliance alone. A false
# positive (marking a genuinely numeric cell "withheld") is a safe failure
# mode for a cost/count cell; the false negative this widens against is not.
_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"disclos\w*",
        r"declin\w*",
        r"confidential",
        r"\bn/?a\b",
        r"not\s+applicable",
        r"\btbd\b",
        r"withheld",
        r"redact\w*",
        r"restrict\w*",
        r"undisclosed",
        r"\bpending\b",
        r"not\s+available",
        r"unavailable",
        r"\bunknown\b",
        r"no\s+data",
        r"cannot\s+say",
    )
]


def is_refusal_text(value: str) -> bool:
    stripped = (value or "").strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _REFUSAL_PATTERNS)
