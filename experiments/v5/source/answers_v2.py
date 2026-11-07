"""Score scalar rational answers with limited presentation normalization, without evaluation."""

import re

from benchmarks.workflow_search.answers import boxed, rational


def grade(answer: str, reference: str) -> tuple[float, str]:
    expected = rational(reference)
    if expected is None:
        raise ValueError("reference must be an exact rational number")
    value = boxed(answer)
    if value is None:
        return 0.0, "unparsed"
    value = re.sub(r"^\s*[A-Za-z]\s*=\s*", "", value).removeprefix(r"\$").strip()
    value = re.sub(
        r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])", lambda m: m[0].replace(",", ""), value
    )
    parsed = rational(value)
    return float(parsed == expected), "parsed" if parsed is not None else "unparsed"
