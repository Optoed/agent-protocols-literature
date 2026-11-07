"""Exact, non-executable scoring for the rational-answer MATH pilot subset."""

import re
from fractions import Fraction


def boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index].strip()
    return None


def rational(text: str) -> Fraction | None:
    value = text.strip().replace("−", "-")
    if len(value) > 120:
        return None
    value = re.sub(r"\\[dt]?frac\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", value)
    value = value.replace(" ", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+|/[+-]?\d+)?", value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def grade(answer: str, reference: str) -> tuple[float, str]:
    expected = rational(reference)
    if expected is None:
        raise ValueError(
            "reference is outside the preregistered rational-answer subset"
        )
    extracted = boxed(answer)
    parsed = rational(extracted) if extracted is not None else None
    return float(parsed == expected), "parsed" if parsed is not None else "unparsed"
