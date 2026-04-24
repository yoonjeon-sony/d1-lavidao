"""Shared robust answer extractors for lmms_eval tasks.

Handles, in priority order:
  1. <answer>X</answer> (and <ans>X</ans>) tags
  2. \\boxed{X} / boxed X
  3. "the answer is X" / "final answer is X" / "answer: X" / "option X" / "choice X"
  4. **X** or **(X)** bold markdown
  5. Trailing letter at end of text
  6. Leading letter (legacy behaviour)
  7. Last standalone letter anywhere in the text

The extractor refuses to guess when no pattern matches (returns "").
"""
import re

__all__ = ["extract_mc_letter", "extract_chartqa_answer"]


def extract_mc_letter(text: str, valid_letters: str = "ABCDE") -> str:
    """Return a single uppercase choice letter from `text`, or '' if nothing plausible."""
    if not text:
        return ""
    v = f"[{valid_letters}]"
    t = text
    # 1. <answer> tags
    for pat in (
        rf"<\s*answer\s*>\s*\(?\s*({v})\s*\)?\s*<\s*/\s*answer\s*>",
        rf"<\s*ans\s*>\s*\(?\s*({v})\s*\)?\s*<\s*/\s*ans\s*>",
    ):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # 2. boxed
    for pat in (
        rf"\\?boxed\{{\s*\(?\s*({v})\s*\)?\s*\}}",
        rf"\\?boxed\s*\(?\s*({v})\s*\)?",
    ):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # 3. answer-phrase patterns
    for pat in (
        rf"(?:final\s+|correct\s+)?answer\s*(?:is|=|:)\s*\(?\s*\*?\*?\s*({v})\s*\*?\*?\s*\)?",
        rf"\banswer\s*[-–—]\s*\(?({v})\)?",
        rf"\bthe\s+answer\s+is\s+\(?\*?\*?({v})\*?\*?\)?",
        rf"\boption\s*\(?({v})\)?",
        rf"\bchoice\s*\(?({v})\)?",
        rf"\*\*\(?({v})\)?\*\*",
        rf"\bis\s*\(\s*({v})\s*\)\s*[\.!]?",
    ):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # 4. Trailing letter near the end of text
    tail = t.rstrip(" .\n\t\"'")
    if tail:
        tail_tail = tail[-10:] if len(tail) > 10 else tail
        m = re.search(rf"(?:^|\s|\()({v})(?:\)|\.|\s|$)", tail_tail)
        if m:
            inner = re.search(rf"({v})", m.group(0))
            if inner:
                return inner.group(1).upper()
    # 5. Leading letter (legacy behaviour)
    m = re.match(rf"[\(\s]*({v})[\)\.\s]*(?:$|\s)", t.strip())
    if m:
        return m.group(1).upper()
    # 6. Last standalone letter anywhere (lowest priority)
    matches = list(re.finditer(rf"(?:^|[\s(\[])({v})(?=[\s)\].,!?:;]|$)", t))
    if matches:
        return matches[-1].group(1).upper()
    return ""


def _to_float(s):
    s = s.strip().strip(",")
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s.replace(",", ""))
    except ValueError:
        return None


def extract_chartqa_answer(text: str) -> str:
    """Return the best candidate answer string from chart-QA response."""
    if not text:
        return ""
    t = text.strip()
    m = re.search(r"<\s*answer\s*>\s*([^<]+?)\s*<\s*/\s*answer\s*>", t, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"\\?boxed\{\s*([^}]+?)\s*\}", t)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"(?:final\s+|correct\s+)?answer\s*(?:is|=|:)\s*([^\n\.\!\?]+)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    if _to_float(t) is not None:
        return t
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", t)
    if nums:
        return nums[-1]
    return t
