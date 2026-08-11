# Static, LLM-free evaluation of a generated estimation's structure.
# Checks that the response contains a title, a task breakdown, a total hours
# line, a recommended team line, and a duration line; that the declared total
# hours match the sum of the individual task hours; and that the response was
# not cut off early. Produces a 0-1 score plus a list of the specific issues found.

import re

_OK_FINISH_REASONS = {"stop"}

_TITLE_RE = re.compile(r"^##\s+\S", re.MULTILINE)
_BREAKDOWN_HEADER_RE = re.compile(r"###\s*Task Breakdown", re.IGNORECASE)
_TASK_LINE_RE = re.compile(r"^\s*\d+\.\s+.+?:\s*([\d.,]+)\s*hours?", re.MULTILINE | re.IGNORECASE)
_TOTAL_HOURS_RE = re.compile(r"\*\*Total Estimated Hours:\s*([\d.,]+)", re.IGNORECASE)
_TEAM_RE = re.compile(r"\*\*Recommended Team:", re.IGNORECASE)
_DURATION_RE = re.compile(r"\*\*Estimated Duration:", re.IGNORECASE)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def evaluate_estimation_structure(text: str, finish_reason: str) -> dict:
    """Score how well a generated estimation matches the expected CAG format.

    Pure regex-based checks, no LLM call involved.
    """
    # Step 1: check that each expected section is present in the text.
    has_title = bool(_TITLE_RE.search(text))
    has_task_breakdown = bool(_BREAKDOWN_HEADER_RE.search(text))
    has_total_hours = bool(_TOTAL_HOURS_RE.search(text))
    has_team_section = bool(_TEAM_RE.search(text))
    has_duration_section = bool(_DURATION_RE.search(text))

    # Step 2: extract the hours from every numbered task line and add them up.
    task_hours = [_to_float(m.group(1)) for m in _TASK_LINE_RE.finditer(text)]
    sum_task_hours = round(sum(task_hours), 2) if task_hours else None

    # Step 3: extract the hours the model declared as the total.
    total_match = _TOTAL_HOURS_RE.search(text)
    declared_total_hours = _to_float(total_match.group(1)) if total_match else None

    # Step 4: cross-check the declared total against the sum of task hours
    # (small tolerance to absorb rounding in the model's own arithmetic).
    hours_match: bool | None
    if sum_task_hours is not None and declared_total_hours is not None:
        hours_match = abs(sum_task_hours - declared_total_hours) <= 1
    else:
        hours_match = None

    # Step 5: flag responses that were cut off before completion.
    finish_reason_ok = finish_reason in _OK_FINISH_REASONS

    # Step 6: combine all checks into a single 0-1 score.
    checks = [
        has_title,
        has_task_breakdown,
        has_total_hours,
        has_team_section,
        has_duration_section,
        bool(hours_match),
        finish_reason_ok,
    ]
    score = round(sum(checks) / len(checks), 3)

    # Step 7: build a human-readable list of what failed, if anything.
    issues = []
    if not has_title:
        issues.append("Missing '## Estimation: <name>' title")
    if not has_task_breakdown:
        issues.append("Missing '### Task Breakdown:' section")
    if not has_total_hours:
        issues.append("Missing '**Total Estimated Hours:**' line")
    if not has_team_section:
        issues.append("Missing '**Recommended Team:**' line")
    if not has_duration_section:
        issues.append("Missing '**Estimated Duration:**' line")
    if hours_match is False:
        issues.append(
            f"Total hours mismatch: declared {declared_total_hours} vs sum of tasks {sum_task_hours}"
        )
    if not finish_reason_ok:
        issues.append(f"Response truncated or unexpected finish_reason='{finish_reason}'")

    return {
        "has_title": has_title,
        "has_task_breakdown": has_task_breakdown,
        "has_total_hours": has_total_hours,
        "has_team_section": has_team_section,
        "has_duration_section": has_duration_section,
        "declared_total_hours": declared_total_hours,
        "sum_task_hours": sum_task_hours,
        "hours_match": hours_match,
        "finish_reason_ok": finish_reason_ok,
        "score": score,
        "issues": issues,
    }
