# Static, LLM-free evaluation of a generated estimation's structure.
# Checks that the response contains a title, a task breakdown table, a total
# hours line, a total cost line, a recommended team line, and a duration
# line; that the declared totals (hours and cost) match the sum of the
# individual table rows; and that the response was not cut off early.
# Produces a 0-1 score plus a list of the specific issues found.

import re

_OK_FINISH_REASONS = {"stop"}

_TITLE_RE = re.compile(r"^\s*##\s+\S", re.MULTILINE)
_HEADER_ROW_RE = re.compile(r"\|\s*Task\s*\|\s*Hours\s*\|\s*Cost", re.IGNORECASE)
_SEPARATOR_ROW_RE = re.compile(r"^[\s|:-]+$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(
    r"^ *\| *(?P<task>[^|\n]+) *\| *(?P<hours>[\d.,]+) *\| *(?P<cost>[\d.,]+) *\| *$",
    re.MULTILINE,
)
_TOTAL_HOURS_RE = re.compile(r"\*\*Total Estimated Hours:\s*([\d.,]+)", re.IGNORECASE)
_TOTAL_COST_RE = re.compile(r"\*\*Total Estimated Cost:\s*([\d.,]+)", re.IGNORECASE)
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
    has_breakdown_table = bool(_HEADER_ROW_RE.search(text))
    has_total_hours = bool(_TOTAL_HOURS_RE.search(text))
    has_total_cost = bool(_TOTAL_COST_RE.search(text))
    has_team_section = bool(_TEAM_RE.search(text))
    has_duration_section = bool(_DURATION_RE.search(text))

    # Step 2: parse every table row (skipping the header/separator rows) and
    # sum their hours and cost columns.
    sum_task_hours = None
    sum_task_cost = None
    if has_breakdown_table:
        running_hours = 0.0
        running_cost = 0.0
        found_rows = False
        for match in _TABLE_ROW_RE.finditer(text):
            if _SEPARATOR_ROW_RE.match(match.group(0)) or match.group("task").strip().lower() == "task":
                continue
            running_hours += _to_float(match.group("hours"))
            running_cost += _to_float(match.group("cost"))
            found_rows = True
        if found_rows:
            sum_task_hours = round(running_hours, 2)
            sum_task_cost = round(running_cost, 2)

    # Step 3: extract the totals the model declared.
    hours_match_obj = _TOTAL_HOURS_RE.search(text)
    declared_total_hours = _to_float(hours_match_obj.group(1)) if hours_match_obj else None
    cost_match_obj = _TOTAL_COST_RE.search(text)
    declared_total_cost = _to_float(cost_match_obj.group(1)) if cost_match_obj else None

    # Step 4: cross-check the declared totals against the summed rows
    # (small tolerance to absorb rounding in the model's own arithmetic).
    hours_match: bool | None
    if sum_task_hours is not None and declared_total_hours is not None:
        hours_match = abs(sum_task_hours - declared_total_hours) <= 1
    else:
        hours_match = None

    cost_match: bool | None
    if sum_task_cost is not None and declared_total_cost is not None and declared_total_cost > 0:
        cost_match = abs(sum_task_cost - declared_total_cost) / declared_total_cost <= 0.02
    else:
        cost_match = None

    # Step 5: flag responses that were cut off before completion.
    finish_reason_ok = finish_reason in _OK_FINISH_REASONS

    # Step 6: combine all checks into a single 0-1 score.
    checks = [
        has_title,
        has_breakdown_table,
        has_total_hours,
        has_total_cost,
        has_team_section,
        has_duration_section,
        bool(hours_match),
        bool(cost_match),
        finish_reason_ok,
    ]
    score = round(sum(checks) / len(checks), 3)

    # Step 7: build a human-readable list of what failed, if anything.
    issues = []
    if not has_title:
        issues.append("Missing '## Estimation: <name>' title")
    if not has_breakdown_table:
        issues.append("Missing '| Task | Hours | Cost |' breakdown table")
    if not has_total_hours:
        issues.append("Missing '**Total Estimated Hours:**' line")
    if not has_total_cost:
        issues.append("Missing '**Total Estimated Cost:**' line")
    if not has_team_section:
        issues.append("Missing '**Recommended Team:**' line")
    if not has_duration_section:
        issues.append("Missing '**Estimated Duration:**' line")
    if hours_match is False:
        issues.append(
            f"Total hours mismatch: declared {declared_total_hours} vs sum of rows {sum_task_hours}"
        )
    if cost_match is False:
        issues.append(
            f"Total cost mismatch: declared {declared_total_cost} EUR vs sum of rows {sum_task_cost} EUR"
        )
    if not finish_reason_ok:
        issues.append(f"Response truncated or unexpected finish_reason='{finish_reason}'")

    return {
        "has_title": has_title,
        "has_breakdown_table": has_breakdown_table,
        "has_total_hours": has_total_hours,
        "has_total_cost": has_total_cost,
        "has_team_section": has_team_section,
        "has_duration_section": has_duration_section,
        "declared_total_hours": declared_total_hours,
        "sum_task_hours": sum_task_hours,
        "hours_match": hours_match,
        "declared_total_cost": declared_total_cost,
        "sum_task_cost": sum_task_cost,
        "cost_match": cost_match,
        "finish_reason_ok": finish_reason_ok,
        "score": score,
        "issues": issues,
    }
