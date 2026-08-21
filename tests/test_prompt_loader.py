"""Template-only tests: no LLM, no network, no Redis. They run in milliseconds
and assert what the rendered prompt actually contains for each form option."""

import pytest
from jinja2 import TemplateNotFound

from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType

DESCRIPTION = "Mobile app with login, chat and push notifications for a retail chain."


def make_request(
    *,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    output_format: OutputFormat = OutputFormat.PHASES_TABLE,
) -> EstimationRequest:
    return EstimationRequest(
        description=DESCRIPTION,
        project_type=ProjectType.MOBILE_APP,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_estimation_prompt_includes_description_in_user_block():
    system, user = render_estimation_prompt(make_request())

    assert "<project_description>" in user
    assert DESCRIPTION in user
    assert "mobile_app" in user
    assert "phases_table" in system


def test_output_format_keyword_only_present_for_its_own_branch():
    phases_table_system, _ = render_estimation_prompt(
        make_request(output_format=OutputFormat.PHASES_TABLE)
    )
    narrative_system, _ = render_estimation_prompt(
        make_request(output_format=OutputFormat.NARRATIVE)
    )

    assert "phases_table" in phases_table_system
    assert "phases_table" not in narrative_system
    assert "flowing prose" in narrative_system


def test_detailed_level_adds_per_phase_assumptions_instruction():
    detailed_system, _ = render_estimation_prompt(
        make_request(detail_level=DetailLevel.DETAILED)
    )
    summary_system, _ = render_estimation_prompt(
        make_request(detail_level=DetailLevel.SUMMARY)
    )

    assert "assumptions you made per phase" in detailed_system
    assert "assumptions you made per phase" not in summary_system


def test_unknown_prompt_version_raises():
    """Versioning is a real dispatch, not a label: asking for a version that
    isn't on disk must fail loudly rather than silently fall back to v1."""
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(make_request(), version="v99")
