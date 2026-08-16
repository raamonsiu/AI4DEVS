import pytest
from jinja2 import TemplateNotFound

from app.prompts.loader import render_estimation_prompt
from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType, ReferenceProject


def test_estimation_prompt_includes_description_in_user_block():
    request = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system, user = render_estimation_prompt(request)

    assert "<project_description>" in user
    assert "Mobile app with login" in user
    assert "phases_table" in system
    assert "confidence_pct" in system


def test_output_format_keyword_only_present_for_its_own_branch():
    base_kwargs = dict(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
    )

    phases_table_request = EstimationRequest(output_format=OutputFormat.PHASES_TABLE, **base_kwargs)
    narrative_request = EstimationRequest(output_format=OutputFormat.NARRATIVE, **base_kwargs)

    phases_table_system, _ = render_estimation_prompt(phases_table_request)
    narrative_system, _ = render_estimation_prompt(narrative_request)

    # "phases_table" only ever appears via the <output_format> tag rendering
    # that exact enum value : unlike "confidence_pct", it isn't also present
    # in the always-included few-shot examples, so it cleanly distinguishes
    # the branch actually taken.
    assert "phases_table" in phases_table_system
    assert "phases_table" not in narrative_system


def test_detailed_level_adds_per_phase_assumptions_instruction():
    base_kwargs = dict(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        output_format=OutputFormat.PHASES_TABLE,
    )

    detailed_request = EstimationRequest(detail_level=DetailLevel.DETAILED, **base_kwargs)
    summary_request = EstimationRequest(detail_level=DetailLevel.SUMMARY, **base_kwargs)

    detailed_system, _ = render_estimation_prompt(detailed_request)
    summary_system, _ = render_estimation_prompt(summary_request)

    assert "list the assumptions" in detailed_system
    assert "list the assumptions" not in summary_system


def test_v2_prompt_version_is_a_deliberately_different_variant():
    request = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    v1_system, _ = render_estimation_prompt(request, version="v1")
    v2_system, _ = render_estimation_prompt(request, version="v2")

    assert v1_system != v2_system
    # v2's deliberate variation: a risk-first framing absent from v1.
    assert "Key risk" in v2_system
    assert "Key risk" not in v1_system


def test_unknown_prompt_version_raises():
    request = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(request, version="v99")


def test_reference_projects_are_rendered_when_present():
    request_with_refs = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        reference_projects=[
            ReferenceProject(
                name="Acme Loyalty App",
                description="A loyalty points mobile app for a retail chain.",
                actual_duration_weeks=9,
                actual_cost_eur=18000,
            )
        ],
    )
    request_without_refs = EstimationRequest(
        description="Mobile app with login, chat and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    _, user_with_refs = render_estimation_prompt(request_with_refs)
    _, user_without_refs = render_estimation_prompt(request_without_refs)

    assert "<reference_projects>" in user_with_refs
    assert "Acme Loyalty App" in user_with_refs
    assert "18000" in user_with_refs
    assert "<reference_projects>" not in user_without_refs
