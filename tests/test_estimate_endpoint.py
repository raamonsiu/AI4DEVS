"""End-to-end tests for POST /api/v1/estimate.

The pipeline is mocked at the ``EstimationService`` level: the test swaps the
real service for a fake whose ``estimate`` records the request it received and
returns a canned ``EstimationResponse``. This isolates the endpoint from
network access while still exercising request validation and response shaping.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_estimation_service
from app.guardrails.input import InputGuardrailViolation
from app.main import app
from app.schemas.estimation import (
    CallMeta,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)

VALID_PAYLOAD = {
    "description": "A small B2B SaaS to manage employee equipment loans across teams.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def _canned_result() -> EstimationResult:
    return EstimationResult(
        summary="Mid-sized B2B SaaS for equipment loans across teams.",
        confidence_pct=70,
        phases=[
            {"name": "Discovery", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Workshops, scoping and tech spike."},
            {"name": "Implementation", "duration_weeks": 6, "cost_eur": 20_000,
             "summary": "Build the core SaaS features."},
            {"name": "QA + launch", "duration_weeks": 1, "cost_eur": 5_000,
             "summary": "Test pass and production rollout."},
        ],
        total_duration_weeks=8,
        total_cost_eur=30_000,
    )


class FakeEstimationService:
    """Records the request and returns a canned response, or raises."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[EstimationRequest] = []
        self.raises = raises

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises
        return EstimationResponse(
            result=_canned_result(),
            prompt_version="v1",
            cached=False,
            meta=CallMeta(model="gpt-4o-mini", provider="openai", cost_usd=0.0004, latency_ms=1200),
        )


@pytest.fixture
def fake_service():
    svc = FakeEstimationService()
    app.dependency_overrides[get_estimation_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_estimation_service, None)


def test_valid_payload_returns_structured_response(client: TestClient, fake_service) -> None:
    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_version"] == "v1"
    assert body["cached"] is False
    assert body["meta"]["cost_usd"] == 0.0004
    assert body["result"]["total_cost_eur"] == 30_000
    assert len(body["result"]["phases"]) == 3
    assert body["result"]["phases"][0]["name"] == "Discovery"


def test_endpoint_forwards_request_to_service(client: TestClient, fake_service) -> None:
    client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert len(fake_service.calls) == 1
    received = fake_service.calls[0]
    assert received.description == VALID_PAYLOAD["description"]
    assert received.project_type.value == "web_saas"


def test_missing_project_type_returns_422(client: TestClient, fake_service) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "project_type"}
    response = client.post("/api/v1/estimate", json=payload)

    assert response.status_code == 422
    assert any(err["loc"][-1] == "project_type" for err in response.json()["detail"])


def test_invalid_enum_value_returns_422(client: TestClient, fake_service) -> None:
    response = client.post(
        "/api/v1/estimate", json={**VALID_PAYLOAD, "project_type": "not_a_real_enum"}
    )
    assert response.status_code == 422


def test_short_description_returns_422(client: TestClient, fake_service) -> None:
    response = client.post("/api/v1/estimate", json={**VALID_PAYLOAD, "description": "too short"})
    assert response.status_code == 422


def test_input_guardrail_violation_returns_400_with_reason(client: TestClient) -> None:
    svc = FakeEstimationService(
        raises=InputGuardrailViolation("Suspicious instruction-like text detected", reason="prompt_injection")
    )
    app.dependency_overrides[get_estimation_service] = lambda: svc
    try:
        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["reason"] == "prompt_injection"
        assert "message" in detail
    finally:
        app.dependency_overrides.pop(get_estimation_service, None)


def test_upstream_failure_returns_502(client: TestClient) -> None:
    svc = FakeEstimationService(raises=RuntimeError("instructor gave up"))
    app.dependency_overrides[get_estimation_service] = lambda: svc
    try:
        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)
        assert response.status_code == 502
    finally:
        app.dependency_overrides.pop(get_estimation_service, None)
