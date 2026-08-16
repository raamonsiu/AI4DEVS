import hashlib
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)

def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    system_template = _env.get_template(f"estimation/{version}/system.j2")
    user_template = _env.get_template(f"estimation/{version}/user.j2")

    context = {
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
        "reference_projects": [rp.model_dump() for rp in (request.reference_projects or [])],
    }

    system = system_template.render(**context)
    user = user_template.render(**context)

    # Content hashes (not full text) so production logs can confirm which
    # exact rendered prompt a given call used : useful for debugging a
    # regression after a prompt edit, without dumping the full prompt (and
    # the user's description inside it) into the log stream.
    log.info(
        "prompt_rendered",
        prompt_version=version,
        system_hash=hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
        user_hash=hashlib.sha256(user.encode("utf-8")).hexdigest()[:12],
    )

    return system, user
