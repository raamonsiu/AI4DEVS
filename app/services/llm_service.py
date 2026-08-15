from app.context.examples import ESTIMATION_EXAMPLES
from app.context.examples import format_examples
from app.dependencies import get_llm_wrapper
from app.services.evaluation import evaluate_estimation_structure

def build_system_prompt() -> str:
    examples_text = format_examples(ESTIMATION_EXAMPLES)
    return f"""You are a software projects estimation expert.
    
Use the following historical budgets as a reference:

{examples_text}

Generate a detailed estimation for the described project."""

def generate_estimation(transcription: str) -> dict:
    system_prompt = build_system_prompt()

    wrapper = get_llm_wrapper()
    result = wrapper.complete(system_prompt=system_prompt, user_message=transcription)

    return {
        "estimation": result["estimation"],
        "model": result["model"],
        "provider": result["provider"],
        "tokens_used": result["usage"]["input_tokens"] + result["usage"]["output_tokens"],
        "estimated_cost": result["cost_usd"],
        "cache_hit": result["cache_hit"],
        "evaluation": evaluate_estimation_structure(result["estimation"], result["finish_reason"]),
    }