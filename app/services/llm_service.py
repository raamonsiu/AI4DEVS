from openai import OpenAI
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.context.examples import format_examples
from app.constants import MODELS_PRICING

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODELS_PRICING.get(model, {"input": 0.50, "output": 1.50})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)

def build_system_prompt() -> str:
    examples_text = format_examples(ESTIMATION_EXAMPLES)
    return f"""You are a software projects estimation expert.
    
Use the following historical budgets as a reference:

{examples_text}

Generate a detailed estimation for the described project."""

def generate_estimation(transcription: str) -> dict:
    system_prompt = build_system_prompt()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription}
        ]
    )

    return {
        "estimation": response.choices[0].message.content,
        "model": settings.LLM_MODEL,
        "provider": settings.LLM_PROVIDER,
        "tokens_used": response.usage.prompt_tokens + response.usage.completion_tokens,
        "estimated_cost": calculate_cost(
            settings.LLM_MODEL,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        ),
    }