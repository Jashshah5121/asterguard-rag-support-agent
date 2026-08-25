from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised in minimal environments
    OpenAI = None  # type: ignore[assignment]

from app.config import settings


def _normalize_output(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u202f": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


class LLMClient:
    """Thin wrapper around Groq's OpenAI-compatible API.

    Initialization is deliberately non-fatal when the optional OpenAI package
    or GROQ_API_KEY is absent. This keeps deterministic parts of the agent
    usable and lets the responder invoke its grounded fallback rather than
    crashing the whole application.
    """

    def __init__(self, client: Any = None):
        self.model = settings.groq_model
        self.client = client

        if self.client is None and OpenAI is not None and settings.groq_api_key:
            self.client = OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self.client is None:
            raise RuntimeError(
                "LLM client is unavailable. Configure GROQ_API_KEY and install openai."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        return _normalize_output(response.choices[0].message.content.strip())
