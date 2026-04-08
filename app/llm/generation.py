from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol
from urllib import error, request


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    evidence_lines: list[str]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    used_generator: bool
    fallback_reason: str | None = None


class Generator(Protocol):
    provider: str

    def generate(self, payload: GenerationRequest) -> GenerationResult:
        ...


@dataclass(frozen=True)
class GeneratorConfig:
    provider: str = "extractive"
    enabled: bool = False
    model: str | None = None
    endpoint: str = "http://localhost:11434/api/generate"
    timeout_seconds: float = 15.0
    temperature: float = 0.0


class ExtractiveGenerator:
    provider = "extractive"

    def generate(self, payload: GenerationRequest) -> GenerationResult:
        lines = ["Evidence-based answer (extractive preview):"]
        for index, evidence in enumerate(payload.evidence_lines[:3], start=1):
            lines.append(f"{index}. {evidence}")
        return GenerationResult(text="\n".join(lines), used_generator=False)


class OllamaGenerator:
    provider = "ollama"

    def __init__(self, *, model: str, endpoint: str, timeout_seconds: float, temperature: float) -> None:
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature

    def generate(self, payload: GenerationRequest) -> GenerationResult:
        evidence_block = "\n".join(payload.evidence_lines)
        prompt = (
            "Answer the question using only the evidence snippets below. "
            "If the evidence is not enough, explicitly say so.\n\n"
            f"Question: {payload.question}\n\n"
            "Evidence:\n"
            f"{evidence_block}\n\n"
            "Return a concise answer and include references in the form [n]."
        )

        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        http_request = request.Request(
            self._endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                payload_json = json.loads(response.read().decode("utf-8"))
            generated = str(payload_json.get("response", "")).strip()
        except error.HTTPError as exc:
            return GenerationResult(text="", used_generator=False, fallback_reason=f"ollama_http_{exc.code}")
        except (error.URLError, TimeoutError):
            return GenerationResult(text="", used_generator=False, fallback_reason="ollama_unreachable")
        except (json.JSONDecodeError, ValueError):
            return GenerationResult(text="", used_generator=False, fallback_reason="ollama_invalid_response")

        if not generated:
            return GenerationResult(text="", used_generator=False, fallback_reason="ollama_empty_response")

        return GenerationResult(text=generated, used_generator=True)



def build_generator(config: GeneratorConfig) -> Generator:
    if not config.enabled:
        return ExtractiveGenerator()

    if config.provider == "extractive":
        return ExtractiveGenerator()

    if config.provider == "ollama":
        if not config.model:
            raise ValueError("Generator provider 'ollama' requires QUERY_GENERATION_MODEL.")
        return OllamaGenerator(
            model=config.model,
            endpoint=config.endpoint,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
        )

    raise ValueError(f"Unknown generator provider '{config.provider}'.")
