import json
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models.llm_usage import LlmUsage, LlmUsageStatus
from app.services.prompt_injection import detect_prompt_injection

# USD per 1M tokens. Rough estimate only - verify against Groq's current
# pricing page before relying on these figures for anything beyond local
# cost visibility; they are not guaranteed to be accurate or up to date.
_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.10, 0.50),
}


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCallDecision(BaseModel):
    blocked: bool = False
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    content: str | None = None


def _to_groq_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = _PRICING_PER_MILLION_TOKENS.get(model, (0.0, 0.0))
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


class LlmGateway:
    """The single chokepoint every LLM call in this app goes through.

    Callers never talk to the Groq client directly - that's what lets cost
    tracking, injection screening, and usage persistence live in exactly
    one place instead of being re-implemented at every future call site.
    """

    def __init__(self, client: Any, model: str, session: Session, feature: str) -> None:
        self._client = client
        self._model = model
        self._session = session
        self._feature = feature

    def decide_tool_call(self, question: str, tools: list[ToolSpec]) -> ToolCallDecision:
        if detect_prompt_injection(question):
            self._record_usage(
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=0,
                status=LlmUsageStatus.BLOCKED,
            )
            return ToolCallDecision(blocked=True)

        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                tools=[_to_groq_tool(tool) for tool in tools],
                tool_choice="auto",
            )
        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._record_usage(
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=latency_ms,
                status=LlmUsageStatus.ERROR,
            )
            raise

        latency_ms = int((time.monotonic() - start) * 1000)
        usage = response.usage
        cost_usd = _estimate_cost(self._model, usage.prompt_tokens, usage.completion_tokens)
        self._record_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            status=LlmUsageStatus.SUCCESS,
        )

        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            return ToolCallDecision(
                tool_name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
        return ToolCallDecision(content=message.content)

    def _record_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: int,
        status: LlmUsageStatus,
    ) -> None:
        self._session.add(
            LlmUsage(
                feature=self._feature,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                status=status,
                created_at=datetime.now(),
            )
        )
        self._session.commit()
