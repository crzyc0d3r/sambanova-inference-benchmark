"""Measure provider-reported timing fields without deriving token counts from text."""
from dataclasses import asdict, dataclass
import math
import time


class MeasurementError(ValueError):
    pass


def number(value, name: str, *, positive: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementError(f"{name} must be numeric")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise MeasurementError(f"{name} is outside its valid range") from None
    if not math.isfinite(converted) or converted < 0 or (positive and converted == 0):
        raise MeasurementError(f"{name} is outside its valid range")
    return converted


@dataclass(frozen=True)
class Turn:
    text: str
    finish_reason: str
    client_elapsed_seconds: float
    server_time_to_first_token_seconds: float
    server_completion_tokens_after_first_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self):
        return asdict(self)


def run_turn(client, messages: list[dict], *, model: str = "GLM-5.3", max_tokens: int = 300,
             clock=time.perf_counter) -> Turn:
    if not messages or any(set(message) != {"role", "content"} or
                           message["role"] not in {"system", "user", "assistant"} or
                           not isinstance(message["content"], str) for message in messages):
        raise MeasurementError("Messages must contain only role and text content")
    if sum(len(message["content"]) for message in messages) > 20000:
        raise MeasurementError("Conversation exceeds the character budget")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
        raise MeasurementError("max_tokens must be between 1 and 4096")
    start = clock()
    stream_deadline = time.monotonic() + 30
    usage = None
    text = []
    characters = 0
    finish_reason = None
    stream = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                             stream=True, stream_options={"include_usage": True})
    try:
        for index, chunk in enumerate(stream):
            if time.monotonic() > stream_deadline:
                raise MeasurementError("Stream exceeds its event time budget")
            if index >= 20000:
                raise MeasurementError("Stream exceeds the chunk budget")
            choices = getattr(chunk, "choices", None)
            if not isinstance(choices, list):
                raise MeasurementError("Stream choices must be a list")
            for choice in choices:
                if getattr(choice, "index", None) != 0:
                    raise MeasurementError("Only one completion choice is supported")
                delta = getattr(choice, "delta", None)
                if not hasattr(delta, "model_dump"):
                    raise MeasurementError("Stream delta is malformed")
                content = getattr(delta, "content", None)
                if content is not None and not isinstance(content, str):
                    raise MeasurementError("Stream text must be a string")
                if content:
                    text.append(content)
                    characters += len(content)
                    if characters > 100000:
                        raise MeasurementError("Stream exceeds the text budget")
                reason = getattr(choice, "finish_reason", None)
                if reason is not None:
                    if not isinstance(reason, str):
                        raise MeasurementError("Stream finish event is malformed")
                    finish_reason = reason
            if chunk.usage is not None:
                if usage is not None:
                    raise MeasurementError("Expected exactly one final usage record")
                if not hasattr(chunk.usage, "model_dump"):
                    raise MeasurementError("Stream usage record is malformed")
                usage = chunk.usage.model_dump()
    finally:
        stream.close()
    elapsed = clock() - start
    number(elapsed, "client elapsed time")
    if finish_reason not in {"stop", "length"}:
        raise MeasurementError("Stream ended without a supported completion finish event")
    if usage is None:
        raise MeasurementError("Provider did not return final usage")
    required = ["time_to_first_token", "completion_tokens_after_first_per_sec",
                "prompt_tokens", "completion_tokens", "total_tokens"]
    if any(name not in usage or usage[name] is None for name in required):
        raise MeasurementError("Provider omitted a required timing or token field")
    first = number(usage["time_to_first_token"], "time_to_first_token")
    rate = number(usage["completion_tokens_after_first_per_sec"],
                  "completion_tokens_after_first_per_sec", positive=True)
    for name in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = usage[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MeasurementError(f"{name} must be a nonnegative integer")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise MeasurementError("Provider token totals are inconsistent")
    return Turn("".join(text), finish_reason, round(elapsed, 6), first, rate,
                usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])


def run_loop(client, prompt: str, *, steps: int = 10, max_tokens: int = 300,
             model: str = "GLM-5.3") -> dict:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise MeasurementError("Prompt must contain 1 to 4000 characters")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10:
        raise MeasurementError("steps must be between 1 and 10")
    messages = [{"role": "user", "content": prompt}]
    turns = []
    for step in range(steps):
        result = run_turn(client, messages, model=model, max_tokens=max_tokens)
        turns.append(result)
        messages.append({"role": "assistant", "content": result.text})
        if step + 1 < steps:
            messages.append({"role": "user", "content": f"Step {step + 2}: identify one implication of your previous answer."})
    return {"steps": steps, "model": model, "turns": [turn.to_dict() for turn in turns],
            "totals": {"client_elapsed_seconds": round(sum(turn.client_elapsed_seconds for turn in turns), 6),
                       "server_time_to_first_token_seconds_sum": sum(turn.server_time_to_first_token_seconds for turn in turns),
                       "reported_completion_tokens": sum(turn.completion_tokens for turn in turns),
                       "reported_prompt_tokens": sum(turn.prompt_tokens for turn in turns),
                       "server_decode_duration_seconds": None},
            "interpretation": "A sequential conversation loop, without autonomous tool execution. "
                              "No decode duration/fraction is inferred from a rate or from text length."}
