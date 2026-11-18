"""Synthetic SSE served through the actual SDK's HTTP transport; no sockets."""
import json

import httpx2
from openai import OpenAI


def sse_events(*, usage=None, text="Synthetic fixture response.", finish_reason="stop", include_usage=True):
    envelope = {"id": "synthetic-completion", "object": "chat.completion.chunk", "created": 0, "model": "GLM-5.3"}
    chunks = [dict(envelope, choices=[{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]),
              dict(envelope, choices=[{"index": 0, "delta": {}, "finish_reason": finish_reason}])]
    if include_usage:
        if usage is None:
            usage = {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13,
                     "time_to_first_token": 0.25, "completion_tokens_after_first_per_sec": 80.0}
        chunks.append(dict(envelope, choices=[], usage=usage))
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def demo_client(requests=None):
    def handler(request):
        payload = json.loads(request.content)
        if requests is not None:
            requests.append(payload)
        count = len(payload["messages"])
        usage = {"prompt_tokens": 8 * count, "completion_tokens": 5, "total_tokens": 8 * count + 5,
                 "time_to_first_token": 0.25, "completion_tokens_after_first_per_sec": 80.0}
        return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse_events(usage=usage))
    return OpenAI(api_key="synthetic-demo-key", base_url="https://fixture.invalid/v1",
                  http_client=httpx2.Client(transport=httpx2.MockTransport(handler)), max_retries=0)
