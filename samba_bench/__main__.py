import argparse
import json
import os
import sys

import httpx2
from openai import OpenAI, OpenAIError

from .client import MeasurementError, run_loop
from .fixtures import demo_client


def main():
    parser = argparse.ArgumentParser(description="Read SambaNova timing usage through the native OpenAI SDK")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Synthetic SSE, no network (the default)")
    mode.add_argument("--live", action="store_true", help="Make billed requests using SAMBANOVA_API_KEY")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--model", default="GLM-5.3")
    parser.add_argument("--prompt", default="Explain in about 250 words why LLM decode is memory-bandwidth bound.")
    args = parser.parse_args()
    if args.live and not os.environ.get("SAMBANOVA_API_KEY"):
        parser.error("Set SAMBANOVA_API_KEY before selecting --live")
    try:
        client = (OpenAI(api_key=os.environ["SAMBANOVA_API_KEY"], base_url="https://api.sambanova.ai/v1",
                         timeout=20, max_retries=0) if args.live else demo_client())
        with client:
            report = run_loop(client, args.prompt, steps=args.steps, max_tokens=args.max_tokens, model=args.model)
        report["source"] = "live provider" if args.live else "synthetic SDK transport fixture"
        report["performance_evidence"] = bool(args.live)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (MeasurementError, OpenAIError, httpx2.HTTPError, ValueError):
        # Provider responses and request metadata may contain sensitive content.
        print("Measurement failed: inspect configuration, timing fields, and stream completeness.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
