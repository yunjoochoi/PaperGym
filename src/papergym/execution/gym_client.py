"""In-sandbox client: the agent's ONLY way to reach an LLM. Talks to the host
metering proxy at $GYM_LLM_URL. No provider keys exist in the sandbox."""
from __future__ import annotations

import json
import os
import urllib.request


def metered_llm_call(messages: list, temperature: float = 0.0) -> str:
    url = os.environ.get("GYM_LLM_URL")
    if not url:
        raise RuntimeError("GYM_LLM_URL not set — no LLM available in sandbox")
    data = json.dumps({"messages": messages, "temperature": temperature}).encode()
    req = urllib.request.Request(url + "/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    body = json.loads(resp.read())
    if getattr(resp, "status", 200) != 200 or "content" not in body:
        raise RuntimeError(f"gym LLM error: {body.get('error', body)}")
    return body["content"]
