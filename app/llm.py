"""Optional LLM enrichment via Groq (OpenAI-compatible API).

The LLM never sets or changes the score or classification. It only rewrites
the three text fields, using the facts already computed by the rules.
If no key is configured or the call fails, the rule-based text is kept.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
TEXT_KEYS = ("quality_assessment", "scoring_explanation", "recommended_next_action")

SYSTEM_PROMPT = (
    "You write sales-qualification notes. You are given a lead and a score that was "
    "already computed by fixed rules. Do not change, question or recompute the score or "
    "classification. Use only the facts provided. Respond only with a JSON object with "
    "keys: quality_assessment (1-2 sentences), scoring_explanation (2-3 sentences), "
    "recommended_next_action (1-2 sentences)."
)


def llm_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def enrich_with_llm(result: dict) -> dict:
    if not llm_enabled():
        return result

    facts = {k: result[k] for k in (
        "name", "company", "score", "classification", "confidence", "factors",
        "positive_factors", "negative_factors", "missing_fields", "data_warnings")}
    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
        ],
    }
    try:
        resp = httpx.post(
            GROQ_URL,
            json=payload,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = json.loads(resp.json()["choices"][0]["message"]["content"])
        for key in TEXT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
        result["llm_used"] = True
    except Exception as exc:  # network, quota, bad JSON: fall back silently
        log.warning("LLM enrichment failed; using rule-based text: %s", exc)
    return result
