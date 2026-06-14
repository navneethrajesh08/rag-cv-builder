import os
import requests
from dotenv import load_dotenv

from generation.prompt import ANALYZE_JOB_PROMPT

load_dotenv()


def analyze_job_with_evidence(job_description: str, candidate_evidence: str) -> str:
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")

    prompt = ANALYZE_JOB_PROMPT.format(
        job_description=job_description,
        candidate_evidence=candidate_evidence,
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a precise, evidence-grounded career assistant. "
                    "Never invent candidate experience. Be honest about gaps.\n\n"
                    f"{prompt}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1800,
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]