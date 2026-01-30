from __future__ import annotations

import asyncio
import json
import os
import re

import numpy as np
import openai
from tqdm.asyncio import tqdm_asyncio


class JudgeClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        max_concurrent: int = 10,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _retry(self, fn, max_retries: int = 3):
        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    return await fn()
                except openai.APIError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.5 * 2**attempt)

    async def score_response(
        self, model: str, prompt: str, response: str
    ) -> float | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Score the response to the given "
                    "prompt on a scale of 1-10. Return ONLY a JSON object with a "
                    "single key 'score' containing an integer from 1 to 10."
                ),
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\n\nResponse: {response}",
            },
        ]
        try:
            result = await self._retry(
                lambda: self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=50,
                )
            )
            text = result.choices[0].message.content.strip()
            try:
                data = json.loads(text)
                return float(data["score"])
            except (json.JSONDecodeError, KeyError, TypeError):
                match = re.search(r"\b(\d+)\b", text)
                if match:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        return float(score)
            return None
        except Exception:
            return None


async def judge_dataset(samples, model: str, client: JudgeClient) -> dict:
    async def score_pair(sample):
        chosen_score = await client.score_response(model, sample.prompt, sample.chosen)
        rejected_score = await client.score_response(
            model, sample.prompt, sample.rejected
        )
        return chosen_score, rejected_score

    results = await tqdm_asyncio.gather(
        *[score_pair(s) for s in samples], desc="Judging"
    )

    margins = []
    mislabeled = []
    easy_by_judge = []
    scored_count = 0

    for i, (cs, rs) in enumerate(results):
        if cs is not None and rs is not None:
            scored_count += 1
            margin = cs - rs
            margins.append(margin)
            if rs > cs:
                mislabeled.append(i)
            if margin > 5:
                easy_by_judge.append(i)

    margins_arr = np.array(margins) if margins else np.array([])

    return {
        "scored_count": scored_count,
        "failed_count": len(samples) - scored_count,
        "mean_margin": float(np.mean(margins_arr)) if len(margins_arr) > 0 else 0.0,
        "std_margin": float(np.std(margins_arr)) if len(margins_arr) > 0 else 0.0,
        "mislabeled_count": len(mislabeled),
        "mislabeled_indices": mislabeled,
        "easy_by_judge_count": len(easy_by_judge),
        "easy_by_judge_indices": easy_by_judge,
        "n_samples": len(samples),
    }
