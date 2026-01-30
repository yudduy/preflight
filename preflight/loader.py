from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field


@dataclass
class Sample:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict = field(default_factory=dict)


def detect_format(line: dict) -> str:
    if "preferred_output" in line:
        return "together_ai"
    if "chosen" in line:
        chosen = line["chosen"]
        rejected = line.get("rejected", "")
        if isinstance(chosen, str) and isinstance(rejected, str):
            if "\n\nHuman:" in chosen and "\n\nHuman:" in rejected:
                return "anthropic_hh"
        return "huggingface"
    raise ValueError(f"Unknown format: keys={list(line.keys())}")


def parse_together(line: dict) -> Sample:
    messages = line.get("input", {}).get("messages", [])
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    chosen = line["preferred_output"][0]["content"]
    rejected = line["non_preferred_output"][0]["content"]
    return Sample(prompt=prompt, chosen=chosen, rejected=rejected, metadata={"format": "together_ai"})


def _extract_assistant_content(msgs: list[dict]) -> str:
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _extract_user_prompt(msgs: list[dict]) -> str:
    parts = []
    for m in msgs:
        if m.get("role") == "user":
            parts.append(m.get("content", ""))
    return "\n".join(parts)


def parse_huggingface(line: dict) -> Sample:
    prompt = line.get("prompt") or line.get("instruction") or ""
    chosen_raw = line["chosen"]
    rejected_raw = line["rejected"]

    if isinstance(chosen_raw, list):
        if not prompt:
            prompt = _extract_user_prompt(chosen_raw)
        chosen = _extract_assistant_content(chosen_raw)
    else:
        chosen = chosen_raw

    if isinstance(rejected_raw, list):
        rejected = _extract_assistant_content(rejected_raw)
    else:
        rejected = rejected_raw

    return Sample(prompt=prompt, chosen=chosen, rejected=rejected, metadata={"format": "huggingface"})


def _parse_hh_turns(text: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    parts = text.split("\n\nHuman:")
    for part in parts:
        if not part.strip():
            continue
        if "\n\nAssistant:" in part:
            human_part, assistant_part = part.split("\n\nAssistant:", 1)
            turns.append(("human", human_part.strip()))
            turns.append(("assistant", assistant_part.strip()))
        else:
            turns.append(("human", part.strip()))
    return turns


def parse_anthropic_hh(line: dict) -> Sample:
    chosen_turns = _parse_hh_turns(line["chosen"])
    rejected_turns = _parse_hh_turns(line["rejected"])

    human_parts = [t[1] for t in chosen_turns if t[0] == "human"]
    prompt = "\n".join(human_parts)

    chosen = ""
    for role, content in reversed(chosen_turns):
        if role == "assistant":
            chosen = content
            break

    rejected = ""
    for role, content in reversed(rejected_turns):
        if role == "assistant":
            rejected = content
            break

    return Sample(prompt=prompt, chosen=chosen, rejected=rejected, metadata={"format": "anthropic_hh"})


_PARSERS = {
    "together_ai": parse_together,
    "huggingface": parse_huggingface,
    "anthropic_hh": parse_anthropic_hh,
}


def load_dataset(path: str) -> tuple[list[Sample], str, int]:
    lines: list[dict] = []
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

    if not lines:
        raise ValueError("No valid JSON lines found")

    fmt = detect_format(lines[0])
    parser = _PARSERS[fmt]

    samples: list[Sample] = []
    n_skipped = 0
    for line in lines:
        try:
            samples.append(parser(line))
        except Exception as e:
            warnings.warn(f"Skipping malformed line: {e}")
            n_skipped += 1

    if not samples:
        raise ValueError("Zero valid samples after parsing")

    return samples, fmt, n_skipped
