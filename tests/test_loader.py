from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from preflight.loader import (
    Sample,
    detect_format,
    load_dataset,
    parse_anthropic_hh,
    parse_huggingface,
    parse_together,
)


def _write_jsonl(lines: list[dict]) -> str:
    p = Path(tempfile.mktemp(suffix=".jsonl"))
    p.write_text("\n".join(json.dumps(l) for l in lines))
    return str(p)


TOGETHER_LINE = {
    "input": {"messages": [{"role": "user", "content": "What is Python?"}]},
    "preferred_output": [{"content": "A programming language."}],
    "non_preferred_output": [{"content": "A snake."}],
}

HF_STRING_LINE = {
    "prompt": "What is Python?",
    "chosen": "A programming language.",
    "rejected": "A snake.",
}

HF_MSG_LINE = {
    "chosen": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A programming language."},
    ],
    "rejected": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "A snake."},
    ],
}

ANTHROPIC_LINE = {
    "chosen": "\n\nHuman: What is Python?\n\nAssistant: A programming language.",
    "rejected": "\n\nHuman: What is Python?\n\nAssistant: A snake.",
}


def test_detect_together():
    assert detect_format(TOGETHER_LINE) == "together_ai"


def test_detect_huggingface():
    assert detect_format(HF_STRING_LINE) == "huggingface"


def test_detect_anthropic():
    assert detect_format(ANTHROPIC_LINE) == "anthropic_hh"


def test_detect_unknown():
    with pytest.raises(ValueError):
        detect_format({"foo": "bar"})


def test_parse_together():
    s = parse_together(TOGETHER_LINE)
    assert s.chosen == "A programming language."
    assert s.rejected == "A snake."
    assert "user" in s.prompt.lower() or "Python" in s.prompt


def test_parse_hf_string():
    s = parse_huggingface(HF_STRING_LINE)
    assert s.prompt == "What is Python?"
    assert s.chosen == "A programming language."
    assert s.rejected == "A snake."


def test_parse_hf_messages():
    s = parse_huggingface(HF_MSG_LINE)
    assert s.chosen == "A programming language."
    assert s.rejected == "A snake."
    assert "Python" in s.prompt


def test_parse_anthropic():
    s = parse_anthropic_hh(ANTHROPIC_LINE)
    assert s.chosen == "A programming language."
    assert s.rejected == "A snake."
    assert "Python" in s.prompt


def test_load_together():
    path = _write_jsonl([TOGETHER_LINE] * 3)
    samples, fmt, skipped = load_dataset(path)
    assert fmt == "together_ai"
    assert len(samples) == 3
    assert skipped == 0


def test_load_skips_malformed():
    bad = {"preferred_output": "not a list"}
    path = _write_jsonl([TOGETHER_LINE, bad, TOGETHER_LINE])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, fmt, skipped = load_dataset(path)
    assert len(samples) == 2
    assert skipped == 1


def test_load_empty_raises():
    path = _write_jsonl([])
    with pytest.raises(ValueError):
        load_dataset(path)


def test_missing_fields():
    line = {"preferred_output": [{"content": "x"}], "non_preferred_output": [{"content": "y"}]}
    s = parse_together(line)
    assert s.prompt == ""
