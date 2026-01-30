from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from preflight.loader import Sample


def _make_completion(content: str):
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def judge_client():
    with patch("preflight.judge.openai") as mock_openai:
        mock_async = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_async
        mock_openai.APIError = Exception
        from preflight.judge import JudgeClient
        client = JudgeClient(api_key="test-key")
        yield client, mock_async


class TestScoreResponse:
    def test_json_format(self, judge_client):
        client, mock_async = judge_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_completion('{"score": 7}')
        )
        score = asyncio.run(client.score_response("m", "p", "r"))
        assert score == 7.0

    def test_raw_number(self, judge_client):
        client, mock_async = judge_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_completion("8")
        )
        score = asyncio.run(client.score_response("m", "p", "r"))
        assert score == 8.0

    def test_malformed_returns_none(self, judge_client):
        client, mock_async = judge_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_completion("no score here")
        )
        score = asyncio.run(client.score_response("m", "p", "r"))
        assert score is None

    def test_out_of_range_returns_none(self, judge_client):
        client, mock_async = judge_client
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_completion("99")
        )
        score = asyncio.run(client.score_response("m", "p", "r"))
        assert score is None

    def test_exception_returns_none(self, judge_client):
        client, mock_async = judge_client
        mock_async.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail"))
        score = asyncio.run(client.score_response("m", "p", "r"))
        assert score is None


def _samples():
    return [
        Sample(prompt="p1", chosen="good", rejected="bad"),
        Sample(prompt="p2", chosen="ok", rejected="great"),
        Sample(prompt="p3", chosen="fine", rejected="fine"),
    ]


class TestJudgeDataset:
    def _run(self, score_pairs):
        from preflight.judge import JudgeClient, judge_dataset
        with patch("preflight.judge.openai") as mock_openai:
            mock_openai.AsyncOpenAI.return_value = MagicMock()
            mock_openai.APIError = Exception
            client = JudgeClient(api_key="test-key")

        async def fake_score(model, prompt, response):
            for s, (cs, rs) in zip(_samples(), score_pairs):
                if prompt == s.prompt and response == s.chosen:
                    return cs
                if prompt == s.prompt and response == s.rejected:
                    return rs
            return None

        client.score_response = fake_score
        return asyncio.run(judge_dataset(_samples(), "m", client))

    def test_basic_scoring(self):
        result = self._run([(8.0, 3.0), (4.0, 6.0), (5.0, 5.0)])
        assert result["scored_count"] == 3
        assert result["failed_count"] == 0
        assert result["n_samples"] == 3

    def test_mislabeled_detection(self):
        result = self._run([(3.0, 8.0), (7.0, 2.0), (5.0, 5.0)])
        assert result["mislabeled_count"] == 1
        assert 0 in result["mislabeled_indices"]

    def test_easy_by_judge(self):
        result = self._run([(10.0, 2.0), (5.0, 5.0), (5.0, 5.0)])
        assert result["easy_by_judge_count"] == 1
        assert 0 in result["easy_by_judge_indices"]

    def test_failed_scoring(self):
        result = self._run([(None, None), (5.0, 3.0), (7.0, 2.0)])
        assert result["scored_count"] == 2
        assert result["failed_count"] == 1
