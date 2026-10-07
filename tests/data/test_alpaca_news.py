import json
from datetime import date

import pytest

from capstone.data.alpaca_news import (
    AlpacaAuthError,
    AlpacaNewsClient,
    AlpacaRequestError,
    collect_symbol,
    month_chunks,
)


class FakeResp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """미리 정해 둔 응답을 순서대로 돌려주고, 받은 params를 기록한다."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        return self.responses.pop(0)


def make_client(responses, **kw):
    sleeps = []
    client = AlpacaNewsClient("k", "s", min_interval=0, session=FakeSession(responses), sleep=sleeps.append, **kw)
    return client, sleeps


def art(i):
    return {"id": i, "headline": f"h{i}", "created_at": "2016-01-05T10:00:00Z"}


def test_month_chunks_boundaries():
    chunks = month_chunks(date(2015, 12, 15), date(2016, 2, 10))
    assert [c[0] for c in chunks] == ["2015-12", "2016-01", "2016-02"]
    assert chunks[0][1:] == ("2015-12-15T00:00:00Z", "2016-01-01T00:00:00Z")
    assert chunks[1][1:] == ("2016-01-01T00:00:00Z", "2016-02-01T00:00:00Z")
    assert chunks[2][1:] == ("2016-02-01T00:00:00Z", "2016-02-11T00:00:00Z")  # end 당일 포함


def test_pagination_follows_tokens_until_none():
    client, _ = make_client(
        [
            FakeResp(payload={"news": [art(1), art(2)], "next_page_token": "t1"}),
            FakeResp(payload={"news": [art(3)], "next_page_token": "t2"}),
            FakeResp(payload={"news": [art(4)], "next_page_token": None}),
        ]
    )
    got = list(client.iter_articles("NVDA", "2016-01-01T00:00:00Z", "2016-02-01T00:00:00Z"))
    assert [a["id"] for a in got] == [1, 2, 3, 4]
    calls = client._session.calls
    assert "page_token" not in calls[0] and calls[1]["page_token"] == "t1" and calls[2]["page_token"] == "t2"
    assert calls[0]["include_content"] == "true" and calls[0]["limit"] == 50


def test_repeated_page_token_raises():
    client, _ = make_client([FakeResp(payload={"news": [], "next_page_token": "t"})] * 3)
    with pytest.raises(AlpacaRequestError):
        list(client.iter_articles("NVDA", "a", "b"))


def test_429_retries_with_backoff_and_counts():
    client, sleeps = make_client(
        [FakeResp(429, headers={"Retry-After": "7"}), FakeResp(500), FakeResp(payload={"news": [art(1)]})]
    )
    page = client.get_page({"symbols": "NVDA"})
    assert page["news"][0]["id"] == 1
    assert client.stats.rate_limited_429 == 1 and client.stats.server_errors == 1 and client.stats.retries == 2
    assert sleeps[0] >= 7  # Retry-After 존중


def test_auth_error_is_not_retried():
    client, _ = make_client([FakeResp(403, payload={"message": "forbidden"})])
    with pytest.raises(AlpacaAuthError):
        client.get_page({})
    assert client.stats.requests == 1


def test_retry_exhaustion_raises():
    client, _ = make_client([FakeResp(429)] * 3, max_retries=2)
    with pytest.raises(AlpacaRequestError):
        client.get_page({})
    assert client.stats.requests == 3


def test_collect_symbol_resume_skips_completed_months(tmp_path):
    responses = [
        FakeResp(payload={"news": [art(1), art(2)]}),  # 2016-01
        FakeResp(payload={"news": []}),  # 2016-02 (빈 달도 완료로 기록)
    ]
    client, _ = make_client(responses)
    n = collect_symbol(client, "NVDA", date(2016, 1, 1), date(2016, 2, 29), tmp_path, progress=lambda _: None)
    assert n == 2
    lines = (tmp_path / "NVDA" / "2016.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in lines] == [1, 2]

    # 재실행: 요청이 한 번도 나가지 않아야 한다.
    client2, _ = make_client([])
    assert collect_symbol(client2, "NVDA", date(2016, 1, 1), date(2016, 2, 29), tmp_path, progress=lambda _: None) == 0
    assert client2.stats.requests == 0 and client2.stats.chunks_skipped == 2


def test_collect_symbol_dedups_by_id_when_state_lost(tmp_path):
    client, _ = make_client([FakeResp(payload={"news": [art(1)]}), FakeResp(payload={"news": [art(1), art(2)]})])
    collect_symbol(client, "NVDA", date(2016, 1, 1), date(2016, 1, 31), tmp_path, progress=lambda _: None)
    (tmp_path / "NVDA" / "_state.json").unlink()  # 쓰기 도중 끊긴 상황 모사
    collect_symbol(client, "NVDA", date(2016, 1, 1), date(2016, 1, 31), tmp_path, progress=lambda _: None)
    ids = [json.loads(x)["id"] for x in (tmp_path / "NVDA" / "2016.jsonl").read_text().splitlines()]
    assert ids == [1, 2]
