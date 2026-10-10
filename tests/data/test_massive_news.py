import json
from datetime import date

import pytest
import requests

from capstone.data.massive_news import (
    API_KEY_ENV,
    MIN_REQUEST_INTERVAL,
    NEWS_URL,
    MassiveAuthError,
    MassiveCredentialsError,
    MassiveNewsClient,
    MassivePaginationError,
    MassiveRequestError,
    MassiveResponseError,
    MassiveStorageError,
    _read_jsonl_ids,
    collect_symbol,
    load_api_key,
    month_chunks,
    validate_next_url,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None, invalid_json=False):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("invalid JSON")
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def article(article_id, published_utc="2024-01-01T12:00:00Z"):
    return {"id": article_id, "title": f"Headline {article_id}", "published_utc": published_utc}


def make_client(responses, *, min_interval=MIN_REQUEST_INTERVAL, **kwargs):
    clock = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    client = MassiveNewsClient(
        "test-key",
        session=FakeSession(responses),
        sleep=sleep,
        monotonic=lambda: clock[0],
        min_interval=min_interval,
        **kwargs,
    )
    return client, waits


def test_missing_key_error_names_variable_only(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(MassiveCredentialsError, match=API_KEY_ENV):
        load_api_key()


def test_load_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "fake-test-key")
    assert load_api_key() == "fake-test-key"


def test_client_rejects_interval_faster_than_five_requests_per_minute():
    with pytest.raises(ValueError, match="12.0"):
        MassiveNewsClient("test-key", session=FakeSession([]), min_interval=11.99)


def test_news_request_uses_bearer_auth_and_published_utc_range():
    client, _ = make_client([FakeResponse({"results": []})])
    assert list(client.iter_articles(" nvda ", date(2024, 1, 1), date(2024, 2, 1))) == []
    url, call = client._session.calls[0]
    assert url == NEWS_URL
    assert client._session.headers["Authorization"] == "Bearer test-key"
    assert call["params"] == {
        "ticker": "NVDA",
        "published_utc.gte": "2024-01-01T00:00:00Z",
        "published_utc.lt": "2024-02-01T00:00:00Z",
        "sort": "published_utc",
        "order": "asc",
        "limit": 1000,
    }
    assert call["allow_redirects"] is False


def test_pagination_follows_next_url_and_keeps_authorization_header():
    next_url = f"{NEWS_URL}?cursor=opaque-cursor"
    client, _ = make_client(
        [
            FakeResponse({"results": [article("a")], "next_url": next_url}),
            FakeResponse({"results": [article("b")]}),
        ]
    )
    assert [item["id"] for item in client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2))] == [
        "a",
        "b",
    ]
    assert client._session.calls[1][0] == next_url
    assert client._session.calls[1][1]["params"] is None
    assert client._session.headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    "url",
    [
        "http://api.massive.com/v2/reference/news?cursor=x",
        "https://evil.example/v2/reference/news?cursor=x",
        "https://api.massive.com.evil.example/v2/reference/news?cursor=x",
        "https://api.massive.com/v2/reference/tickers?cursor=x",
        "https://user:pass@api.massive.com/v2/reference/news?cursor=x",
        "https://api.massive.com:444/v2/reference/news?cursor=x",
    ],
)
def test_validate_next_url_rejects_untrusted_urls(url):
    with pytest.raises(MassivePaginationError):
        validate_next_url(url)


def test_validate_next_url_accepts_massive_default_https_port():
    value = "https://api.massive.com:443/v2/reference/news?cursor=x"
    assert validate_next_url(value) == value


def test_pagination_rejects_unsafe_next_url_before_following_it():
    client, _ = make_client([FakeResponse({"results": [], "next_url": "https://evil.example/news"})])
    with pytest.raises(MassivePaginationError):
        list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)))
    assert len(client._session.calls) == 1


def test_missing_results_and_invalid_article_fields_raise_response_error():
    for payload in ({}, {"results": "wrong"}, {"results": [None]}, {"results": [{"id": "x"}]}):
        client, _ = make_client([FakeResponse(payload)])
        with pytest.raises(MassiveResponseError):
            list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)))


def test_invalid_json_raises_response_error_without_retry():
    client, _ = make_client([FakeResponse(invalid_json=True)])
    with pytest.raises(MassiveResponseError):
        list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)))
    assert client.stats.requests == 1


def test_auth_and_non_retryable_http_errors():
    client, _ = make_client([FakeResponse(status_code=401)])
    with pytest.raises(MassiveAuthError):
        client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)).__next__()

    client, _ = make_client([FakeResponse(status_code=404)])
    with pytest.raises(MassiveRequestError, match="404"):
        client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)).__next__()


def test_429_retries_with_retry_after_and_respects_request_spacing():
    client, waits = make_client(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "20"}),
            FakeResponse({"results": []}),
        ],
        max_retries=1,
    )
    list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)))
    assert client.stats.requests == 2 and client.stats.retries == 1
    assert waits[0] >= 20
    assert sum(waits) >= 20


@pytest.mark.parametrize(
    "error",
    [requests.exceptions.ConnectionError("offline"), requests.exceptions.Timeout("timeout")],
)
def test_transient_network_errors_are_retried(error):
    client, _ = make_client([error, FakeResponse({"results": []})], max_retries=1)
    assert list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2))) == []
    assert client.stats.requests == 2 and client.stats.retries == 1


def test_retry_limit_raises_request_error():
    client, _ = make_client([FakeResponse(status_code=500)] * 3, max_retries=2)
    with pytest.raises(MassiveRequestError, match="재시도 한도"):
        list(client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)))
    assert client.stats.requests == 3


def test_month_chunks_are_half_open_and_include_end_day():
    assert month_chunks(date(2023, 12, 31), date(2024, 1, 1)) == [
        ("2023-12", date(2023, 12, 31), date(2024, 1, 1)),
        ("2024-01", date(2024, 1, 1), date(2024, 1, 2)),
    ]


def test_collect_saves_by_published_utc_year_and_deduplicates(tmp_path):
    client, _ = make_client(
        [
            FakeResponse({"results": [article("dec", "2023-12-31T23:59:59Z")]}),
            FakeResponse({"results": [article("jan"), article("jan")]}),
        ]
    )
    assert collect_symbol(
        client,
        "nvda",
        date(2023, 12, 31),
        date(2024, 1, 1),
        tmp_path,
        progress=lambda _: None,
    ) == 2
    ticker_dir = tmp_path / "NVDA"
    assert [json.loads(line)["id"] for line in (ticker_dir / "2023.jsonl").read_text().splitlines()] == ["dec"]
    assert [json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text().splitlines()] == ["jan"]


def test_collect_deduplicates_ids_already_on_disk(tmp_path):
    ticker_dir = tmp_path / "NVDA"
    ticker_dir.mkdir()
    (ticker_dir / "2024.jsonl").write_text(json.dumps(article("existing")) + "\n", encoding="utf-8")
    client, _ = make_client([FakeResponse({"results": [article("existing"), article("new")]})])
    saved = collect_symbol(
        client, "NVDA", date(2024, 1, 1), date(2024, 1, 1), tmp_path, progress=lambda _: None
    )
    assert saved == 1
    rows = [json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text().splitlines()]
    assert rows == ["existing", "new"]


def test_completed_months_are_skipped_on_resume(tmp_path):
    client, _ = make_client([FakeResponse({"results": []})])
    assert collect_symbol(
        client, "NVDA", date(2024, 1, 1), date(2024, 1, 1), tmp_path, progress=lambda _: None
    ) == 0
    resumed, _ = make_client([])
    assert collect_symbol(
        resumed, "NVDA", date(2024, 1, 1), date(2024, 1, 1), tmp_path, progress=lambda _: None
    ) == 0
    assert resumed.stats.requests == 0 and resumed.stats.months_skipped == 1


def test_jsonl_reader_rejects_corrupt_records(tmp_path):
    path = tmp_path / "2024.jsonl"
    path.write_text('{"id":"good"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(MassiveStorageError, match="손상된 JSONL"):
        _read_jsonl_ids(path)
