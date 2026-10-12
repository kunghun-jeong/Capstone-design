import json
import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from capstone.data import massive_news
from capstone.data.massive_news import (
    API_KEY_ENV,
    MIN_REQUEST_INTERVAL,
    NEWS_URL,
    MassiveCredentialsError,
    MassiveNewsClient,
    MassivePermissionError,
    MassivePaginationError,
    MassiveRequestError,
    MassiveResponseError,
    MassiveStorageError,
    MassiveUnauthorizedError,
    _read_jsonl_ids,
    collect_symbol,
    load_api_key,
    validate_next_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_SPEC = importlib.util.spec_from_file_location(
    "fetch_massive_news", REPO_ROOT / "scripts" / "fetch_massive_news.py"
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI_MODULE)


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
    monkeypatch.setattr(massive_news, "load_dotenv", lambda **_: None)
    with pytest.raises(MassiveCredentialsError, match=API_KEY_ENV):
        load_api_key()


def test_cli_default_output_directory_is_ticker_data_root():
    assert CLI_MODULE.DEFAULT_OUT_DIR == REPO_ROOT / "data" / "raw" / "massive_news_ticker"


def test_cli_parses_dates_and_utc_z_datetimes():
    assert CLI_MODULE.parse_datetime("2026-10-12") == datetime(2026, 10, 12, tzinfo=timezone.utc)
    assert CLI_MODULE.parse_datetime("2026-10-12T09:15:30Z") == datetime(
        2026, 10, 12, 9, 15, 30, tzinfo=timezone.utc
    )


def test_cli_rejects_naive_datetime():
    with pytest.raises(CLI_MODULE.argparse.ArgumentTypeError, match="THH:MM:SSZ"):
        CLI_MODULE.parse_datetime("2026-10-12T09:00:00")


def test_cli_rejects_non_utc_datetime_offset():
    with pytest.raises(CLI_MODULE.argparse.ArgumentTypeError, match="THH:MM:SSZ"):
        CLI_MODULE.parse_datetime("2026-10-12T09:00:00+09:00")


def test_cli_uses_default_start_when_argument_is_omitted(monkeypatch):
    class FakeStats:
        def as_dict(self):
            return {"requests": 0}

    class FakeClient:
        stats = FakeStats()

    collected = []
    monkeypatch.setattr(
        "sys.argv", ["fetch_massive_news.py", "--symbols", "NVDA", "--end", "2026-10-13"]
    )
    monkeypatch.setattr(CLI_MODULE, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(CLI_MODULE, "MassiveNewsClient", lambda api_key: FakeClient())
    monkeypatch.setattr(CLI_MODULE, "collect_symbol", lambda *args: collected.append(args))
    monkeypatch.setattr(CLI_MODULE, "log_run", lambda *args: None)

    assert CLI_MODULE.main() == 0
    assert collected[0][2] == datetime(2016, 6, 22, tzinfo=timezone.utc)


def test_cli_uses_end_one_hour_before_current_utc_when_omitted(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 12, 14, 30, tzinfo=timezone.utc)

    class FakeStats:
        def as_dict(self):
            return {"requests": 0}

    class FakeClient:
        stats = FakeStats()

    collected = []
    monkeypatch.setattr(CLI_MODULE, "datetime", FixedDateTime)
    monkeypatch.setattr("sys.argv", ["fetch_massive_news.py", "--symbols", "NVDA"])
    monkeypatch.setattr(CLI_MODULE, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(CLI_MODULE, "MassiveNewsClient", lambda api_key: FakeClient())
    monkeypatch.setattr(CLI_MODULE, "collect_symbol", lambda *args: collected.append(args))
    monkeypatch.setattr(CLI_MODULE, "log_run", lambda *args: None)

    assert CLI_MODULE.main() == 0
    assert collected[0][3] == datetime(2026, 10, 12, 13, 30, tzinfo=timezone.utc)


def test_cli_returns_failure_when_run_log_cannot_be_saved(monkeypatch, capsys):
    class FakeStats:
        def as_dict(self):
            return {"requests": 0}

    class FakeClient:
        stats = FakeStats()

    def fail_to_log(*args, **kwargs):
        raise MassiveStorageError("run log is unavailable")

    monkeypatch.setattr(
        "sys.argv",
        [
            "fetch_massive_news.py",
            "--symbols",
            "NVDA",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
        ],
    )
    monkeypatch.setattr(CLI_MODULE, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(CLI_MODULE, "MassiveNewsClient", lambda api_key: FakeClient())
    monkeypatch.setattr(CLI_MODULE, "collect_symbol", lambda *args, **kwargs: 0)
    monkeypatch.setattr(CLI_MODULE, "log_run", fail_to_log)

    assert CLI_MODULE.main() == 1
    assert "실행 로그 저장 오류" in capsys.readouterr().err


def test_load_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "fake-test-key")
    monkeypatch.setattr(massive_news, "load_dotenv", lambda **_: None)
    assert load_api_key() == "fake-test-key"


def test_load_api_key_loads_repository_dotenv_before_reading_environment(monkeypatch):
    calls = []

    def fake_load_dotenv(*, dotenv_path):
        calls.append(dotenv_path)
        monkeypatch.setenv(API_KEY_ENV, "fake-dotenv-key")

    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr(massive_news, "load_dotenv", fake_load_dotenv)

    assert load_api_key() == "fake-dotenv-key"
    assert calls == [REPO_ROOT / ".env"]


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


def test_news_request_normalizes_datetime_offsets_to_utc():
    client, _ = make_client([FakeResponse({"results": []})])
    start = datetime(2026, 10, 12, 9, 0, tzinfo=timezone(timedelta(hours=9)))
    end = datetime(2026, 10, 12, 10, 30, tzinfo=timezone(timedelta(hours=9)))
    list(client.iter_articles("NVDA", start, end))
    params = client._session.calls[0][1]["params"]
    assert params["published_utc.gte"] == "2026-10-12T00:00:00Z"
    assert params["published_utc.lt"] == "2026-10-12T01:30:00Z"


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
    with pytest.raises(MassiveUnauthorizedError):
        client.iter_articles("NVDA", date(2024, 1, 1), date(2024, 1, 2)).__next__()

    client, _ = make_client([FakeResponse(status_code=403)])
    with pytest.raises(MassivePermissionError):
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


def test_collect_requests_full_range_once_paginates_and_saves_by_utc_year(tmp_path):
    next_url = f"{NEWS_URL}?cursor=year-boundary"
    client, _ = make_client(
        [
            FakeResponse(
                {"results": [article("dec", "2023-12-31T23:59:59Z")], "next_url": next_url}
            ),
            FakeResponse({"results": [article("jan", "2024-01-01T00:00:00Z"), article("jan")]})
        ]
    )
    assert collect_symbol(
        client,
        "nvda",
        date(2023, 12, 31),
        date(2024, 1, 2),
        tmp_path,
        progress=lambda _: None,
    ) == 2
    ticker_dir = tmp_path / "NVDA"
    assert len(client._session.calls) == 2
    first_url, first_call = client._session.calls[0]
    assert first_url == NEWS_URL
    assert first_call["params"]["published_utc.gte"] == "2023-12-31T00:00:00Z"
    assert first_call["params"]["published_utc.lt"] == "2024-01-02T00:00:00Z"
    assert client._session.calls[1][0] == next_url
    assert [json.loads(line)["id"] for line in (ticker_dir / "2023.jsonl").read_text().splitlines()] == ["dec"]
    assert [json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text().splitlines()] == ["jan"]
    state = json.loads((ticker_dir / "_state.json").read_text(encoding="utf-8"))
    assert state == {"last_successful_end_exclusive": "2024-01-02T00:00:00Z"}


def test_collect_deduplicates_ids_already_on_disk(tmp_path):
    ticker_dir = tmp_path / "NVDA"
    ticker_dir.mkdir()
    (ticker_dir / "2024.jsonl").write_text(json.dumps(article("existing")) + "\n", encoding="utf-8")
    client, _ = make_client([FakeResponse({"results": [article("existing"), article("new")]})])
    saved = collect_symbol(
        client, "NVDA", date(2024, 1, 1), date(2024, 1, 2), tmp_path, progress=lambda _: None
    )
    assert saved == 1
    rows = [json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text().splitlines()]
    assert rows == ["existing", "new"]


def test_completed_range_is_skipped_when_end_has_not_advanced(tmp_path):
    client, _ = make_client([FakeResponse({"results": []})])
    assert collect_symbol(
        client, "NVDA", date(2024, 1, 1), date(2024, 1, 2), tmp_path, progress=lambda _: None
    ) == 0
    resumed, _ = make_client([])
    assert collect_symbol(
        resumed, "NVDA", date(2024, 1, 1), date(2024, 1, 2), tmp_path, progress=lambda _: None
    ) == 0
    assert resumed.stats.requests == 0 and resumed.stats.tickers_skipped == 1
    state = json.loads((tmp_path / "NVDA" / "_state.json").read_text(encoding="utf-8"))
    assert state == {"last_successful_end_exclusive": "2024-01-02T00:00:00Z"}


def test_resume_starts_at_last_successful_end_and_saves_new_watermark(tmp_path):
    first, _ = make_client([FakeResponse({"results": [article("first", "2024-01-02T12:00:00Z")]})])
    assert collect_symbol(
        first, "NVDA", date(2024, 1, 1), date(2024, 1, 3), tmp_path, progress=lambda _: None
    ) == 1

    resumed, _ = make_client([FakeResponse({"results": [article("second", "2024-01-03T12:00:00Z")]})])
    assert collect_symbol(
        resumed, "NVDA", date(2024, 1, 1), date(2024, 1, 4), tmp_path, progress=lambda _: None
    ) == 1
    params = resumed._session.calls[0][1]["params"]
    assert params["published_utc.gte"] == "2024-01-03T00:00:00Z"
    assert params["published_utc.lt"] == "2024-01-04T00:00:00Z"
    state = json.loads((tmp_path / "NVDA" / "_state.json").read_text(encoding="utf-8"))
    assert state == {"last_successful_end_exclusive": "2024-01-04T00:00:00Z"}
    rows = [
        json.loads(line)["id"]
        for line in (tmp_path / "NVDA" / "2024.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows == ["first", "second"]


def test_legacy_completed_range_is_used_as_resume_watermark(tmp_path):
    ticker_dir = tmp_path / "NVDA"
    ticker_dir.mkdir()
    (ticker_dir / "_state.json").write_text(
        json.dumps(
            {
                "completed_tickers": {
                    "2024-01-01..2024-01-03": {
                        "start": "2024-01-01",
                        "end_exclusive": "2024-01-03",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client, _ = make_client([FakeResponse({"results": []})])
    assert collect_symbol(
        client, "NVDA", date(2024, 1, 1), date(2024, 1, 4), tmp_path, progress=lambda _: None
    ) == 0
    assert client._session.calls[0][1]["params"]["published_utc.gte"] == "2024-01-03T00:00:00Z"
    state = json.loads((ticker_dir / "_state.json").read_text(encoding="utf-8"))
    assert state == {"last_successful_end_exclusive": "2024-01-04T00:00:00Z"}


def test_failed_later_page_leaves_no_range_checkpoint_and_retry_deduplicates(tmp_path):
    next_url = f"{NEWS_URL}?cursor=second-page"
    first, _ = make_client(
        [
            FakeResponse({"results": [article("saved-before-failure")], "next_url": next_url}),
            requests.exceptions.ConnectionError("offline"),
        ],
        max_retries=0,
    )
    with pytest.raises(MassiveRequestError):
        collect_symbol(
            first, "NVDA", date(2024, 1, 1), date(2025, 1, 1), tmp_path, progress=lambda _: None
        )

    ticker_dir = tmp_path / "NVDA"
    assert not (ticker_dir / "_state.json").exists()
    assert [
        json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text(encoding="utf-8").splitlines()
    ] == ["saved-before-failure"]

    resumed, _ = make_client([FakeResponse({"results": [article("saved-before-failure"), article("new")]})])
    assert collect_symbol(
        resumed, "NVDA", date(2024, 1, 1), date(2025, 1, 1), tmp_path, progress=lambda _: None
    ) == 1
    assert [
        json.loads(line)["id"] for line in (ticker_dir / "2024.jsonl").read_text(encoding="utf-8").splitlines()
    ] == ["saved-before-failure", "new"]


def test_failed_incremental_request_keeps_previous_watermark(tmp_path):
    ticker_dir = tmp_path / "NVDA"
    ticker_dir.mkdir()
    state_path = ticker_dir / "_state.json"
    state_path.write_text(
        json.dumps({"last_successful_end_exclusive": "2024-01-03T00:00:00Z"}), encoding="utf-8"
    )
    client, _ = make_client(
        [
            FakeResponse({"results": [article("written-before-failure", "2024-01-03T12:00:00Z")], "next_url": f"{NEWS_URL}?cursor=next"}),
            requests.exceptions.ConnectionError("offline"),
        ],
        max_retries=0,
    )
    with pytest.raises(MassiveRequestError):
        collect_symbol(
            client, "NVDA", date(2024, 1, 1), date(2024, 1, 5), tmp_path, progress=lambda _: None
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_successful_end_exclusive"] == "2024-01-03T00:00:00Z"


def test_jsonl_reader_rejects_corrupt_records(tmp_path):
    path = tmp_path / "2024.jsonl"
    path.write_text('{"id":"good"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(MassiveStorageError, match="손상된 JSONL"):
        _read_jsonl_ids(path)
