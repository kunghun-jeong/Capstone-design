import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from capstone.data.reddit_arctic_shift import (
    MIN_REQUEST_INTERVAL,
    PAGE_LIMIT,
    ArcticShiftClient,
    RedditConfigError,
    RedditPaginationError,
    RedditRequestError,
    Target,
    collect_unit,
    load_targets,
    month_units,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_SPEC = importlib.util.spec_from_file_location("fetch_reddit", REPO_ROOT / "scripts" / "fetch_reddit.py")
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI_MODULE)


def ts(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs["params"]))
        return self.responses.pop(0)


def page(*items):
    return FakeResponse({"data": list(items)})


def comment(item_id, created, **extra):
    return {"id": item_id, "created_utc": created, "body": f"text {item_id}", "all_awardings": [], **extra}


def make_client(responses):
    clock = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    client = ArcticShiftClient(session=FakeSession(responses), sleep=sleep, monotonic=lambda: clock[0])
    return client, waits


def test_month_units_split_at_utc_month_boundaries():
    units = month_units(ts("2026-04-24"), ts("2026-06-02"))
    assert units == [
        ("2026-04", ts("2026-04-24"), ts("2026-05-01")),
        ("2026-05", ts("2026-05-01"), ts("2026-06-01")),
        ("2026-06", ts("2026-06-01"), ts("2026-06-02")),
    ]


def test_month_units_handles_december_and_rejects_empty_range():
    assert [m for m, _, _ in month_units(ts("2025-12-31"), ts("2026-01-02"))] == ["2025-12", "2026-01"]
    with pytest.raises(RedditConfigError):
        month_units(ts("2026-01-02"), ts("2026-01-02"))


def test_load_targets_expands_kinds(tmp_path):
    config = tmp_path / "subs.yaml"
    config.write_text("subreddits:\n  - name: stocks\n    kinds: [posts]\n  - name: AMD_Stock\n", encoding="utf-8")
    assert load_targets(config) == [
        Target("stocks", "posts"), Target("AMD_Stock", "posts"), Target("AMD_Stock", "comments"),
    ]


def test_load_targets_rejects_unknown_kind(tmp_path):
    config = tmp_path / "subs.yaml"
    config.write_text("subreddits:\n  - name: stocks\n    kinds: [votes]\n", encoding="utf-8")
    with pytest.raises(RedditConfigError, match="kinds"):
        load_targets(config)


def test_client_rejects_interval_below_minimum():
    with pytest.raises(ValueError):
        ArcticShiftClient(min_interval=MIN_REQUEST_INTERVAL / 2)


def test_get_page_retries_server_timeout_422_then_succeeds():
    client, waits = make_client([
        FakeResponse({"data": None, "error": "Timeout. Maybe slow down a bit"}, status_code=422),
        page(comment("a", 1)),
    ])
    assert client.get_page("comments", {"subreddit": "stocks"}) == [comment("a", 1)]
    assert client.stats.server_timeouts_422 == 1
    assert client.stats.retries == 1
    assert 2.0 in waits  # 첫 재시도 백오프


def test_get_page_waits_for_rate_limit_reset_header():
    client, waits = make_client([
        FakeResponse({"error": "rate limited"}, status_code=429, headers={"x-ratelimit-reset": "30"}),
        page(),
    ])
    assert client.get_page("posts", {}) == []
    assert client.stats.rate_limited_429 == 1
    assert 31.0 in waits


def test_get_page_does_not_retry_bad_request():
    client, _ = make_client([FakeResponse({"data": None, "error": "'x' is not a valid field"}, status_code=400)])
    with pytest.raises(RedditRequestError, match="not a valid field"):
        client.get_page("posts", {})
    assert client.stats.requests == 1


def test_get_page_gives_up_after_max_retries():
    timeout = FakeResponse({"error": "Timeout"}, status_code=422)
    client, _ = make_client([timeout] * 9)
    with pytest.raises(RedditRequestError, match="재시도 한도"):
        client.get_page("comments", {})
    assert client.stats.requests == 9


def test_iter_items_requests_next_page_from_last_second_inclusive():
    full = [comment(f"c{i}", 1000 + i // 2) for i in range(PAGE_LIMIT)]  # 마지막 두 건이 같은 초(1049)
    rest = [full[-1], comment("tail", 1049), comment("next", 1050)]
    client, _ = make_client([page(*full), page(*rest)])
    items = list(client.iter_items("stocks", "comments", 999, 2000))
    calls = client._session.calls
    assert calls[0][1]["after"] == 999 and calls[0][1]["before"] == 2000
    assert calls[1][1]["after"] == 1048  # 마지막 created_utc(1049) − 1
    ids = [item["id"] for item, _ in items]
    assert ids.count(full[-1]["id"]) == 2  # 경계 중복은 호출 쪽에서 거른다
    assert "tail" in ids and "next" in ids


def test_iter_items_raises_when_page_cannot_advance():
    stuck = [comment(f"s{i}", 1000) for i in range(PAGE_LIMIT)]
    client, _ = make_client([page(*stuck), page(*stuck)])
    with pytest.raises(RedditPaginationError, match="같은 초"):
        list(client.iter_items("stocks", "comments", 999, 2000))


def test_collect_unit_writes_trimmed_month_file_and_completes(tmp_path):
    client, _ = make_client([page(comment("a", ts("2026-05-01")), comment("b", ts("2026-05-02")))])
    target = Target("stocks", "comments")
    saved = collect_unit(client, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path, progress=lambda _: None)

    assert saved == 2
    params = client._session.calls[0][1]
    assert params["after"] == ts("2026-05-01") - 1  # 월 시작 시각 정각 항목도 포함
    assert params["before"] == ts("2026-06-01")
    rows = [json.loads(line) for line in (tmp_path / "stocks" / "comments" / "2026-05.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a", "b"]
    assert "all_awardings" not in rows[0] and rows[0]["body"] == "text a"
    state = json.loads((tmp_path / "stocks" / "comments" / "2026-05.state.json").read_text())
    assert state["complete_until"] == "2026-06-01T00:00:00Z"


def test_collect_unit_skips_completed_month_without_requests(tmp_path):
    target = Target("stocks", "posts")
    client, _ = make_client([page(comment("a", ts("2026-05-03")))])
    collect_unit(client, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path, progress=lambda _: None)

    again, _ = make_client([])
    assert collect_unit(again, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path, progress=lambda _: None) == 0
    assert again.stats.units_skipped == 1 and again.stats.requests == 0


def test_collect_unit_resumes_from_cursor_and_skips_saved_ids(tmp_path):
    target = Target("stocks", "comments")
    first_page = [comment(f"c{i}", ts("2026-05-01") + i) for i in range(PAGE_LIMIT)]
    stop_after_first = iter([False, True])
    client, _ = make_client([page(*first_page), page(first_page[-1], comment("late", ts("2026-05-20")))])
    collect_unit(
        client, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path,
        progress=lambda _: None, should_stop=lambda: next(stop_after_first),
    )
    state_path = tmp_path / "stocks" / "comments" / "2026-05.state.json"
    state = json.loads(state_path.read_text())
    assert state["complete_until"] is None
    assert state["next_after"] == first_page[-1]["created_utc"] - 1

    resumed, _ = make_client([page(first_page[-1], comment("late", ts("2026-05-20")))])
    saved = collect_unit(resumed, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path, progress=lambda _: None)
    assert resumed._session.calls[0][1]["after"] == first_page[-1]["created_utc"] - 1
    assert saved == 1  # 경계의 c99는 이미 저장돼 있어 건너뜀
    lines = (tmp_path / "stocks" / "comments" / "2026-05.jsonl").read_text().splitlines()
    assert len(lines) == PAGE_LIMIT + 1
    assert json.loads(state_path.read_text())["complete_until"] == "2026-06-01T00:00:00Z"


def test_collect_unit_extends_partial_month_when_end_moves_later(tmp_path):
    target = Target("stocks", "posts")
    client, _ = make_client([page(comment("a", ts("2026-10-05")))])
    collect_unit(client, target, "2026-10", ts("2026-10-01"), ts("2026-10-10"), tmp_path, progress=lambda _: None)

    later, _ = make_client([page(comment("b", ts("2026-10-12")))])
    assert collect_unit(later, target, "2026-10", ts("2026-10-01"), ts("2026-10-20"), tmp_path, progress=lambda _: None) == 1
    assert later._session.calls[0][1]["after"] == ts("2026-10-10") - 1


def test_collect_unit_refuses_backfill_before_recorded_start(tmp_path):
    target = Target("stocks", "posts")
    client, _ = make_client([page()])
    collect_unit(client, target, "2026-05", ts("2026-05-10"), ts("2026-06-01"), tmp_path, progress=lambda _: None)
    earlier, _ = make_client([])
    with pytest.raises(RedditConfigError, match="out-dir"):
        collect_unit(earlier, target, "2026-05", ts("2026-05-01"), ts("2026-06-01"), tmp_path, progress=lambda _: None)


def test_cli_parses_dates_and_defaults():
    assert CLI_MODULE.parse_datetime("2026-10-24") == datetime(2026, 10, 24, tzinfo=timezone.utc)
    with pytest.raises(CLI_MODULE.argparse.ArgumentTypeError):
        CLI_MODULE.parse_datetime("2026-10-24T09:00:00+09:00")
    end = CLI_MODULE.default_end(datetime(2026, 10, 26, 13, 45, tzinfo=timezone.utc))
    assert end == datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)
    assert CLI_MODULE.default_start(end) == datetime(2026, 4, 25, tzinfo=timezone.utc)


def test_cli_select_targets_uses_config_kinds_and_filters():
    targets = [Target("stocks", "posts"), Target("stocks", "comments"), Target("wallstreetbets", "comments")]
    assert CLI_MODULE.select_targets(targets, None, "comments") == [
        Target("stocks", "comments"), Target("wallstreetbets", "comments"),
    ]
    assert CLI_MODULE.select_targets(targets, "wallstreetbets", None) == [Target("wallstreetbets", "comments")]
    assert CLI_MODULE.select_targets(targets, "options", "posts") == [Target("options", "posts")]
    with pytest.raises(RedditConfigError):
        CLI_MODULE.select_targets(targets, None, "votes")
