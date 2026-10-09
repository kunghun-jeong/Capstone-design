import math

import pandas as pd
import pytest

from capstone.data.fnspid_news import (
    DEFAULT_TICKERS,
    collect,
    filter_chunks,
    find_symbol_column,
    load_tickers,
    source_url,
    summarize,
)

HEADER = "Unnamed: 0,Date,Article_title,Stock_symbol,Url,Publisher,Author,Article\n"


def write_csv(path, body: str):
    path.write_text(HEADER + body, encoding="utf-8")
    return path


@pytest.fixture
def sample_csv(tmp_path):
    body = (
        '0,2023-12-16 09:00:00 UTC,NVDA beats,NVDA,u0,,,"body with, comma"\n'
        '1,2023-12-16 10:30:15 UTC,Apple stuff,AAPL,u1,,,x\n'
        '2,2015-03-01 00:00:00 UTC,ON news, on ,u2,Zacks,,\n'  # 공백·소문자 티커
        '3,2015-03-02 00:00:00 UTC,Multiline,MU,u3,,,"line1\nline2"\n'  # 본문 안 줄바꿈
        "4,2015-03-03 00:00:00 UTC,broken,MU,u4,,,a,b,c,d\n"  # 필드 수가 많은 깨진 줄
        "5,2016-01-01 12:00:00 UTC,AMD up,AMD,u5,Benzinga,,\n"
    )
    return write_csv(tmp_path / "news.csv", body)


def test_find_symbol_column_variants():
    assert find_symbol_column(["Date", "Stock_symbol"]) == "Stock_symbol"
    assert find_symbol_column(["date", " ticker "]) == " ticker "
    with pytest.raises(ValueError):
        find_symbol_column(["Date", "Title"])


def test_source_url():
    assert source_url("nasdaq").endswith("Stock_news/nasdaq_exteral_data.csv")
    with pytest.raises(ValueError):
        source_url("nope")


def test_load_tickers_default_when_config_missing(tmp_path):
    assert load_tickers(tmp_path / "missing.yaml") == DEFAULT_TICKERS
    assert load_tickers(None) == DEFAULT_TICKERS


def test_load_tickers_rejects_unquoted_on(tmp_path):
    cfg = tmp_path / "u.yaml"
    cfg.write_text("tickers:\n  - nvda\n  - ON\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tickers(cfg)
    cfg.write_text('tickers:\n  - nvda\n  - "ON"\n', encoding="utf-8")
    assert load_tickers(cfg) == ["NVDA", "ON"]


def test_collect_filters_symbols_and_keeps_raw(sample_csv, tmp_path):
    out = tmp_path / "out" / "nasdaq_filtered.csv"
    # 깨진 줄이 청크 첫 줄에 걸리면 pandas가 건너뛰지 않고 앞 필드만 읽는다 → 경계에 안 걸리게 청크를 크게
    stats = collect(sample_csv, ["nvda", "MU", "ON"], out, chunk_rows=10, progress=lambda _: None)

    df = pd.read_csv(out, dtype=str)
    assert sorted(df["Url"]) == ["u0", "u2", "u3"]  # AAPL, AMD 제외, 깨진 줄(u4) 건너뜀
    assert df.loc[df["Url"] == "u3", "Article"].item() == "line1\nline2"
    assert df.loc[df["Url"] == "u0", "Article"].item() == "body with, comma"
    assert df.loc[df["Url"] == "u2", "Stock_symbol"].item() == " on "  # 원본 값은 가공하지 않음
    assert stats.rows_kept == 3 and not stats.stopped_early
    assert not out.with_suffix(".csv.part").exists()


def test_collect_max_chunks_stops_early(sample_csv, tmp_path):
    out = tmp_path / "o.csv"
    stats = collect(sample_csv, ["NVDA", "AMD"], out, chunk_rows=2, max_chunks=1, progress=lambda _: None)
    assert stats.stopped_early and stats.chunks == 1
    assert pd.read_csv(out, dtype=str)["Url"].tolist() == ["u0"]


def test_collect_no_match_writes_header_only(sample_csv, tmp_path):
    out = tmp_path / "o.csv"
    collect(sample_csv, ["TSLA"], out, progress=lambda _: None)
    df = pd.read_csv(out, dtype=str)
    assert df.empty and "Stock_symbol" in df.columns


def test_collect_failure_keeps_previous_result(sample_csv, tmp_path, monkeypatch):
    out = tmp_path / "o.csv"
    out.write_text("previous", encoding="utf-8")

    def boom(*a, **k):
        yield pd.read_csv(sample_csv, dtype=str, nrows=1)
        raise ConnectionResetError("원격 호스트에 의해 강제로 끊김")

    monkeypatch.setattr("capstone.data.fnspid_news.read_chunks", boom)
    with pytest.raises(ConnectionResetError):
        collect(sample_csv, ["NVDA"], out, progress=lambda _: None)
    assert out.read_text(encoding="utf-8") == "previous"
    assert not out.with_suffix(".csv.part").exists()


def test_filter_chunks_counts():
    chunks = [
        pd.DataFrame({"Stock_symbol": ["NVDA", "AAPL"]}),
        pd.DataFrame({"Stock_symbol": ["AAPL", None]}),
        pd.DataFrame({"Stock_symbol": ["amd"]}),
    ]
    hits = list(filter_chunks(chunks, ["NVDA", "AMD"], progress=lambda _: None))
    assert [len(h) for h in hits] == [1, 1]  # 빈 청크는 내보내지 않음


def test_summarize(sample_csv):
    df = pd.read_csv(sample_csv, dtype=str, on_bad_lines="skip")
    s = summarize(df)
    assert s["rows"] == 5
    assert s["by_symbol_year"].loc["ON", 2015] == 1
    assert s["by_symbol_year"].loc["NVDA", 2023] == 1
    assert s["empty_ratio"]["Author"] == 1.0
    assert s["empty_ratio"]["Publisher"] == pytest.approx(3 / 5)
    assert s["top_publishers"]["(빈칸)"] == 3
    # 정각 시각: 09:00:00, 00:00:00, 00:00:00, 12:00:00 → 4/5
    assert math.isclose(s["on_the_hour_ratio"], 4 / 5)
