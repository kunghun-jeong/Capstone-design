"""FNSPID 뉴스 데이터셋(Hugging Face `Zihan1004/FNSPID`)에서 대상 종목 기사만 걸러내는 수집기.

- 뉴스 CSV가 5.7GB / 23GB라 통째로 내려받지 않고, 청크 단위로 스트리밍하며 `Stock_symbol`로 필터링한다.
- 원본 컬럼은 가공하지 않고 그대로 저장한다 (전부 문자열). 필드 수가 맞지 않는 깨진 줄은 건너뛴다.
  단, 깨진 줄이 청크의 첫 줄에 걸리면 pandas가 건너뛰지 않고 앞쪽 필드만 읽는다 (pandas 동작, 빈도는 미확인).
- 결과는 임시 파일에 쓴 뒤 끝까지 성공했을 때만 최종 경로로 바꾼다 — 중간에 끊기면 이전 결과가 깨지지 않는다.
- 데이터셋 라이선스는 CC BY-NC 4.0. 결과 CSV는 `data/raw/`(gitignore)에만 두고 커밋하지 않는다.

조사 결과와 주의점(시각 정밀도, 두 파일의 관계 등)은 `reports/fnspid_news_report.md` 참고.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"
FILES = {
    # 제목 전용, 2009~2020-06. nasdaq 파일에 포함된 것으로 보임 (리포트 3절)
    "all": "All_external.csv",
    # 본문 약 70% 포함, 2009-08~2024-01. 철자 `exteral`은 원본 파일명 그대로
    "nasdaq": "nasdaq_exteral_data.csv",
}

# configs/universe.yaml이 아직 main에 없을 때 쓰는 기본값 (알파카 PR의 임시 유니버스와 동일)
DEFAULT_TICKERS = ["NVDA", "AMD", "TSM", "MU", "SNDK", "MPWR", "VICR", "ADI", "ON"]

SYMBOL_COLUMNS = ("stock_symbol", "symbol", "ticker")
DEFAULT_CHUNK_ROWS = 200_000


def source_url(file_key: str) -> str:
    if file_key not in FILES:
        raise ValueError(f"알 수 없는 파일: {file_key!r} (가능: {sorted(FILES)})")
    return BASE_URL + FILES[file_key]


def load_tickers(config_path: Path | None) -> list[str]:
    """`configs/universe.yaml`이 있으면 거기서, 없으면 DEFAULT_TICKERS를 쓴다."""
    if config_path is None or not Path(config_path).exists():
        return list(DEFAULT_TICKERS)
    import yaml

    with open(config_path, encoding="utf-8") as f:
        tickers = yaml.safe_load(f)["tickers"]
    bad = [t for t in tickers if not isinstance(t, str)]
    if bad:  # 따옴표 없는 ON은 YAML에서 bool(True)로 파싱된다
        raise ValueError(f"티커는 문자열이어야 한다 (따옴표 확인): {bad}")
    return [t.strip().upper() for t in tickers]


def find_symbol_column(columns: Iterable[str]) -> str:
    for c in columns:
        if str(c).strip().lower() in SYMBOL_COLUMNS:
            return c
    raise ValueError(f"종목 컬럼을 찾지 못함. 실제 컬럼: {list(columns)}")


def normalize_symbols(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper()


@dataclass
class FilterStats:
    chunks: int = 0
    rows_read: int = 0
    rows_kept: int = 0
    stopped_early: bool = False
    columns: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "started_at"}
        d["elapsed_sec"] = round(time.time() - self.started_at, 1)
        return d


def read_chunks(source: str | Path, chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """URL 또는 로컬 경로의 CSV를 청크 단위로 읽는다."""
    return pd.read_csv(
        source,
        chunksize=chunk_rows,
        dtype=str,  # 숫자/문자가 섞인 컬럼이 있어 전부 문자열로 읽는다
        on_bad_lines="skip",
    )


def filter_chunks(
    chunks: Iterable[pd.DataFrame],
    symbols: Iterable[str],
    *,
    max_chunks: int | None = None,
    stats: FilterStats | None = None,
    progress: Callable[[str], None] = print,
):
    """청크를 순회하며 대상 종목 행만 내보낸다 (청크마다 DataFrame 하나, 비어 있으면 건너뜀)."""
    wanted = {s.strip().upper() for s in symbols}
    stats = stats if stats is not None else FilterStats()
    sym_col = None
    for chunk in chunks:
        if sym_col is None:
            sym_col = find_symbol_column(chunk.columns)
            stats.columns = list(chunk.columns)
        stats.chunks += 1
        stats.rows_read += len(chunk)
        hit = chunk[normalize_symbols(chunk[sym_col]).isin(wanted)]
        stats.rows_kept += len(hit)
        progress(
            f"청크 {stats.chunks}: 누적 {stats.rows_read:,}행 / 대상 종목 {stats.rows_kept:,}행 "
            f"({time.time() - stats.started_at:.0f}초)"
        )
        if not hit.empty:
            yield hit
        if max_chunks is not None and stats.chunks >= max_chunks:
            stats.stopped_early = True
            return


def collect(
    source: str | Path,
    symbols: Iterable[str],
    out_path: Path,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    max_chunks: int | None = None,
    progress: Callable[[str], None] = print,
) -> FilterStats:
    """source를 끝까지(또는 max_chunks까지) 읽어 대상 종목 행을 out_path CSV로 저장한다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    stats = FilterStats()
    wrote_header = False
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            for hit in filter_chunks(
                read_chunks(source, chunk_rows), symbols, max_chunks=max_chunks, stats=stats, progress=progress
            ):
                hit.to_csv(f, index=False, header=not wrote_header)
                wrote_header = True
            if not wrote_header and stats.columns:  # 한 건도 없으면 헤더만 남긴다
                pd.DataFrame(columns=stats.columns).to_csv(f, index=False)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out_path)
    return stats


# ------------------------------------------------------------------ 요약 ----


def _col(df: pd.DataFrame, name: str) -> str | None:
    return next((c for c in df.columns if str(c).strip().lower() == name), None)


def summarize(df: pd.DataFrame) -> dict:
    """리포트용 요약: 종목x연도 기사 수, 컬럼별 빈칸 비율, 언론사 상위, 기간."""
    out: dict = {"rows": len(df)}
    sym_col = find_symbol_column(df.columns)
    sym = normalize_symbols(df[sym_col])

    date_col = _col(df, "date")
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        out["by_symbol_year"] = (
            pd.crosstab(sym, dates.dt.year.astype("Int64")).rename_axis(index="symbol", columns="year")
        )
        out["date_min"], out["date_max"] = dates.min(), dates.max()
        # 정각(분·초 0) 비율 — 시각 정밀도 점검용 (리포트 5절)
        valid = dates.dropna()
        out["on_the_hour_ratio"] = (
            float(((valid.dt.minute == 0) & (valid.dt.second == 0)).mean()) if len(valid) else float("nan")
        )

    out["empty_ratio"] = {
        c: float((df[c].isna() | (df[c].astype(str).str.strip() == "")).mean()) if len(df) else 0.0
        for c in df.columns
    }

    pub_col = _col(df, "publisher")
    if pub_col is not None:
        out["top_publishers"] = df[pub_col].fillna("(빈칸)").value_counts().head(10)
    return out


def print_summary(summary: dict, out: Callable[[str], None] = print) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        out(f"\n총 {summary['rows']:,}행")
        if "by_symbol_year" in summary:
            out("\n[종목별 x 연도별 기사 수]")
            out(summary["by_symbol_year"].to_string())
            out(f"\n[기간] {summary['date_min']} ~ {summary['date_max']}")
            out(f"[정각(:00:00) 시각 비율] {summary['on_the_hour_ratio']:.1%}")
        if "top_publishers" in summary:
            out("\n[언론사 상위 10]")
            out(summary["top_publishers"].to_string())
        out("\n[컬럼별 빈칸 비율]")
        for c, r in summary["empty_ratio"].items():
            out(f"  {c}: {r:.1%}")
