"""FNSPID 뉴스 데이터셋에서 우리 종목만 걸러내는 파일럿 (하드코딩 버전)

FNSPID 뉴스 파일은 5.7GB / 23GB라 통째로 받지 않는다.
허깅페이스에서 조금씩(청크 단위) 읽으면서 우리 종목 행만 남겨 저장한다.

준비:
  pip install pandas

실행:
  python fnspid_news_pilot.py               # 맛보기: 앞부분 몇 청크만 읽고 멈춤
  python fnspid_news_pilot.py --full        # 파일 끝까지 전부 읽기 (오래 걸림, 인터넷 속도에 따라 수십 분~몇 시간)
  python fnspid_news_pilot.py --file nasdaq --full   # 본문이 들어있는 23GB 파일로

결과:
  fnspid_<파일>_filtered.csv   우리 종목 기사만 모은 파일
  화면에 종목별·연도별 기사 수, 본문/제목 비어있는 비율 출력
"""

import argparse
import time

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------- 여기만 바꾸면 됨 ----------------
# 팀 레포 configs/universe.yaml(알파카 PR)과 같은 임시 목록
TICKERS = ["NVDA", "AMD", "TSM", "MU", "SNDK", "MPWR", "VICR", "ADI", "ON"]
CHUNK_ROWS = 200_000      # 한 번에 읽을 행 수 (메모리 부족하면 줄이기)
PREVIEW_CHUNKS = 5        # --full 없이 돌릴 때 읽을 청크 수
# --------------------------------------------------

BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"
FILES = {
    "all": "All_external.csv",            # 약 5.7GB
    "nasdaq": "nasdaq_exteral_data.csv",  # 약 23GB (철자 exteral이 원래 파일명)
}


def find_symbol_col(columns) -> str:
    for c in columns:
        if c.strip().lower() in ("stock_symbol", "symbol", "ticker"):
            return c
    raise ValueError(f"종목 컬럼을 못 찾음. 실제 컬럼: {list(columns)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", choices=FILES, default="all")
    p.add_argument("--full", action="store_true", help="파일 끝까지 읽기")
    args = p.parse_args()

    url = BASE + FILES[args.file]
    out_path = f"fnspid_{args.file}_filtered.csv"
    wanted = set(TICKERS)
    print(f"읽는 중: {url}")

    reader = pd.read_csv(
        url,
        chunksize=CHUNK_ROWS,
        dtype=str,              # 숫자/문자 섞인 컬럼 때문에 깨지지 않게 전부 문자열로
        on_bad_lines="skip",    # 형식이 깨진 줄은 건너뜀
    )

    kept, total_rows, first = [], 0, True
    started = time.time()
    for i, chunk in enumerate(reader, 1):
        if first:
            sym_col = find_symbol_col(chunk.columns)
            print(f"컬럼: {list(chunk.columns)}\n")
            first = False

        total_rows += len(chunk)
        sym = chunk[sym_col].fillna("").str.strip().str.upper()
        hit = chunk[sym.isin(wanted)]
        if not hit.empty:
            kept.append(hit)

        n_kept = sum(len(k) for k in kept)
        print(f"청크 {i}: 누적 {total_rows:,}행 읽음 / 우리 종목 {n_kept:,}행 ({time.time() - started:.0f}초)")

        if not args.full and i >= PREVIEW_CHUNKS:
            print(f"\n맛보기라 {PREVIEW_CHUNKS}청크에서 멈춤. 전부 받으려면 --full")
            break

    if not kept:
        print("\n우리 종목 기사를 하나도 못 찾음 (맛보기 범위에 없을 수 있음 → --full로 다시)")
        return

    df = pd.concat(kept, ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path} ({len(df):,}행)")
    summarize(df, sym_col)


def summarize(df: pd.DataFrame, sym_col: str):
    df = df.copy()
    df["_sym"] = df[sym_col].str.strip().str.upper()

    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if date_col:
        df["_year"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.year
        print("\n[종목별 x 연도별 기사 수]")
        print(df.pivot_table(index="_sym", columns="_year", values=sym_col, aggfunc="count", fill_value=0))
        print("\n[날짜 예시 — 시간대 표시가 있는지 확인용]")
        print(df[date_col].head(3).to_string(index=False))

    pub_col = next((c for c in df.columns if c.strip().lower() == "publisher"), None)
    if pub_col:
        print("\n[언론사별 기사 수 — 상위 10개]")
        print(df[pub_col].fillna("(빈칸)").value_counts().head(10).to_string())

    if date_col:
        print(f"\n[기간] {df[date_col].min()}  ~  {df[date_col].max()}")

    print("\n[컬럼별 비어있는 비율]")
    for c in df.columns:
        if c.startswith("_"):
            continue
        empty = df[c].isna() | (df[c].str.strip() == "")
        print(f"  {c}: {empty.mean():.1%}")


if __name__ == "__main__":
    main()
