# Capstone-design

뉴스·Reddit 여론에 담긴 "기대 심리"를 수치화해 미국 AI 섹터 10개 종목의 3거래일 후 주가 흐름을 예측하는
시스템 — 성균관대학교 캡스톤 디자인 수업 프로젝트 (팀 4명).

## 개요

**가설:** 정형 데이터(가격·거래량·매크로)로 설명되지 않는 사람들의 기대 심리를 수치로 해석해 모델에 반영하면,
AI 섹터 종목의 n 거래일 뒤 주가 흐름을 예측할 수 있다. 현재 **n = 3 거래일**이다.

예측 대상을 두 모델로 나눈다. 종목 i, 시점 t(종가 기준)의 3거래일 수익률을 `r`, 모델 A의 예측을 `Â`라 하면:

```
ε = r − Â          (모델 A가 설명하지 못한 부분)
최종 예측 = Â + B̂  (B̂ = 모델 B가 예측한 ε)
```

- **모델 A — 정형 데이터:** 공개 사전학습 시계열 모델 **Chronos-2**에 가격·거래량을, 매크로는 외부 변수(공변량)로
  넣는다. 모델 A 담당 팀원이 운영하며 레포에는 코드가 없다. 출력은 `date, ticker, target_date, close, real_ret, pred_ret, pred_q10,
  pred_q90, residual` 형태의 CSV다.
- **모델 B — 텍스트 데이터:** 뉴스와 Reddit에서 종목별 **언급량(강도)과 감성(방향)의 "평소 대비 편차"** 및 둘의
  곱을 만들어 ε를 예측한다. 뉴스와 Reddit은 50:50으로 반영하되, 글 수가 훨씬 많은 Reddit의 글 하나당
  가중치를 뉴스보다 낮춘다(글 수에 반비례) — 설계 합의 사항이며 아직 코드로 구현되지 않았다.

**대상 종목 (10개):** NVDA, AMD, MU, TSM, AVGO, MRVL, MSFT, GOOGL(GOOG), AMZN, META.
SNDK와 전력주(MPWR, VICR, ON 등)는 제외했다.

**학습 구간:** 2026-05-01 ~ 2026-09-30 (귀속 거래일 기준 105거래일). 10종목 × 105거래일 = 1,050행.
"평소 대비" 베이스라인을 위해 그 이전 구간도 함께 수집·전처리한다.

## 데이터

| 종류 | 소스 |
|---|---|
| 뉴스 | Alpaca News(Benzinga), Massive. FNSPID·회사 IR RSS 수집기도 있으나 이번 학습 구간에는 쓰지 않는다 |
| 커뮤니티 | Reddit — Arctic Shift API(키 불필요), 서브레딧은 `configs/reddit_subreddits.yaml` |
| 감성 점수 | FinBERT(`ProsusAI/finbert`), 글 점수 = P(긍정) − P(부정). SocBERT는 아직 적용하지 않았다 |
| 주가·매크로 | 모델 A 담당이 관리. 레포에는 없음 |

소스별 수집 결과와 한계는 `reports/`, 전처리 결정과 근거는 `docs/plans/`에 있다.

**시점 정합(point-in-time):** 글의 발행 시각을 미국 동부시간으로 바꿔 NYSE 거래일에 귀속한다. 해당 거래일의
**장 마감(조기 폐장 반영) 이후, 주말, 휴장일**에 나온 글은 다음 거래일로 넘긴다. 그래서 날짜 t의 피처는 t 종가까지의
정보만 쓴다 (`src/capstone/data/news_preprocessing/trading_calendar.py`).

**데이터 파일은 레포에 없다.** `data/`는 전부 gitignore이고, 특히 Reddit 원문이 들어 있는 파일은 공개 레포나 외부에
올리지 않는다.

## 파이프라인

```
# 수집 (.env에 API 키 필요: Alpaca, Massive. Reddit/Arctic Shift는 키 불필요)
python scripts/fetch_alpaca_news.py --symbols <티커,…> --start <날짜> --end <날짜>
python scripts/fetch_massive_news.py --symbols <티커,…> --start <날짜> --end <날짜>
python scripts/fetch_reddit.py ...

# 전처리 + 감성
python scripts/preprocess_news.py          # 1단계: 정규화 · 거래일 귀속 · 중복 제거 · 종목·일 집계
python scripts/score_news_sentiment.py     # 2단계: 노이즈 표시 · FinBERT 점수 · 종목·일 감성 피처
python scripts/preprocess_reddit.py        # Reddit 1~2단계 (같은 방식)
```

노이즈는 행을 지우지 않고 `drop_reason`/플래그로 표시하고, 감성 집계에서만 제외한다. 결과물은
`data/processed/{news,reddit}/` 아래 종목×일 단위 parquet이다.

## 평가

- **주지표: IC** — 같은 날짜의 종목 간 예측 순위와 실제 순위의 상관(스피어만), 날짜별로 구해 평균한다.
- **검증:** 시간순 교차검증이며 학습과 평가 사이에 3거래일 간격(embargo)을 둔다(3일 수익률 구간이 겹치므로).
  신뢰구간은 날짜 단위 블록 부트스트랩으로 구한다.
- **비교 기준:** "텍스트 없이 모델 A만 쓴 최종 예측". **모델 A 출력을 입력에 넣은 기준 모델은 쓰지 않는다** —
  정답이 `ε = r − Â`라서 `Â`와 ε가 기계적으로 음의 상관을 가져 점수가 부풀려진다.
- **표본의 한계:** 105거래일 × 10종목이고 3일 수익률이 겹쳐 독립 시점은 약 35개다. 평균 IC가 대략 0.14 이상이어야
  2σ 수준으로 유의하다. 결과는 유의성 검정이라기보다 방향과 크기를 보는 탐색적 증거로 읽을 것.

모델 B 구현 방법론(FinBERT/SocBERT 등 인코더, 범용 LLM 등)은 계속 비교할 후보다.

## 현재 상태 (2026-11 초)

- **완료:** 뉴스·Reddit 수집, 뉴스 전처리 + FinBERT 감성(`main`), Reddit 전처리(`main`), 모델 A 출력 수령
- **탐색 결과:** 선형(릿지) 모델로 뉴스·Reddit 텍스트 입력의 효과를 시험했으나 통계적으로 검출되지 않았다. 입력
  6개 모두 가중치의 95% 신뢰구간이 0을 포함했다. 또 모델 A 자체에 종목 간 예측력이 거의 없어(IC ≈ −0.01) ε가 사실상
  실제 수익률과 같아진다. 이 실험 코드는 아직 레포에 없다.
- **TBD:**
  - Reddit 감성 점수 코드 — PR #9가 `main`이 아닌 `feature/reddit-preprocessing`으로 머지돼, `main` 대상 PR이 필요하다
  - 모델 B 학습·평가 코드, 글 단위 가중치 구현
  - `configs/universe.yaml`은 아직 반도체 9종목의 임시 목록이다(실제 사용 목록은 `configs/news_preprocessing.yaml`)
  - Reddit 감성의 SocBERT 적용, 표본 확대(구간 연장)
  - 팀원별 역할 분담, 일정

디렉토리 구조는 [`AGENTS.md`](./AGENTS.md#directory-structure) 참고.

## 이 레포에서 작업하기 전에

AI 코딩 에이전트(Claude Code 등)를 쓴다면 [`AGENTS.md`](./AGENTS.md)를 먼저 읽을 것 — 셋업 커맨드,
danger zone, 브랜치 전략이 정리되어 있다. 이 레포가 왜 이렇게 세팅됐는지는
[`HARNESS.md`](./HARNESS.md)에 있다.
