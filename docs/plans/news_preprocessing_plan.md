# 모델 B용 뉴스 데이터 전처리 계획

- 작성일: 2026-11-01
- 범위: 뉴스만 (Reddit 별도). 학습 구간 2026-04-01 ~ 2026-10-31, 호라이즌 n = 3 거래일.
- 근거 자료: `AGENTS.md`, `reports/{alpaca_news,massive_news,ir_press_releases}_report.md`, `configs/universe.yaml`,
  수집기 3종의 출력 스키마, `git log`/`git branch`, `data/` 디렉토리 상태.
- 이 문서는 계획이다. 코드는 없다.

## ① 현재 상태 요약

1. **이 PC의 `data/raw`, `data/interim`, `data/processed`는 전부 비어 있다**(`.gitkeep`만). 리포트가 말하는 원본 JSONL/CSV는 다른 환경에 있다. 커버리지 평가는 리포트 수치로만 했다.
2. 수집기 4종 중 이번 범위에 쓸 수 있는 것은 **Alpaca(Benzinga)와 Massive** 둘뿐이다. FNSPID는 2023-12 종료, IR RSS는 6종목 중 새 유니버스에 속한 것이 AMD 하나이고 제목만 있다.
3. **Alpaca와 Massive는 둘 다 Benzinga 계열**이다(Massive 출처 중 Benzinga 5,397행). 소스 간 중복이 구조적으로 생긴다.
4. 기존 수집은 반도체 9종목 기준이다. **AMZN·GOOGL·META·MSFT는 어느 소스에도 없다.** 재수집이 전처리의 선행 조건이다.
5. 기존 종목 중 새 유니버스에 남는 NVDA·AMD·TSM·MU는 2026년 기사량이 충분하다(Alpaca 1~10월 각 4,471 / 1,234 / 781 / 1,646건). ADI는 얇다(Alpaca 175, Massive 47).
6. 수집 종료 시점이 Alpaca 2026-10-07, Massive 2026-10-10이라 **10월 말까지 연장 수집**이 필요하다.
7. 두 소스 모두 발행 시각이 UTC 문자열이다(Alpaca `created_at`, Massive `published_utc`). Alpaca는 API 필터가 `updated_at` 기준이라 수집 후 `created_at`으로 재분류해야 한다.
8. Alpaca 기사의 64.9%가 다종목 태그(최대 2,649개). Massive도 전역 고유 ID 35,292 vs 행 44,214로 종목 간 반복이 있다.
9. 텍스트 필드가 소스마다 다르다. Alpaca는 `content`(HTML, 69.5%)·`summary`(47.7%), Massive는 `description`(중앙값 192자, 97.6%)뿐이고 본문이 없다.
10. `AGENTS.md`는 아직 "반도체 섹터, t+5, SOXX" 기준으로 쓰여 있어 현재 설계(AI 섹터, n=3, 잔차 타깃)와 어긋난다. 이번 작업 범위는 아니지만 별도 PR로 갱신해야 한다.

## ② 결정 표

각 행: **결정 / 근거 / 대안 / 사용자 확인**.

| # | 항목 | 결정 | 근거 | 대안 | 확인 |
|---|---|---|---|---|---|
| 1 | 소스 커버리지 | 사용 소스 = **Alpaca + Massive**. IR·FNSPID·EDGAR 제외. 2026-01-01 ~ 2026-11-01(UTC) 구간을 10종목 전체 **재수집**한다. | 4개 빅테크는 미수집. IR은 10종목 중 AMD만 있어 종목 간 비대칭이 생김. EDGAR는 미머지·403. 워밍업(8번)을 위해 1월부터 받는다. Alpaca는 200 req/min이라 10종목 10개월이 수 분, Massive는 5 req/min이라 수십 분 수준(리포트 기준 9종목 전 기간 10분). | IR의 AMD만 "회사 발표" 플래그로 추가 — 1종목뿐이라 보류. | **필요**: 원본 데이터와 API 키가 어디 있는지, 내가 직접 재수집해도 되는지. |
| 2 | 스키마 통합 | 공통 컬럼: `article_id`(source+원본 id), `source`, `published_utc`, `title`, `summary`, `body`, `url`, `publisher`, `tickers`(리스트), `n_tags`, `raw_extra`(json). 결측은 빈 문자열이 아니라 **null**로 통일. | Alpaca: id/headline/summary/content/url/source/symbols/created_at. Massive: id/title/description/article_url/publisher.name/tickers/published_utc. `body`는 Alpaca만 있음. Massive `insights`는 `raw_extra`에 보존. | 소스별 테이블을 그대로 두고 조인 — 중복 제거가 어려워져 기각. | 불필요 |
| 3 | 시점 정합 | 모든 시각 → UTC → `America/New_York`. **컷오프 = 16:00 ET(정규장 마감)**. 거래일 d의 16:00 이전 발행 → 귀속일 d. 16:00 이후·주말·휴장일 → **다음 거래일**. 거래일 캘린더는 NYSE(`pandas_market_calendars`). Alpaca는 `created_at` 사용. | 모델 A가 종가 t 기준으로 예측하고 라벨이 close(t)→close(t+3)이면, t의 마감 전 정보까지 t에 귀속하는 것이 A와 같은 정보 집합이다. 16:00 직전 기사는 실질 거래 불가라는 문제가 있으나 비중이 작다. | 컷오프를 09:30 ET(개장)로 당겨 "전일 마감 후~당일 개장 전" 기사만 t에 귀속 — 더 보수적. config로 두고 ablation 가능하게 한다. | **필요**: 모델 A의 예측 기준 시점(종가 t 기준인지)과 맞춰야 함. |
| 3-라벨 | 라벨 정합 | 라벨 r(t→t+3) = close(t+3)/close(t) − 1. 모델 B 타깃 = r − Â(t→t+3). 라벨용 가격은 **2026-11-04까지** 필요(10-31은 토요일이라 마지막 거래일 10-30 + 3거래일). 라벨 조인은 전처리가 아니라 `features/`에서 한다. | 전처리 산출물은 (ticker, 귀속일) 키로 끝나고, 가격·A 예측과의 조인은 그 뒤 단계. | — | 참고: OHLCV 수집기가 레포에 없다. 누가 받을지 결정 필요. |
| 4 | 종목 귀속 | `tickers`를 **유니버스 교집합으로 explode**해 (article_id, ticker) 행을 만든다. `n_tags` 보존. `is_primary` = 제목에 티커 또는 회사명(별칭 사전) 포함. `is_broad` = `n_tags > 10`. **전처리에서 행을 버리지 않는다.** | 언급량 신호는 "몇 번 언급됐나"라서 삭제하면 왜곡된다. 가중치·필터는 피처 단계에서 `n_tags`/`is_primary`로 실험한다. 빅테크는 시황 기사에 늘 태그되므로 `is_primary` 분리가 필요하다. | 주종목 1개만 지정 — 판단 규칙이 없어 기각. 50개 초과 태그 기사 삭제 — 1.5%라 플래그로 충분. | 불필요 |
| 5 | 중복 제거 | 1단계 소스 내: 원본 id. 2단계 소스 간: **정규화 URL 일치**, 없으면 **정규화 제목 + 같은 귀속일**. 대표 레코드는 텍스트가 더 긴 쪽(Alpaca 우선). `sources`(리스트)와 `n_sources` 컬럼으로 확산 정도를 남긴다. 유사 제목(신디케이션) 근사 중복은 이번엔 **하지 않고** 플래그만 가능하면 추가. | Alpaca·Massive 모두 Benzinga 계열이라 정확 중복이 많다. 언급량은 고유 기사 수(`n_articles`)와 소스 포함 수(`n_rows`)를 둘 다 집계해 피처 단계에서 선택한다. | simhash/MinHash 근사 중복 — Motley Fool 46.6% 쏠림 때문에 필요할 수 있으나 2차 과제. | 불필요 |
| 6 | 텍스트 정제 | 공통 텍스트 단위 = **`title` + `summary`(Alpaca summary / Massive description)**. Alpaca `content`는 HTML 제거(`beautifulsoup4`)해 `body`로 보존하되 기본 입력에서 제외. 길이 상한 없음(저장), 인코더 입력 시 512 토큰 절단. 제어문자·공백 정규화. 언어 필터 없음. | Massive에 본문이 없어 소스 간 비교 가능한 단위는 제목+요약뿐이다. Alpaca summary는 2021~ 45~72%만 있어 결측이면 제목만. Benzinga는 영문이라 언어 필터 불필요(샘플로 확인). | 본문 기반 감성 — Alpaca 한정 ablation으로 가능. | **필요**: 모델 B 인코더 담당과 "제목+요약"을 공통 입력으로 합의. |
| 7 | 관련성 필터 | **하드 필터 없음.** `is_primary`, `is_broad`, `n_tags`, `publisher`를 플래그로 제공. | 7번을 전처리에서 결정하면 언급량 편차 정의가 바뀐다. 피처 단계에서 (전체 / primary만) 두 버전을 비교한다. | 키워드 블랙리스트(시황 요약, ETF 목록) — 데이터를 본 뒤 2차 적용. | 불필요 |
| 8 | 집계·베이스라인 | 두 산출물: (a) 기사 단위 통합 테이블, (b) **(ticker, 귀속일) 단위** 테이블: `n_articles`, `n_rows`, `n_primary`, `n_sources`, `n_publishers`, 텍스트 id 목록. "평소 대비 편차" = 과거 **60거래일 롤링 중앙값·MAD** 기반 robust z, 창은 **t 이전까지만**(shift 1). 워밍업: 2026-01-01부터 수집해 4월 1일 시점에 약 60거래일 확보. 기사 없는 날은 `n=0`으로 채움(결측 아님). | 언급량은 꼬리가 길어 평균·표준편차보다 중앙값·MAD가 안정적. 60일은 분기 1개 분량. 감성 집계는 인코더 점수가 나온 뒤 같은 모듈에서 수행(인터페이스만 열어둠). | 20·120일 창 — config로 두고 비교. 로그 변환 후 z — 대안으로 기록. | 불필요 |
| 9 | 분할·embargo | 시간순 3분할: **학습 4/1~8/31, 검증 9/1~9/30, 테스트 10/1~10/31**, 경계마다 **3거래일 embargo**(라벨 겹침 제거). | 표본 ≈ 148거래일 × 10종목. 3일 라벨은 인접일이 겹치므로 경계 purge 필수. | 모델 B 1차가 z-score의 고정 결합(학습 없음)이면 **전 구간이 아웃오브샘플**이라 7개월 전체로 IC를 볼 수 있다. 권장. | 불필요(제안으로 기록) |
| 10 | 출력·경로 | Parquet. `data/interim/news/{alpaca,massive}_normalized.parquet` → `data/interim/news/articles.parquet`(중복 제거, explode 전) → `data/processed/news/article_ticker.parquet`, `data/processed/news/ticker_day.parquet`. 설정은 `configs/news_preprocessing.yaml`(유니버스 경로, 기간, 컷오프, 창 길이, 별칭 사전). 실행 로그에 config 해시·입력 파일 해시·git sha 기록. | 디렉토리 구조는 AGENTS.md를 따른다. 재현성은 config+해시로 충분(난수 없음). | CSV — 리스트 컬럼 때문에 기각. | 불필요 |
| 11 | 품질 검증 | 리포트 `reports/news_preprocessing_report.md`: 종목·월별 `n_articles`, 결측률, 소스 간 중복률, `is_primary` 비율, 귀속일 분포(주말·장후 비중). **누수 테스트**(pytest): 귀속일 ≥ 발행일(ET), 16:00 이후 기사가 같은 날에 귀속되지 않음, 롤링 베이스라인이 t 당일을 포함하지 않음, 휴장일에 귀속된 행 0. 스키마 테스트, 중복 제거 테스트, 소규모 고정 샘플(fixture). | AGENTS.md "시점 정합 필수" 규칙. | — | 불필요 |
| 12 | 작업 분해 | 모듈 `src/capstone/data/news_preprocessing/`(`normalize.py`, `calendar.py`, `dedup.py`, `aggregate.py`, `config.py`), CLI `scripts/preprocess_news.py`, 테스트 `tests/data/news_preprocessing/`. ③ 참고. | AGENTS.md 디렉토리 규칙. 피처(`features/`)와의 경계: 베이스라인 편차까지는 데이터 단계, 감성·상호작용·라벨 조인은 피처 단계. | 전부 `features/`에 — 수집기와 가까운 정규화·중복 제거는 data 쪽이 맞다. | 불필요 |

신규 의존성: `pyarrow`, `pandas_market_calendars`, `beautifulsoup4`. `requirements.txt`에 추가(별도 PR 아님, 1단계 PR에 포함).

## ③ 단계별 작업 계획

| 단계 | 브랜치 | 입력 | 출력 | 완료 기준 |
|---|---|---|---|---|
| 0. 유니버스·설정 | `feature/universe-ai-sector` | 팀 확정 티커 | `configs/universe.yaml` 갱신(`status: confirmed`, 10종목, 미확정 종목은 주석), `configs/news_preprocessing.yaml` 초안 | 팀원 1명 리뷰. 미확정 2종목은 TBD 주석으로 두고 머지 가능. |
| 1. 재수집 | 브랜치 불필요(기존 CLI) | API 키, 유니버스 | `data/raw/alpaca_news/<T>/2026.jsonl`, `data/raw/massive_news_ticker/<T>/2026.jsonl` (2026-01-01~2026-11-01) | 10종목 × 2소스 모두 `_state.json`/`_runs.jsonl` 성공. 종목·월별 건수표를 PR 설명에 첨부. **Alpaca는 `updated_at` 필터라 11-01까지 받아 `created_at`로 재분류.** |
| 2. 정규화·스키마 | `feature/news-normalize` | 1의 JSONL | `*_normalized.parquet`, `normalize.py`, 스키마 테스트 | 두 소스가 같은 컬럼·타입. 결측 null. fixture 테스트 통과. |
| 3. 거래일 귀속 | `feature/news-calendar-align` | 2의 산출물 | `calendar.py`, `attributed_day` 컬럼, 누수 테스트 | 11번의 누수 테스트 4종 통과. 주말·휴장일·장후 샘플 케이스 테스트. |
| 4. 중복 제거·종목 explode | `feature/news-dedup` | 3의 산출물 | `articles.parquet`, `article_ticker.parquet`, `dedup.py` | 소스 간 중복률 수치 보고. explode 후 행 수 = Σ 유니버스 태그 수. `is_primary`/`is_broad` 비율 보고. |
| 5. 종목·일 집계·베이스라인 | `feature/news-ticker-day` | 4의 산출물 | `ticker_day.parquet`, `aggregate.py`, 감성 집계 인터페이스 | 모든 (ticker, 거래일) 조합 존재(0 포함). 롤링 창 shift 테스트 통과. 4월 1일 행의 베이스라인 창 ≥ 40거래일. |
| 6. 품질 리포트 | 5와 같은 PR 또는 `feature/news-preprocessing-report` | 전 단계 | `reports/news_preprocessing_report.md` | ② 11번 지표 전부 포함. 모델 B 담당이 읽고 입력 정의에 동의. |

의존성: 0 → 1 → 2 → 3 → 4 → 5 → 6. 1은 API 키 소유자에 달려 있고, 2~3은 1의 소량 샘플만 있어도 fixture로 먼저 개발할 수 있다. 각 PR은 작게, AGENTS.md 규칙대로 `python -m pytest` 통과 후 올린다.

## ④ 위험요소

1. **원본 데이터 부재**: 이 PC에 데이터가 없다. 재수집이 막히면 2단계 이후 전부 지연된다. 가장 큰 리스크.
2. **빅테크 4종목 기사량 폭증**: NVDA가 Alpaca 2026년 4,471건인데 AMZN/MSFT는 그 이상일 수 있다. 시황·ETF 기사 태그 비중이 커서 `is_primary` 없이는 언급량이 시장 전체 뉴스량을 재는 지표가 된다.
3. **소스 중복이 언급량을 부풀림**: Benzinga 계열 두 소스를 합치면 같은 기사가 2번 잡힌다. 5번 결정으로 `n_articles`(고유)와 `n_rows`를 분리해 둔다.
4. **ADI(포함 시) 희소성**: 2026년 월 15건 수준. 일 단위 편차가 0/1 토글이 된다. 유니버스 결정에 반영할 것.
5. **Massive 출처 편향**: Motley Fool 46.6%. 감성 점수가 특정 매체 문체를 학습할 수 있다. `publisher`를 보존해 매체별 IC를 따로 본다.
6. **`updated_at` 드리프트**: 10월 말 기사가 11월에 갱신되면 10-31 컷으로는 누락된다. 11-01 이후 한 번 더 받아 `created_at`로 재분류해야 하며, 재수집 시점이 늦을수록 안전하다.
7. **표본 크기**: 7개월·10종목. 월 단위 테스트 구간(약 22거래일)의 IC 유의성은 약하다. 9번 대안(무학습 신호로 전 구간 평가)을 병행해야 한다.
8. **컷오프 규칙과 모델 A의 불일치**: A가 종가 외 다른 시점을 쓰면 B 잔차 정의가 틀어진다.
9. **AGENTS.md 구식 서술**: t+5·SOXX·섹터 ETF 설계가 그대로 남아 있어 AI 도구가 옛 설계를 따를 수 있다.

## ⑤ 사용자에게 확인할 질문 (우선순위순)

1. **원본 데이터와 API 키 위치**: 리포트의 Alpaca/Massive JSONL은 어느 환경에 있는가? 내가 Alpaca·Massive 키로 10종목 2026-01-01~11-01을 직접 재수집해도 되는가?
2. **유니버스 나머지 2종목과 ADI 포함 여부**: 현재 확정분은 AMZN, GOOGL, META, MSFT, NVDA, AMD, TSM, MU(8개). GOOGL/GOOG 중 어느 것을 쓰는가?
3. **컷오프 16:00 ET와 모델 A 기준 시점**: A는 종가 t 기준으로 t→t+3을 예측하는가? 그렇다면 16:00 ET 컷오프로 간다.
4. **공통 텍스트 단위 = 제목+요약**: Massive에 본문이 없어 이렇게 맞추려 한다. 모델 B 인코더 담당(다혜/서) 동의가 필요하다.
5. **워밍업 수집 2026-01-01부터**: 베이스라인 60거래일용. 수집 비용은 작다. 이의 없으면 진행.
6. **라벨용 OHLCV(2026-11-04까지)**: 누가, 어떤 소스(yfinance 등)로 받는가? 전처리 범위 밖이지만 라벨 조인의 선행 조건이다.
7. **IR RSS 제외**: 10종목 중 AMD만 해당되고 제목만 있어 이번엔 제외하려 한다. 동의하는가?

## 확인되지 않은 가정

- 리포트의 원본 데이터가 아직 어딘가에 보존돼 있다(이 PC엔 없다).
- Alpaca 무료 플랜과 Massive Basic 플랜이 지금도 2026-01~10 구간을 같은 조건으로 제공한다(Massive는 문서상 2년 이내 접근이라 문제없을 것으로 **추정**).
- AMZN/GOOGL/META/MSFT의 Alpaca·Massive 기사량은 NVDA 이상일 것으로 **추정**. 실측 전.
- Alpaca·Massive 기사가 영문뿐이라는 것은 리포트에 없고 Benzinga 성격에서 **추정**. 샘플 확인 필요.
- 모델 A가 종가 t 기준 예측이라는 것은 회의 메모에 없다. 질문 3.
- Massive `insights`의 감성 라벨은 37.9%만 있어 입력이 아니라 비교용으로만 쓸 수 있다고 **판단**. 품질은 미확인.
- (확인됨, 2026-11-01) `pandas_market_calendars` NYSE 2026: 4/1~10/31 거래일 148, 1분기 61, 10-31은 토요일(마지막 거래일 10-30), 11-02~11-04 정상 거래. 휴장일 테이블 자체의 정확성은 라이브러리를 믿는 가정.
