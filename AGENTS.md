# AGENTS.md

이 문서는 이 레포에서 AI 코딩 에이전트(Claude Code, Cursor, Codex, Copilot 등)가 작업할 때 따라야 할
단일 기준 문서다. 다른 도구별 파일(`CLAUDE.md` 등)은 이 문서를 가리키기만 하고 내용을 복제하지 않는다.

## Project

뉴스·Reddit 여론의 "기대 심리"를 수치화해 미국 AI 섹터 10개 종목의 **3거래일 후** 주가 흐름을 예측하는
시스템. 성균관대학교 캡스톤 디자인 수업 프로젝트(팀 4명).

- **종목(10):** NVDA, AMD, MU, TSM, AVGO, MRVL, MSFT, GOOGL(GOOG), AMZN, META. SNDK·전력주는 제외.
- **학습 구간:** 2026-05-01 ~ 2026-09-30 (귀속 거래일 기준 105거래일).

```
ε = r − Â            r: 3거래일 수익률, Â: 모델 A 예측
최종 예측 = Â + B̂    B̂: 모델 B가 예측한 ε
```

- **모델 A (정형 데이터)**: 공개 사전학습 시계열 모델 **Chronos-2**(가격·거래량 + 매크로를 외부 변수로 입력).
  코드는 `notebooks/modelA_predict.ipynb`(Colab 노트북)이고 데이터는 레포에 없다. 출력은 CSV(`date, ticker, target_date, close, real_ret, pred_ret,
  pred_q10, pred_q90, residual`)로 받는다. `date`가 예측을 만든 시점 t, `target_date`가 t+3 거래일이다.
- **모델 B (텍스트 데이터)**: 뉴스·Reddit의 종목별 **언급량과 감성의 "평소 대비 편차"** 및 둘의 곱을 입력으로
  ε를 예측한다. 뉴스:Reddit은 50:50이되 글 하나당 가중치는 글 수에 반비례(설계 합의, 아직 미구현).
- 평가는 날짜별 종목 간 IC를 주지표로, 시간순 교차검증(경계 3거래일 embargo)과 부트스트랩 신뢰구간으로 한다.

## Stack

Python 생태계. 지금 쓰는 것: pandas·pyarrow(parquet), `pandas_market_calendars`(NYSE 거래일), PyTorch(CPU판으로
충분) + transformers(FinBERT 감성), pytest. 모델 B는 선형 모델(릿지)로 탐색했고 GBDT·MLflow는 아직 쓰지 않는다.
**아직 미확정:** 패키지 버전 고정, 패키지 매니저(지금은 pip + `requirements.txt`).

## Directory structure

파이프라인 단계 순서를 그대로 반영한 구조다 (데이터 → 피처 → 모델 A/B → 백테스팅).

```
configs/                     # 티커·기간·전처리 설정 (news_preprocessing.yaml, reddit_*.yaml 등)
data/{raw,interim,processed} # 데이터 산출물 — 전부 gitignore, 코드만 레포에 남음
docs/plans/                  # 전처리 설계 결정과 근거 (뉴스, Reddit)
reports/                     # 소스별 수집·전처리 리포트
src/capstone/
  data/                      # 수집기(alpaca/massive/fnspid/ir/reddit)와 전처리
    news_preprocessing/      #   뉴스: 정규화, 거래일 귀속, 중복 제거, 노이즈 표시, FinBERT 감성, 종목·일 집계
    reddit_preprocessing/    #   Reddit: 같은 방식의 전처리
  features/                  # (Reddit 감성 코드가 들어올 자리 — PR #9는 아직 main 미반영)
  models/model_a/            # 비어 있음 (모델 A 코드는 notebooks/modelA_predict.ipynb)
  models/model_b/            # 비어 있음 (모델 B 학습 코드 예정)
  backtest/                  # 비어 있음
  eval/                      # 비어 있음
notebooks/                    # 탐색/실험용 노트북
scripts/                      # CLI 진입점 (fetch_*, preprocess_*, score_news_sentiment)
tests/                        # pytest, src/ 구조 미러링
```

코드가 있는 곳은 `data/` 아래뿐이고 나머지는 빈 스캐폴드(`__init__.py`/`.gitkeep`)다. 새 코드를 넣을 때 위 목록과
실제가 어긋나면 이 섹션도 같이 고칠 것.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

패키지 매니저는 아직 미확정이라 임시로 pip + `requirements.txt`를 쓴다. FinBERT용 `torch`는 CPU판이면 충분하다
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

API 키는 레포 루트 `.env`에 둔다 (커밋 금지): `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`(Alpaca),
`MASSIVE_API_KEY`. Reddit(Arctic Shift)은 키가 필요 없다. 지금 `.env.example`에는 Alpaca 항목만 있다.

## Build, run, and verify

| Task | Command |
|---|---|
| 수집 | `python scripts/fetch_alpaca_news.py --help`, `fetch_massive_news.py`, `fetch_reddit.py` (모두 `--symbols`로 티커 지정 가능) |
| 뉴스 전처리 | `python scripts/preprocess_news.py` → `python scripts/score_news_sentiment.py` |
| Reddit 전처리 | `python scripts/preprocess_reddit.py` |
| Run tests | `python -m pytest` |
| Lint / format | TBD |
| Type-check (if applicable) | TBD (사용한다면) |

ROS가 설치된 WSL에서는 시스템 파이썬의 pytest 플러그인이 충돌할 수 있다. 그럴 땐
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`로 실행한다.

**`python -m pytest`가 통과하기 전까지 작업은 완료된 게 아니다.** PR을 올리기 전이 아니라 완료를 보고하기 전에
실행할 것.

`.github/workflows/verify.yml`이 PR과 `main` push마다 같은 `python -m pytest`를 돌린다. lint 스텝만 아직
placeholder다. 위 표와 워크플로우가 서로 다른 커맨드를 말하게 두지 말 것.

## Code conventions

- **시점 정합(point-in-time alignment) 필수**: 텍스트의 발행 시각을 NYSE 거래일에 귀속할 때 해당 거래일의
  **장 마감 이후·주말·휴장일 글은 다음 거래일**로 넘긴다(`trading_calendar.attribute_trading_day`를 재사용하고
  새로 만들지 말 것). 날짜 t의 피처는 t 종가까지의 정보만 쓴다. 새 피처를 추가할 때마다 이 정합이 깨지지
  않았는지 확인할 것.
- **"평소 대비 편차"의 베이스라인은 당일을 포함하지 않는다**(과거 거래일의 롤링 통계, shift 1). 글이 없는 날은
  0이 아니라 결측으로 두고 별도 표시한다.
- **노이즈는 행을 지우지 않고 표시한다**(`drop_reason`, 플래그). 걸러내기는 감성 집계 단계에서 명시적으로 하고
  제거 건수를 기록한다. 언급량 신호가 필터에 의해 조용히 바뀌지 않게 하려는 것이다.
- **Reddit 점수(score)·댓글 수는 약 36시간 뒤 값으로 갱신되므로** t 시점 피처에 그대로 쓰면 미래 정보가 섞인다.
  지금은 건수만 센다.
- **모델 B 평가 기준:** 정답은 `ε = r − Â`다. **모델 A 출력(`pred_ret` 등)을 입력에 넣은 기준 모델과 비교하지
  말 것** — `Â`가 ε에 빼기로 들어가 있어 기계적으로 높은 IC가 나온다. 비교 기준은 "텍스트 없이 A만 쓴 최종
  예측"이다. 표본이 작아(독립 시점 약 35개) IC 약 0.14 미만은 구분할 수 없으니 신뢰구간과 함께 보고할 것.
- **모델 B 방법론은 아직 탐색 단계다**: 지금 감성은 FinBERT(뉴스·Reddit 공통)이고, SocBERT, 다른 사전학습모델,
  범용 LLM 프롬프트 등이 후보다. FinBERT는 WSB 은어와 풍자를 잘 못 읽는다. 실험 코드/노트북에는 사용한
  모델명·설정을 명시할 것.
- **학습/검증/테스트 구간은 항상 시간순 분리**하고, 3일 수익률이 겹치므로 경계에 3거래일 간격(embargo)을
  둔다. 미래 데이터가 과거 예측에 섞이면 안 됨.
- 린트/포맷 설정이 생기면 그 설정이 잡아내는 규칙은 여기 다시 적지 않는다.

## Permissions / danger zone

다음 행동은 자율 세션 중이라도 사람의 명시적 승인이 필요하다:

- Force-push, 또는 공유 브랜치의 히스토리 재작성
- 다른 사람이 쓰고 있는 브랜치 삭제
- `.env`, API 키(Alpaca, Massive 등 credentials), 그 외 비밀정보를 읽거나 수정·커밋
- CI 설정이나 레포 권한 수정
- `main`(보호 브랜치)에 직접 merge/push
- 이 파일이나 다른 adapter 파일, hook 스크립트에 시크릿/API 키를 커밋
- 원본 뉴스/Reddit 텍스트 대용량 데이터셋을 레포에 직접 커밋 (용량·라이선스 문제 — `.gitignore` 참고)

## Branches

작업은 파이프라인 순서를 따르는 단계형 프로젝트다 — 병렬성은 항상 있는 게 아니라 단계별로 다르다.

1. **데이터 수집/전처리** — 모델이 시작할 수 있는 선행 조건이라 병목 구간. 4명 전원이 같이 진행.
2. **모델 A / 모델 B** — 설계상 독립 학습 후 합산하는 구조라 이 단계에서만 2명씩 나눠 병렬 진행.
3. **백테스팅/평가** — 모델 A·B 결과가 모두 있어야 시작 가능한 후행 단계.

`feature/<작업 내용>` 브랜치에서 작업하고 PR로 `main`에 병합한다. 리뷰는 최소 1명 권장(과목 특성상
강제하지는 않되, 다음 단계의 전제가 되는 변경 — 특히 1단계 데이터 파이프라인 — 은 리뷰를 거치는 게
안전하다).

**다른 PR 위에 쌓은 PR(stacked PR)은 머지 순서를 지킬 것.** 아래 PR이 `main`에 먼저 들어간 뒤, 위 PR의 대상
브랜치를 `main`으로 바꾸고 머지한다. 순서가 바뀌면 위 PR이 "머지됨"으로 표시되는데도 코드는 `main`에 없을 수
있다(Reddit 감성 PR #9가 이렇게 `main`에서 빠졌다). 머지 뒤에는 `git log origin/main`으로 실제로 들어갔는지
확인할 것.

## Handoff

- PR 설명에는 **무엇을** 바꿨는지뿐 아니라 **왜** 바꿨는지도 적는다 — 다음 사람(본인이든 팀원이든)이
  커밋 목록만 보고 이유를 추측하지 않아도 되게.
- 진행 중 막히거나 다음 세션에 넘길 내용이 있으면 커밋 메시지나 PR 설명에 구체적으로 남긴다.

## Multi-agent notes

- 1단계(데이터)와 3단계(백테스팅)는 전원이 같은 `src/capstone/data/`(또는 `backtest/`)를 동시에
  건드릴 수 있다는 뜻이므로 커밋 단위를 작게 유지하고 자주 병합할 것. 2단계(모델 A/B)에서는 각자
  `models/model_a/` 또는 `models/model_b/` 밖을 건드리지 않는 걸 기본으로 한다.
- `AGENTS.md`, `requirements.txt` 등 공용 파일을 같은 시점에 고쳐야 하면, 각자 브랜치에서 고치고
  PR로 합친다 — 직접 `main`에서 동시에 고치지 않는다.
- 어떤 AI 코딩 도구를 쓸지는 아직 팀 내 확정 전이다. Claude Code 외 다른 도구(Cursor, Copilot, Gemini
  CLI 등)를 쓰는 팀원이 생기면, 해당 도구의 adapter 파일을 하나 추가하면 된다 (이 파일이 이미 단일
  소스이므로 새 adapter는 짧은 pointer 파일이면 충분하다).
