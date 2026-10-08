# AGENTS.md

이 문서는 이 레포에서 AI 코딩 에이전트(Claude Code, Cursor, Codex, Copilot 등)가 작업할 때 따라야 할
단일 기준 문서다. 다른 도구별 파일(`CLAUDE.md` 등)은 이 문서를 가리키기만 하고 내용을 복제하지 않는다.

## Project

매크로-텍스트 분해 구조를 활용한 미국 반도체 섹터 주가 예측 시스템. 성균관대학교 캡스톤 디자인 수업
프로젝트(팀 4명)로, SOXX 상위 10개 반도체 종목을 대상으로 종목 i의 t+5 거래일 수익률을 섹터 수익률과
종목별 초과수익으로 분해해 예측한다.

```
r_i,t+h = r_S,t+h + ε_i,t+h
```

- **모델 A (섹터 타이밍)**: 매크로 변수(금리, VIX, 유가, 달러인덱스 등, FRED 기준)를 입력으로 섹터
  ETF(SOXX/SMH) 수익률 `r_S,t+h`를 예측하는 시계열 모델.
- **모델 B (종목 선별)**: 종목별 뉴스·Reddit 텍스트 + 기술지표를 입력으로 섹터 대비 초과수익
  `ε_i,t+h`를 예측하는 횡단면 랭킹 모델.
- 최종 예측 = 모델 A 출력 + 모델 B 출력. 평가는 ablation 사다리(무조건 상승 → 가격/기술지표 GBDT →
  +매크로 → +텍스트)와 walk-forward 백테스팅으로 진행.

## Stack

Python 생태계. 확정된 것: pandas 계열 데이터 처리, GBDT(LightGBM 등), PyTorch 계열 딥러닝, 실험 추적은
MLflow 예정. **아직 미확정:** 정확한 패키지 목록/버전, 패키지 매니저(pip/uv/poetry).
확정되는 대로 이 섹션을 업데이트할 것.

## Directory structure

파이프라인 단계 순서를 그대로 반영한 구조다 (데이터 → 피처 → 모델 A/B → 백테스팅).

```
configs/                     # 하이퍼파라미터, 학습 기간, 티커 목록 등 설정 (yaml 예정)
data/{raw,interim,processed} # 데이터 산출물 — 전부 gitignore, 코드만 레포에 남음
src/capstone/
  data/                      # 수집/전처리 파이프라인 (뉴스, reddit, ohlcv, macro)
  features/                  # 피처 엔지니어링 (기술지표, 시점정합, 텍스트 인코딩)
  models/model_a/            # 섹터 타이밍 (매크로 시계열)
  models/model_b/            # 종목 선별 (텍스트 랭킹)
  backtest/                  # walk-forward 백테스팅, ablation 사다리
  eval/                      # 평가지표 (IC, 방향적중률, Sharpe)
notebooks/                    # 탐색/실험용 노트북
scripts/                      # CLI 진입점 (데이터 다운로드, 학습 실행 등)
tests/                        # pytest, src/ 구조 미러링
```

지금은 각 디렉토리에 `__init__.py`/`.gitkeep`만 있는 빈 스캐폴드다. 실제 코드가 들어가면 이 목록은
자연히 최신 상태를 반영하게 되므로, 코드 구조가 크게 바뀌지 않는 한 이 섹션을 따로 갱신할 필요는
없다.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

패키지 매니저는 아직 미확정이라 임시로 pip + `requirements.txt`(최소 목록)를 쓴다. 확정되면 갱신할 것.
API 키는 레포 루트 `.env`에 둔다 (`.env.example` 참고, 커밋 금지).

## Build, run, and verify

| Task | Command |
|---|---|
| Run the app | TBD — 모델/백테스트 파이프라인 없음. 뉴스 수집 CLI: `python scripts/fetch_alpaca_news.py --help` |
| Run tests | `python -m pytest` |
| Lint / format | TBD |
| Type-check (if applicable) | TBD (사용한다면) |

테스트 커맨드가 채워지면, **그 커맨드가 통과하기 전까지 작업은 완료된 게 아니다.** PR을 올리기 전이
아니라 완료를 보고하기 전에 실행할 것.

`.github/workflows/verify.yml`에 CI 스텁이 이미 있다 — 지금은 빈 placeholder라 실제로 아무것도
검증하지 않는다. 위 표의 Lint/Test 커맨드를 채울 때 그 워크플로우의 해당 스텝도 같이 채울 것(둘이
다른 커맨드를 말하게 두지 말 것).

## Code conventions

- **시점 정합(point-in-time alignment) 필수**: 뉴스·Reddit 텍스트의 발행 시각과 실제 거래 가능 시점을
  맞추지 않고 피처를 만들면 미래 정보 누수(look-ahead bias)가 생긴다. 새 피처를 추가할 때마다 이 정합이
  깨지지 않았는지 확인할 것.
- **모델 B 방법론은 아직 탐색 단계다**: FinBERT/SocBERT류 사전학습 인코더, 그 외 다른 사전학습모델,
  범용 LLM 프롬프트 엔지니어링 등 여러 방법론 후보를 병행 실험 중이며 표준으로 못박힌 게 없다. 실험
  코드/노트북에는 사용한 모델명·설정을 명시할 것.
- **학습/검증/테스트 구간은 항상 시간순 분리(walk-forward)**. 미래 데이터가 과거 예측에 섞이면 안 됨.
- 린트/포맷 설정이 생기면 그 설정이 잡아내는 규칙은 여기 다시 적지 않는다.

## Permissions / danger zone

다음 행동은 자율 세션 중이라도 사람의 명시적 승인이 필요하다:

- Force-push, 또는 공유 브랜치의 히스토리 재작성
- 다른 사람이 쓰고 있는 브랜치 삭제
- `.env`, API 키(Alpha Vantage, Alpaca, FRED, Reddit 등 credentials), 그 외
  비밀정보를 읽거나 수정·커밋
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
