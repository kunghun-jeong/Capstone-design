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
MLflow 예정. **아직 미확정:** 정확한 패키지 목록/버전, 패키지 매니저(pip/uv/poetry), 디렉토리 구조.
확정되는 대로 이 섹션을 업데이트할 것.

## Setup

<!-- 아직 requirements.txt/pyproject.toml이 없음 — 첫 데이터 파이프라인 PR에서 채울 것 -->
```
# TBD — 패키지 매니저 확정 후 채울 것
python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt   (아직 없음)
```

## Build, run, and verify

<!-- 빌드/실행 대상이 아직 없어 커맨드를 지어내지 않음. 첫 파이프라인/모델 코드가 생기는 PR에서
반드시 채우고, 그 전까지는 "이 저장소는 아직 실행 가능한 산출물이 없다"는 게 사실이다. -->

| Task | Command |
|---|---|
| Run the app | TBD — 아직 실행 가능한 파이프라인 없음 |
| Run tests | TBD — 첫 테스트 추가 시 채울 것 |
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

4명이 같은 주에 다른 파트(데이터 파이프라인/모델 A/모델 B/백테스팅 등)를 동시에 작업할 가능성이 높은
학기 프로젝트다. `feature/<작업 내용>` 브랜치에서 작업하고 PR로 `main`에 병합한다. 리뷰는 최소 1명
권장(과목 특성상 강제하지는 않되, 서로 다른 파트를 건드리는 큰 변경은 리뷰를 거치는 게 안전하다).

## Handoff

- PR 설명에는 **무엇을** 바꿨는지뿐 아니라 **왜** 바꿨는지도 적는다 — 다음 사람(본인이든 팀원이든)이
  커밋 목록만 보고 이유를 추측하지 않아도 되게.
- 진행 중 막히거나 다음 세션에 넘길 내용이 있으면 커밋 메시지나 PR 설명에 구체적으로 남긴다.

## Multi-agent notes

- 4명이 각자 다른 파트를 동시에 작업할 수 있으므로, 되도록 각자 자기 파트 디렉토리 밖은 건드리지 않는
  걸 기본으로 한다. `AGENTS.md`, `requirements.txt` 등 공용 파일을 같은 시점에 고쳐야 하면, 각자
  브랜치에서 고치고 PR로 합친다 — 직접 `main`에서 동시에 고치지 않는다.
- 어떤 AI 코딩 도구를 쓸지는 아직 팀 내 확정 전이다. Claude Code 외 다른 도구(Cursor, Copilot, Gemini
  CLI 등)를 쓰는 팀원이 생기면, 해당 도구의 adapter 파일을 하나 추가하면 된다 (이 파일이 이미 단일
  소스이므로 새 adapter는 짧은 pointer 파일이면 충분하다).
