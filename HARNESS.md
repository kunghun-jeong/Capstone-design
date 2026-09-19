# HARNESS.md — 이 레포가 이렇게 세팅된 이유

## 왜 이렇게 돼 있나

- **`AGENTS.md`**가 이 레포에서 에이전트가 일하는 방식의 단일 기준 문서다. `CLAUDE.md`는 그걸 가리키기만
  하는 짧은 포인터라, 규칙이 두 곳에 따로 존재하면서 서서히 어긋나는 일이 없다.
- **검증:** 아직 빌드/테스트 대상이 없어 `AGENTS.md`의 커맨드 표가 전부 TBD다. `.github/workflows/verify.yml`도
  같은 이유로 아직 빈 placeholder다 — 첫 데이터 파이프라인/모델 코드가 들어가는 PR에서 둘 다 실제
  커맨드로 채워야 한다.
- **Danger zone:** force-push, 보호 브랜치(`main`) 직접 push, 시크릿/API 키 관련 작업, 대용량 원본
  텍스트 데이터셋 직접 커밋 — 이런 것들은 되돌리기 어렵거나 이 레포 밖(계정, 외부 데이터 라이선스)에
  영향을 주기 때문에 자율 세션에서도 사람 확인을 거친다.
- **Handoff:** 지금은 PR 설명(무엇을/왜)과 커밋 메시지만으로 충분하다고 보고 별도 로그 파일은 두지
  않았다. 4명이 같은 주에 다른 파트를 동시에 작업할 수는 있지만, 학기 프로젝트 초기 규모에서 파일 단위
  decision/defect 로그까지는 과한 ceremony라고 판단해 걷어냈다 — 나중에 팀 규모나 충돌 빈도가 달라지면
  다시 추가할 수 있다.
- **Multi-agent:** 각자 다른 파트 디렉토리 밖은 건드리지 않는 걸 기본으로 하고, 공용 파일은 브랜치+PR로
  합치는 정도의 가벼운 규칙만 둔다.

## 세팅하면서 내린 판단

- **파일 단위 decisions/status-defects, doc-map, `scripts/status.py`, pre-commit 훅은 만들었다가
  걷어냈다.** 4인 학기 프로젝트 규모에는 과한 ceremony라고 판단함 (`references/principles.md`의 "Scale
  the ceremony" 기준으로 지금은 "small team / class project" 등급). 나중에 병렬 작업 충돌이 실제로
  잦아지면 `repo-harness-init` 스킬로 다시 추가하면 된다.
- **기본 브랜치를 `main`으로 맞췄다.** 이 레포는 원래 계정 git 설정(`init.defaultBranch` 미설정 →
  git 기본값인 `master`) 때문에 `master`로 생성됐었다. 로컬 브랜치를 `main`으로 rename하고, 같은 문제가
  다음 레포에서도 반복되지 않도록 `git config --global init.defaultBranch main`으로 계정 설정 자체를
  고쳤다.
- **CI(`verify.yml`)는 빈 스텁으로 먼저 넣어뒀다.** 실제 lint/test 커맨드가 없어 지금은 checkout만 하고
  아무것도 검증하지 않지만, 팀원이 파이프라인 코드를 추가하기 시작할 때 "커맨드만 채우면 되는" 자리를
  미리 만들어 둔 것.

## Self-check

가끔, 특히 새 팀원이 합류하기 전에 한 번씩 훑어볼 것:

- [ ] AGENTS.md가 여전히 실제와 맞는가, 프로젝트가 거기서 적힌 것과 달라지지 않았는가?
- [ ] AGENTS.md의 build/test/verify 커맨드와 `verify.yml`이 실제로 동작하는가? (지금은 둘 다 TBD/스텁 —
      채워졌는지부터 확인)
- [ ] Danger zone 목록이 여전히 적당한 길이인가 — 너무 길어 아무도 안 읽는 상태는 아닌가, 최근에
      위험해진 게 빠져 있진 않은가?
- [ ] 새로 합류한 사람(팀원이든 에이전트든)이 진행 중인 작업의 맥락을 PR/커밋 히스토리만으로 파악할 수
      있는가? 안 된다면 이 시점에 handoff 방식(예: 파일 단위 decision 로그)을 다시 고려할 것.
- [ ] 병렬 작업 충돌(같은 파일을 여러 명이 동시에 고치다 머지 충돌)이 잦아지기 시작했는가? 그렇다면
      `docs/decisions/`, `docs/status-defects/` 같은 파일 단위 구조를 다시 넣는 게 맞다.
