# 모델 B용 Reddit 데이터 전처리 계획 (1~2단계)

- 작성일: 2026-11-03
- 범위: Reddit만(뉴스는 별도 담당). 학습 구간 **2026-05-01 ~ 2026-09-30**(귀속 거래일 기준), 호라이즌 3거래일.
- 근거 자료: `AGENTS.md`, `docs/plans/news_preprocessing_plan.md`, `src/capstone/data/news_preprocessing/`,
  `configs/news_preprocessing.yaml`, `reports/reddit_arctic_shift_report.md`(6·7·9절), `src/capstone/data/reddit_arctic_shift.py`,
  `data/raw/reddit/` 파일 목록과 샘플 4행.
- 이 문서는 계획이다. 코드는 승인 후 작성한다.

## ① 현재 상태 요약

1. 원본은 이 PC의 `data/raw/reddit/<subreddit>/{posts,comments}/<YYYY-MM>.jsonl`에 있다. 7개 서브레딧 × 2종류 × 7개월(2026-04-24 ~ 10-23 UTC), 약 529만 건, 2.9GB.
2. 학습 구간 5/1~9/30은 NYSE 거래일 **105일**이다(`trading_sessions`로 확인). 10월 지연 수집 문제(리포트 6절, 15.6%가 하루 이상 늦게 수집)는 **학습 구간 밖**이라 이번엔 영향이 없다.
3. 수집 시작이 2026-04-24 00:00 UTC(= 4/23 20:00 ET)라서, 4/24 거래일에 귀속될 4/23 16:00~20:00 ET 글이 빠져 있다. **4/24는 불완전한 날**이고, 온전한 첫 거래일은 4/27이다. 5/1 이전 온전한 거래일은 4일뿐이다.
4. 원본 필드(샘플 확인):
   - 댓글: `id, name, created_utc, retrieved_on, _meta, subreddit, author, body, link_id, parent_id, link_title, permalink, score, controversiality, is_submitter, stickied, distinguished, edited`
   - 게시글: 위 공통 + `title, selftext, url, domain, is_self, link_flair_text, upvote_ratio, num_comments, over_18, locked, removed_by_category`
   - `_meta`에는 `retrieved_2nd_on` 외에 **`removal_type`, `was_deleted_later`, `edited_title`** 같은 "나중에 알게 된 정보"가 들어 있다(샘플에서 확인). 이 필드들로 거르면 미래 정보가 섞인다(④ 2번).
5. 뉴스 전처리 코드의 `attribute_trading_day()`, `trading_sessions()`, `aggregate_ticker_day()`, `_robust_z()`는 입력 컬럼만 맞추면 그대로 쓸 수 있다. `aggregate_ticker_day()`가 요구하는 컬럼은 `article_id, ticker, attributed_day, n_rows, is_primary, is_broad, publisher`다.
6. **뉴스 설정과 이번 요청이 어긋난다.** `configs/news_preprocessing.yaml`은 8종목(AVGO·MRVL 없음)에 4/1~10/31 구간이다. 이번 요청은 10종목에 5/1~9/30이다. Reddit 설정은 요청대로 하고, 뉴스 설정은 건드리지 않는다(⑤ 질문 5).
7. 이 PC의 `.venv`에 `pandas_market_calendars`가 없어서 `requirements.txt`대로 설치했다. 현재 `python -m pytest`는 104개 전부 통과한다.

## ② 결정 표 (요청 4절의 7항목 + 추가 4항목)

| # | 항목 | 결정 | 근거 | 대안 |
|---|---|---|---|---|
| 1 | 종목 언급 판정 | **티커 매칭.** `(?<![A-Za-z0-9])\$?TICKER(?![A-Za-z0-9])`, 대소문자 구분. `$nvda`처럼 `$`가 붙으면 소문자도 인정한다. GOOG·GOOGL → **GOOGL**(뉴스와 같은 키). **TSMC(대문자)는 TSM으로** 친다. 회사명(뉴스 `aliases` + AVGO: Broadcom, MRVL: Marvell)만 있으면 `is_name_only=True`이고 **집계에서는 기본 제외**한다. 한 글에 여러 종목이 나오면 종목마다 행을 복제하고 `n_tags`(유니버스 종목 수)를 남긴다. 제외 종목은 config의 `other_tickers` 목록(SNDK, INTC, QCOM, ARM, ASML, SMCI 등)으로 `n_other_tickers`를 센다. | 요청의 기본안. 회사명 오탐은 요청자 표본(0.8%)과 수집 리포트 9-3에서 확인됐다. Reddit에서는 TSMC 표기가 흔하다. 기존 `scripts/analyze_reddit.py`도 `TSMC?`로 셌다. | 회사명 + 문맥 필터(주가·실적 단어 동시 출현). 2차 과제로 남긴다. |
| 2 | 정규화 단위 | robust z(60 또는 20거래일 롤링 중앙값/MAD, 당일 제외)를 **두 버전**으로 만든다. (a) 원 건수 `n_articles_z`, `n_primary_z`. (b) 비율 `mention_rate` = 종목 언급 글 수 ÷ 그날 2단계 통과 전체 글 수 × 10,000(만 건당), 그리고 `mention_rate_z`. 비율 z는 단위가 달라서 `scale_floor`를 따로 둔다(`ratio_scale_floor`, 실측 후 결정). | 요청 4-2. 분모를 만들려면 **종목 언급이 없는 글까지 하루 전체 건수**가 필요하다. 그래서 별도 `daily_activity.parquet`(서브레딧·종류·일별 전체/통과 건수)를 만든다. 뉴스의 `scale_floor: 1.0`은 건수 단위라 비율에는 맞지 않는다. | 서브레딧별 비율의 평균. 서브레딧별 테이블(6번)로 나중에 계산할 수 있다. |
| 3 | 베이스라인 워밍업 | **기본: ② 창 단축(window=20, min_periods=20)** + **선택: ① 2/1~4/23 추가 수집**(승인 시 백그라운드). 4/24는 불완전한 날이라 베이스라인에서 뺀다(`baseline_start: 2026-04-27`). | 학습 105거래일 중 z가 비는 날 수: **60/40이면 36일(첫 z 6/24)**, **20/20이면 16일(첫 z 5/26)**, 20/15면 11일(5/18). ①을 하면 5/1 시점 과거가 62거래일이라 60일 창을 뉴스와 똑같이 쓸 수 있다. ① 비용은 리포트 8절 속도(WSB 댓글 월 약 7,600요청, 시간당 5,400~6,000요청)로 **약 25,000요청, 정상이면 4~5시간**이다. 다만 서버 장애가 있었으니(10/26 4.5시간) 수요일 17:00 마감에는 보장이 안 된다. 그래서 ②로 먼저 결과를 내고, ①이 끝나면 config만 바꿔 다시 돌린다. | ③ 6월부터 사용: 학습일 3분의 1이 빠져서 기각한다. |
| 4 | score·num_comments | **피처로 쓰지 않는다.** interim에만 `score_t36h`, `num_comments_t36h`라는 이름으로 남기고, processed 산출물에는 넣지 않는다. | 리포트 6절: 약 36시간 뒤 값으로 갱신된다. 이름에 시점을 박아서 실수로 쓰는 걸 막는다. | 36시간 이상 늦춘 참여도 가중. 나중에 피처 단계에서 실험한다. |
| 5 | 삭제·제거 글 | **댓글:** 본문이 `[removed]`/`[deleted]`면 2단계에서 제거한다. 본문이 없어서 언급 판정도 감성도 할 수 없다. **게시글:** `removed_by_category`(첫 수집 시점 값)를 `is_removed` 플래그로 남기고, 2단계 기본은 **포함**이다. `config: drop_removed_posts`로 끌 수 있게 한다. **`_meta.was_deleted_later`, `_meta.removal_type`은 거르는 데 쓰지 않는다**(분석용 플래그만). | 요청 4-5 기본안. 게시글은 WSB 87%가 제거 표시라 빼면 게시글 신호가 거의 사라진다. 리포트 7절: 본문은 대부분 남아 있다. `was_deleted_later`는 나중에 일어난 일이다. t 시점 필터로 쓰면 "나중에 지워질 글"이라는 미래 정보로 표본을 고르는 셈이 된다. | (a) 제거 표시 글 전부 제외. config로 비교할 수 있다. 팀 결정은 ⑤ 질문 3. |
| 6 | 서브레딧 구분 | `subreddit` 컬럼을 보존한다. `ticker_day.parquet` = **전체 합계**(뉴스와 같은 스키마). `ticker_day_by_subreddit.parquet` = 서브레딧별(같은 컬럼 + `subreddit`). | 뉴스와 병합할 메인 테이블은 키가 (ticker, day) 하나여야 한다. 서브레딧별은 별도 파일로 둔다. | 한 파일에 `subreddit='all'` 행을 섞기. 조인 실수가 생기기 쉬워서 기각한다. |
| 7 | 글 종류·텍스트 | `kind` ∈ {post, comment}. `text_for_sentiment`: 게시글 = `title + "\n" + selftext`(selftext가 비었거나 `[removed]`면 제목만), 댓글 = `body`. 이모지·대소문자는 보존하고, 공백 정규화와 HTML 엔티티 해제만 한다(뉴스 `clean_text` 재사용). 댓글의 `link_title`(소속 게시글 제목)은 문맥용 컬럼으로만 남기고 **언급 판정에는 쓰지 않는다**. | 요청 4-7. `link_title`로 판정하면 게시글 하나의 종목이 댓글 수백 개로 번져서 언급량이 부풀려진다. | 문맥 포함 감성(`link_title` + `body`). 감성 단계에서 실험한다. |
| 8 | 뉴스 스키마 대응 | 글 단위: `article_id`=`reddit:<name>`(예 `reddit:t1_op1cfsg`), `source`=`reddit`, `published_utc`=`created_utc`, `title`=게시글 제목(댓글은 null), `summary`=null, `body`=selftext/body, `url`=`https://www.reddit.com`+permalink, `publisher`=subreddit, `tickers`, `n_tags`. 글×종목에서 `is_primary`, `is_broad`는 ⑩ 참고. | 뉴스 컬럼 이름과 타입을 그대로 쓴다. `publisher`=subreddit로 두면 `n_publishers`가 "언급한 서브레딧 수"가 된다. | — |
| 9 | 작성자 | 원본 `author`는 봇 판정에만 쓰고, 산출물에는 **`author_hash`**(config의 salt + sha256 앞 16자)만 남긴다. | 안건의 개인정보 항목. 작성자별 상한(안건)을 나중에 걸려면 식별자는 있어야 한다. | 작성자 컬럼 삭제. 상한을 못 건다. |
| 10 | `is_primary`·`is_broad` | `is_primary` = 그 글에 나온 유니버스 종목이 **이 종목 하나**(`n_tags == 1`)이고, 게시글이면 **제목**에 티커가 있을 때. `is_broad` = `n_tags + n_other_tickers ≥ broad_tag_threshold`(기본 5, 실측 후 확정). | 뉴스의 "제목에 나오면 primary"를 Reddit에 그대로 옮길 수 없다(댓글엔 제목이 없다). "HODL SNDK MU AMD NVDA" 같은 관심 종목 나열은 broad로 표시한다. | 게시글·댓글 구분 없이 `n_tags == 1`만. 실측 비율을 보고 고른다. |
| 11 | 처리 방식 | 원본을 **파일 단위로 스트리밍**한다. 정규화와 티커 판정을 하면서 (a) 종목이나 회사명이 하나라도 나온 글만 interim에 행으로 남기고, (b) 모든 글은 `daily_activity`에 건수로만 더한다. | 529만 건 전체 텍스트를 메모리에 올리면 16GB 머신에서 위험하다. 언급 없는 글은 분모(2번)에만 필요하다. 언급 행은 하나도 버리지 않으므로 신호가 왜곡되지 않는다. | 전체를 interim에 저장. 디스크 1~2GB, 이점이 적다. |

## ③ 2단계 노이즈 정리 (1단계 산출물 위 별도 단계)

모든 규칙은 **사유별 제거 건수**(전체 / 종목 언급 글 기준 둘 다)를 리포트에 남긴다. 먼저 전체 데이터로 비율을 다시 재고, 요청자 표본(WSB 댓글 4,800건)과 크게 다르면 리포트에 적고 기준을 조정한다.

| 노이즈 | 처리 | 비고 |
|---|---|---|
| 삭제·제거 본문(댓글) | 제거 | ② 5번 |
| 봇(`AutoModerator`, `VisualMod` 등, config 목록) | 제거 | 기존 `analyze_reddit.py`의 BOTS + 실측으로 추가 |
| 운영진 글(`distinguished == "moderator"`)·고정 글(`stickied`) | 제거 | |
| 링크만 있는 글(URL을 빼면 남는 글자 0) | 제거 | |
| **20자 미만 짧은 글** | **`is_short` 플래그 + 감성 입력에서만 제외, 언급량에는 포함** (요청과 다름) | "NVDA 🚀", "MU calls" 같은 짧은 글도 실제 언급이다. 지우면 요청 1절의 "필터가 언급량을 왜곡하면 안 된다"와 충돌한다. 기준 길이는 실측 후 확정한다. ⑤ 질문 2. |
| 같은 날·같은 서브레딧·같은 정규화 본문(복붙·밈) | 1건으로 합치고 `n_rows` 보존 | 뉴스 dedup과 같은 원칙. `n_articles`(고유)와 `n_rows`(원본 행)를 둘 다 집계한다. |
| 이모지 | 보존 | 감성 모델용 |
| 수정된 글(`edited`, `_meta.is_edited`) | `is_edited` 플래그만 | 수정본이 저장됐을 수 있다(미래 텍스트). 비율이 크면 감성 ablation 대상. |
| 회사명만 있는 글 | `is_name_only` 플래그, 집계 기본 제외 | ② 1번 |

- 풍자·반어, 옵션 용어("Bought META puts")는 전처리로 고칠 수 없다. 감성 단계에서 확인하고, 필요하면 은어 사전을 만든다.
- 뉴스 2단계는 건드리지 않는다.

## ④ 시점 정합과 누수 방지

1. **거래일 귀속:** `created_utc` → `attribute_trading_day(cutoff="close")`를 그대로 호출한다. 장 마감 이후·주말·휴장일 글은 다음 거래일로 간다.
2. **"나중에 알게 된 정보"로 거르지 않는다:** score, num_comments, `_meta.was_deleted_later`, `_meta.removal_type`, `_meta.edited_title`. 필터에 쓰는 값은 **첫 수집 시점(`retrieved_on`, 중앙값 약 30초 뒤) 값**만 쓴다. `removed_by_category`도 첫 수집 값이다.
3. **pytest 누수 테스트:**
   - 귀속일 ≥ 발행일(ET)
   - 장 마감 이후 글이 같은 날에 귀속되지 않음
   - 롤링 베이스라인이 당일을 포함하지 않음(건수 z와 비율 z 둘 다)
   - 휴장일에 귀속된 행 0
   - processed 산출물에 score 계열 컬럼이 없음
4. **기간:** processed의 `ticker_day`는 귀속일 5/1~9/30만 담는다. 베이스라인 계산용 4/27~4/30은 계산 뒤 잘라낸다. 귀속일 원본이 필요하면 interim에 남아 있다.

## ⑤ 산출물·구조

| 경로 | 내용 |
|---|---|
| `configs/reddit_preprocessing.yaml` | tickers(10), `ticker_patterns`(GOOGL: `GOOGL?`, TSM: `TSMC?`), aliases, `other_tickers`, period(`baseline_start`, start, end), cutoff, baseline(window, min_periods, scale_floor, ratio_scale_floor), `broad_tag_threshold`, bots, `min_chars`, `drop_removed_posts`, `author_salt`, paths |
| `src/capstone/data/reddit_preprocessing/` | `config.py`, `normalize.py`(스트리밍 정규화, 티커 판정), `clean.py`(2단계 노이즈), `pipeline.py`(조립과 실행 로그). **집계는 `news_preprocessing.aggregate`를 import한다.** 비율 z만 `_robust_z`를 재사용하는 얇은 함수를 추가한다. |
| `scripts/preprocess_reddit.py` | CLI (`--config`, `--stage {1,2,all}`) |
| `data/interim/reddit/posts_comments.parquet` | 정규화 후, 노이즈 필터 전. 종목·회사명이 나온 글만, 플래그 포함 |
| `data/interim/reddit/daily_activity.parquet` | (subreddit, kind, day)별 전체 건수, 2단계 통과 건수. 비율의 분모 |
| `data/processed/reddit/article_ticker.parquet` | 2단계 후 글×종목. 뉴스 컬럼 + `subreddit, kind, is_name_only, is_short, is_removed, is_edited, n_other_tickers, author_hash, text_for_sentiment` |
| `data/processed/reddit/ticker_day.parquet` | 뉴스 컬럼(`ticker, day, n_articles, n_rows, n_primary, n_broad, n_publishers, article_ids, n_articles_z, n_primary_z`) + `n_total_items, mention_rate, mention_rate_z` |
| `data/processed/reddit/ticker_day_by_subreddit.parquet` | 위와 같은 구조 + `subreddit` |
| `data/processed/reddit/_run.json` | config 해시, 입력 파일 해시(98개), git sha, 단계별 행 수 |
| `tests/data/reddit_preprocessing/` | 티커 판정(대소문자, `$`, GOOG 합침, TSMC, 단어 경계), 회사명 플래그, 정규화 스키마, 노이즈 규칙별 fixture, 복붙 합치기, 누수 테스트 5종 |
| `reports/reddit_preprocessing_report.md` | 종목·월별 건수, 단계별·사유별 제거 건수, 결측률, 노이즈 비율 재측정(요청자 표본과 비교), 누수 테스트 결과, 워밍업으로 z가 빈 날 수 |

**순서와 소요 예상**
1. 설정과 정규화, fixture 테스트
2. 전체 데이터로 1단계 실행(스트리밍, 수 분~수십 분으로 추정)
3. 노이즈 비율 재측정과 기준 확정
4. 2단계와 집계
5. 리포트, `python -m pytest`, PR

수요일 17:00 마감에 맞춰 한 PR(`feature/reddit-preprocessing` → `main`)로 올린다. 감성 점수화는 이 PR 범위 밖이고, `text_for_sentiment`가 그 입력이다.

## ⑥ 확인해야 할 질문 (최대 5개)

1. **워밍업:** 창 20일로 먼저 가고(학습일 16일 z 없음), 2/1~4/23 추가 수집(약 4~5시간, 장애 시 더 걸림)을 백그라운드로 돌려 60일로 다시 맞추는 안에 동의하는가? 추가 수집은 Arctic Shift에 약 2.5만 요청을 보낸다.
2. **짧은 글:** 20자 미만을 **지우지 않고** 감성에서만 빼는 쪽으로 바꿨다. 요청은 "제거"였다. 언급량 왜곡 원칙 때문인데, 이대로 가도 되는가?
3. **제거 표시 게시글:** 기본을 "포함 + 플래그"로 두었다. 팀이 (a) 제외로 정하면 config 한 줄이다. 팀 결정을 기다릴지, 이 기본으로 진행할지?
4. **TSMC를 TSM으로 집계:** 뉴스는 태그 기반이라 해당이 없다. Reddit만 TSMC를 넣으면 TSM 언급량이 늘어난다. 넣는 게 맞다고 보는데 동의하는가?
5. **뉴스 설정 불일치:** 뉴스 config는 8종목에 4/1~10/31이다. Reddit은 요청대로 10종목에 5/1~9/30으로 간다. 뉴스 쪽 갱신은 뉴스 담당에게 넘기면 되는가?

## 확인되지 않은 가정

- 추가 수집 시간 4~5시간은 10/26 정상 구간 속도로 **추정**한 값이다. 2월~4월 WSB 활동량이 5~9월과 비슷하다는 가정이다.
- 종목 언급 글 비율은 수집 리포트의 언급 통계로 보면 전체의 몇 %대로 **추정**된다. 그래서 interim 크기가 작을 것으로 본다. 실측 전이다.
- `removed_by_category`가 첫 수집 시점 값이라는 것은 수집기가 첫 응답을 저장한다는 점에서 **판단**했다. Arctic Shift가 2차 수집 때 이 필드를 덮어쓰는지는 **미확인**이다. 덮어쓴다면 이 필드도 미래 정보가 된다. 구현 중 `retrieved_2nd_on`이 있는 글과 없는 글의 비율을 비교해 확인한다.
- `$`가 붙은 소문자(`$nvda`) 허용, `broad_tag_threshold=5`, `min_chars=20`은 실측 전 기본값이다.
- `author_salt`는 config에 평문으로 둔다. 비밀값 수준의 보호는 아니다(같은 salt를 알면 알려진 아이디를 대조할 수 있다). 수업용 내부 데이터라는 전제다.
