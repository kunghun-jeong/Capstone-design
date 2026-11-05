"""2단계 노이즈 정리: 감성 점수를 매기기 전에 걸러낼 기사를 표시한다.

행을 지우지 않고 ``drop_reason``(제외 사유, 없으면 None)과 보존용 플래그만 붙인다. 언급량(1단계 집계)은
이 필터와 무관하게 유지되고, 감성 집계에서만 ``drop_reason``이 있는 기사를 뺀다.

제외(drop): 종목 고유 정보가 없는 기사
 - template_whale / template_open_up_down / template_boilerplate: Benzinga 자동 생성 템플릿 (제목·요약 패턴)
 - broad: 태그가 broad_tag_threshold개 초과 (한 종목의 감성이 아님)
플래그만(keep): is_list_promo(리스트·광고성), is_analyst_action(목표가·등급 변경)
기준은 2026-05~09 고유 기사 12,076건 실측에 근거한다 (docs/plans/news_preprocessing_plan.md 7번의 후속).
"""
from __future__ import annotations

import pandas as pd

_WHALE = r"whale (?:activity|alerts?)|whales? (?:with|are|spotted)"
_OPEN_UP_DOWN = r"open up or down today"
_BOILERPLATE_SUMMARY = (
    r"^(?:this whale alert can help traders|benzinga examined the prospects for many investors|"
    r"in (?:today's|the) (?:rapidly|ever-evolving|dynamic|fast-paced)[^.]{0,60}(?:competitive|evolving|changing))"
)
_LIST_PROMO = (
    r"\bstocks? to (?:buy|watch|sell)\b|\bbest (?:growth |ai |tech )?stocks?\b|\bshould you\b|\bis it (?:a )?buy\b|"
    r"\bbetter (?:buy|stock)\b|\bforever stock\b|\bmillionaire\b|\b\d+ (?:top |best )?(?:stocks?|etfs?)\b"
)
_ANALYST = r"\b(?:price target|upgrades?|downgrades?|initiates? coverage|maintains?|reiterates?)\b"


def flag_noise(articles: pd.DataFrame, broad_tag_threshold: int) -> pd.DataFrame:
    """기사 단위 테이블(title, summary, n_tags 필요)에 drop_reason 및 플래그 컬럼을 붙여 돌려준다."""
    title = articles["title"].fillna("")
    summary = articles["summary"].fillna("")
    out = articles.copy()
    flags = {
        "template_whale": title.str.contains(_WHALE, case=False, regex=True) | summary.str.startswith("This whale alert", na=False),
        "template_open_up_down": title.str.contains(_OPEN_UP_DOWN, case=False, regex=True),
        "template_boilerplate": summary.str.contains(_BOILERPLATE_SUMMARY, case=False, regex=True),
        "broad": articles["n_tags"] > broad_tag_threshold,
    }
    reason = pd.Series([None] * len(out), index=out.index, dtype="object")
    for name in ("template_whale", "template_open_up_down", "template_boilerplate", "broad"):  # 앞쪽 사유 우선
        reason = reason.where(reason.notna(), flags[name].map({True: name, False: None}))
    out["drop_reason"] = reason
    out["is_list_promo"] = title.str.contains(_LIST_PROMO, case=False, regex=True)
    out["is_analyst_action"] = title.str.contains(_ANALYST, case=False, regex=True)
    return out


def sentiment_text(articles: pd.DataFrame) -> pd.Series:
    """감성 모델 입력: 제목 + 요약(있으면). 소스 간 비교 가능한 단위(뉴스 계획서 6번)."""
    title = articles["title"].fillna("").str.strip()
    summary = articles["summary"].fillna("").str.strip()
    return (title + ". " + summary).where(summary != "", title).str.strip()
