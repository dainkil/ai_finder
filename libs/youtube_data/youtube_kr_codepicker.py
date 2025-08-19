import os
import time
import math
import csv
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone  # ← timezone 추가



class KRCategory_SpikePicker:
    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("no api key (set env GOOGLE_API_KEY or pass api_key)")
        self.timeout = timeout

    def _req(self, url: str, params: Dict) -> Dict:
        params = {**params, "key": self.api_key}
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _recent_candidates(
        self,
        category_id: str,
        days_window: int = 7,
        region_code: str = "KR",
        relevance_language: str = "ko",
        pages_date: int = 4,
        pages_view: int = 2,
        per_page: int = 50,
    ) -> List[str]:
        now_utc = datetime.now(timezone.utc)  
        published_after = (now_utc - timedelta(days=days_window)).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = set()

        # 최신 업로드 정렬
        page_token = None
        for _ in range(pages_date):
            params = {
                "part": "snippet",
                "type": "video",
                "regionCode": region_code,
                "relevanceLanguage": relevance_language,
                "videoCategoryId": category_id,
                "order": "date",
                "publishedAfter": published_after,
                "maxResults": per_page,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._req(self.SEARCH_URL, params)
            for it in data.get("items", []):
                vid = it.get("id", {}).get("videoId")
                if vid:
                    ids.add(vid)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)

        # 최근 기간 내 조회수 상위(풀 확장)
        page_token = None
        for _ in range(pages_view):
            params = {
                "part": "snippet",
                "type": "video",
                "regionCode": region_code,
                "relevanceLanguage": relevance_language,
                "videoCategoryId": category_id,
                "order": "viewCount",
                "publishedAfter": published_after,
                "maxResults": per_page,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._req(self.SEARCH_URL, params)
            for it in data.get("items", []):
                vid = it.get("id", {}).get("videoId")
                if vid:
                    ids.add(vid)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)

        return list(ids)

    def _fetch_stats(self, video_ids: List[str]) -> Dict[str, Dict]:
        out = {}
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            params = {
                "part": "snippet,statistics,status,contentDetails",
                "id": ",".join(chunk),
            }
            data = self._req(self.VIDEOS_URL, params)
            for it in data.get("items", []):
                out[it["id"]] = it
            time.sleep(0.1)
        return out

    @staticmethod
    def _parse_published_at(item: Dict) -> Optional[datetime]:
        pub = item.get("snippet", {}).get("publishedAt")
        if not pub:
            return None
        try:
            return datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    def _velocity_score(self, old: Dict, new: Dict, delta_hours: float) -> float:
        if delta_hours <= 0:
            return -1e9

        # 비공개/지역 차단 제외
        if new.get("status", {}).get("privacyStatus") != "public":
            return -1e9
        rr = new.get("contentDetails", {}).get("regionRestriction", {})
        if "blocked" in rr and "KR" in rr["blocked"]:
            return -1e9

        os_ = old.get("statistics", {})
        ns_ = new.get("statistics", {})

        ov = int(os_.get("viewCount", 0))
        ol = int(os_.get("likeCount", 0) or 0)
        oc = int(os_.get("commentCount", 0) or 0)

        nv = int(ns_.get("viewCount", 0))
        nl = int(ns_.get("likeCount", 0) or 0)
        nc = int(ns_.get("commentCount", 0) or 0)

        dv = max(nv - ov, 0)
        dl = max(nl - ol, 0)
        dc = max(nc - oc, 0)

        vph = dv / delta_hours
        lph = dl / delta_hours
        cph = dc / delta_hours

        # 업로드 신선도 가산
        pub_dt = self._parse_published_at(new) or datetime.utcnow()
        hours_since = max((datetime.utcnow() - pub_dt).total_seconds() / 3600.0, 1 / 60)

        # 고정 인기 회피: 누적 조회수 로그 가중치 낮춤(감산)
        size_penalty = -0.25 * math.log10(max(nv, 1))

        # 참여율 보너스: 증가분 기준
        engage_ratio = (dl + dc) / dv if dv > 0 else 0.0
        engage_bonus = min(0.5, engage_ratio)

        score = (
            math.log10(vph + 1) * 1.3
            + math.log10(lph + 1) * 1.0
            + math.log10(cph + 1) * 0.8
            + min(1.5, 1.5 * (48.0 / (hours_since + 1.0)))
            + size_penalty
            + engage_bonus
        )
        return score

    def select_videos(
        self,
        category_id: str,
        top_k: int = 30,
        days_window: int = 7,
        region_code: str = "KR",
        snapshot_gap_minutes: int = 60,
        percentile: float = 0.2,  # 상위 20% 절삭
        old_snapshot: Optional[Dict[str, Dict]] = None,  # 과거 스냅샷(T-60 등)
        new_snapshot: Optional[Dict[str, Dict]] = None,  # 최근 스냅샷(T-30 등)
        delta_hours: Optional[float] = None,             # 두 스냅샷 간 시간(시간 단위)
        old_ts: Optional[datetime] = None,               # 과거 스냅샷 수집 시각(선택)
        new_ts: Optional[datetime] = None,               # 최근 스냅샷 수집 시각(선택)
        use_recent_candidates: bool = True,              # 후보 교집합 사용할지 여
    ) -> List[str]:
        # 1) 최근 업로드 후보 수집
        candidates = self._recent_candidates(
            category_id=category_id,
            days_window=days_window,
            region_code=region_code,
            relevance_language="ko",
        ) if use_recent_candidates else []

        if old_snapshot is None or new_snapshot is None:
            raise ValueError("old_snapshot and new_snapshot must be provided to avoid waiting.")
        if delta_hours is None:
            if not (old_ts and new_ts):
                raise ValueError("Provide delta_hours or both old_ts and new_ts.")
            delta_hours = max((new_ts - old_ts).total_seconds() / 3600.0, 1/60)

        snap_a = old_snapshot
        snap_b = new_snapshot

        # 5) 점수 계산
        scored: List[Tuple[float, str]] = []
        iterable = snap_b.items() if not use_recent_candidates else [
        (vid, snap_b[vid]) for vid in candidates if vid in snap_b
    ]


        for vid, bitem in iterable:
            aitem = snap_a.get(vid)
            if not aitem:
                continue
            # 최근성 필터: 업로드 1~7일 내만
            pub_dt = self._parse_published_at(bitem)
            if not pub_dt:
                continue
            days_old = (datetime.now(timezone.utc) - pub_dt).days
            if days_old < 0 or days_old > days_window:
                continue

            s = self._velocity_score(aitem, bitem, delta_hours)
            if s > -1e9:
                scored.append((s, vid))

        # 6) 상대적 급상승: 상위 퍼센타일 → top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        cut_index = max(1, int(len(scored) * percentile))
        winners = [vid for _, vid in scored[:cut_index]][:top_k]

        return winners
        # 7) CSV 저장 (videoId만)
        """ with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["videoId"])
            for vid in winners:
                writer.writerow([vid])
 """
