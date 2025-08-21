import os
import requests
from datetime import datetime
from typing import List, Dict
from datetime import datetime
from googleapiclient.discovery import build


class VideoSelector:
    SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.timeout = timeout

    @staticmethod
    def get_default_param(
        q: str,  # searching keyword
        category_id: str,
        published_before: datetime,
        published_after: datetime,
        region_code: str = "KR",
        relevance_language: str = "ko",
        order: str = "date",
        per_page: int = 50,
    ):
        published_before_str = published_before.strftime("%Y-%m-%dT%H:%M:%SZ")
        published_after_str = published_after.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "q": q,
            "part": "snippet",
            "type": "video",
            "regionCode": region_code,
            "relevanceLanguage": relevance_language,
            "videoCategoryId": category_id,
            "order": order,
            "publishedBefore": published_before_str,
            "publishedAfter": published_after_str,
            "maxResults": per_page,
        }

    def get_videos_from_params(self, params: Dict, search_num: int = 4):
        videos = []
        page_token = None
        for _ in range(search_num):
            if page_token:
                params["pageToken"] = page_token
            current_page_vids, page_token = self._get_videos_single_page(params)
            videos.extend(current_page_vids)
            if not page_token:
                break

        return videos  # list of dicts, keys = {title, video_id, published_at, channel_title}

    def _get_videos_single_page(self, params: Dict):
        search_response = self.youtube.search().list(**params).execute()
        # print("Search_response : ", search_response)
        videos = []
        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                video_info = {
                    "title": search_result["snippet"]["title"],
                    "video_id": search_result["id"]["videoId"],
                    "published_at": search_result["snippet"]["publishedAt"],
                    "channel_title": search_result["snippet"]["channelTitle"],
                }
                videos.append(video_info)
        next_page_token = search_response.get("nextPageToken")

        return videos, next_page_token


class GetYouTubeCategories:

    URL = "https://www.googleapis.com/youtube/v3/videoCategories"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout

    def fetch_categories(self) -> List[Dict[str, str]]:
        """
        regionCode=KR, relevanceLanguage=ko 설정으로 카테고리 목록을 가져와
        [{"id": "10", "title": "음악"}, ...] 형태의 리스트로 반환.
        assignable 필터링 없이 응답에 있는 항목을 그대로 사용합니다.
        """
        params = {
            "part": "snippet",
            "regionCode": "KR",
            "relevanceLanguage": "ko",
            "key": self.api_key,
        }
        resp = requests.get(self.URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            results.append(
                {
                    "id": item.get("id", ""),
                    "title": snippet.get("title", ""),
                }
            )
        return results

    def print_list(self) -> None:
        """
        fetch_categories로 받은 리스트를 콘솔에 단순 출력.
        """
        categories = self.fetch_categories()
        for c in categories:
            print(f'{c["id"]}: {c["title"]}')
