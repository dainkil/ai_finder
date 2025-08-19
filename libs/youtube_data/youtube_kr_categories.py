import os
import requests
from typing import List, Dict, Optional
from srcs.config import GOOGLE_API_KEY

class YouTubeKR_Categories:


    URL = "https://www.googleapis.com/youtube/v3/videoCategories"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("no api key")
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
            results.append({
                "id": item.get("id", ""),
                "title": snippet.get("title", ""),
            })
        return results

    def print_list(self) -> None:
        """
        fetch_categories로 받은 리스트를 콘솔에 단순 출력.
        """
        categories = self.fetch_categories()
        for c in categories:
            print(f'{c["id"]}: {c["title"]}')
