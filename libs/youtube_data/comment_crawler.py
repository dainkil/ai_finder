import pandas as pd
from googleapiclient.discovery import build
from pathlib import Path


class CommentCrawler:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def fetch_comments(self, video_id: str) -> list[dict]:
        """특정 유튜브 영상의 댓글과 대댓글 수집"""
        comments_list = []
        try:
            response = (
                self.youtube.commentThreads()
                .list(part="snippet,replies", videoId=video_id, maxResults=100)
                .execute()
            )

            while response:
                for item in response.get("items", []):
                    # 상위 댓글
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments_list.append(
                        {
                            "author": top_comment["authorDisplayName"],
                            "published_at": top_comment["publishedAt"],
                            "like_count": top_comment["likeCount"],
                            "text": top_comment["textDisplay"],
                            "is_reply": False,
                        }
                    )

                    # 대댓글
                    if "replies" in item:
                        for reply_item in item["replies"]["comments"]:
                            reply = reply_item["snippet"]
                            comments_list.append(
                                {
                                    "author": reply["authorDisplayName"],
                                    "published_at": reply["publishedAt"],
                                    "like_count": reply["likeCount"],
                                    "text": reply["textDisplay"],
                                    "is_reply": True,
                                }
                            )

                if "nextPageToken" in response:
                    response = (
                        self.youtube.commentThreads()
                        .list(
                            part="snippet,replies",
                            videoId=video_id,
                            pageToken=response["nextPageToken"],
                            maxResults=100,
                        )
                        .execute()
                    )
                else:
                    break
        except Exception as e:
            print(f"[ERROR] Failed to fetch comments: {e}")
            return []

        return comments_list

    def to_csv(self, comments: list[dict], filename: Path):
        """댓글 데이터를 CSV 파일로 저장"""
        df = pd.DataFrame(comments)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved {len(df)} comments to {filename}")
