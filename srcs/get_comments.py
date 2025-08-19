import pandas as pd
from srcs.config import API_KEY, PATH_TO_DB
from libs.youtube_data import CommentCrawler, CommentFilter
from pathlib import Path


list_of_video_ids = []
path_to_db = Path(PATH_TO_DB)


for video_id in list_of_video_ids:
    crawler = CommentCrawler(API_KEY)
    print(f"[INFO] Fetching comments for video: {video_id} ...")
    comments = crawler.fetch_comments(video_id)

    if not comments:
        print("[ERROR] No comments fetched. Exiting.")
        continue

    csv_filename = path_to_db / f"youtube_comments_{video_id}.csv"
    crawler.to_csv(comments, csv_filename)

    print("[INFO] Done.")
