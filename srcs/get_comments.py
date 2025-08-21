import pandas as pd
from srcs.config import GOOGLE_API_KEY, PATH_TO_DB
from libs.youtube_data import CommentCrawler, CommentFilter
from pathlib import Path
from typing import List
import csv

path_to_video_id_lists = Path("dbs/video_id_lists.csv")
df = pd.read_csv(path_to_video_id_lists)

list_of_video_ids = df["video_id"].tolist()

real_comments_list_path = Path("dbs/real_comments_list.csv")

with open(real_comments_list_path, "w", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "published_at", "like_count", "author", "is_reply"])

    # list of dicts key = {"author", "published_at", "like_count", "text", "is_reply"}
    crawler = CommentCrawler(GOOGLE_API_KEY)

    for video_id in list_of_video_ids:
        print(f"[INFO] Fetching comments for video: {video_id} ...")
        comments = crawler.fetch_comments(video_id)

        datum = []
        if not comments:
            print("[ERROR] No comments fetched. Exiting.")
            continue
        else:
            print(f"[INFO] Total {len(comments)} number of comment found")
            for comment in comments:
                datum.append(
                    [
                        comment["text"],
                        comment["published_at"],
                        comment["like_count"],
                        comment["author"],
                        comment["is_reply"],
                    ]
                )
            writer.writerows(datum)


# def get_youtube_comments(list_of_video_ids: List[str], path_to_db: Path):
#     # grab youtube comments of the list of video ids
#     for video_id in list_of_video_ids:
#         crawler = CommentCrawler(GOOGLE_API_KEY)
#         print(f"[INFO] Fetching comments for video: {video_id} ...")
#         comments = crawler.fetch_comments(video_id)

#         if not comments:
#             print("[ERROR] No comments fetched. Exiting.")
#             continue

#         csv_filename = path_to_db / f"youtube_comments_{video_id}.csv"
#         crawler.to_csv(comments, csv_filename)

#         print("[INFO] Done.")
