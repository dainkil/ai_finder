import pandas as pd
from srcs.config import API_KEY, PATH_TO_DB
from libs.youtube_data import CommentCrawler, CommentFilter
from pathlib import Path


list_of_video_ids = []
path_to_db = Path(PATH_TO_DB)


for video_id in list_of_video_ids:

    csv_filename = path_to_db / f"youtube_comments_{video_id}.csv"
    df = pd.read_csv(csv_filename)
    comment_filter = CommentFilter(df)
    filtered_df = comment_filter.apply_rule_based_filter()

    comment_filter.save_flagged_comments(
        path_to_db / "flagged_spam_comments_{video_id}.csv"
    )
