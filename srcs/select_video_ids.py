from libs.gpt_wrapper import llm_comment_generator as llm_gen
from srcs.config import OPENAI_API_KEY, GOOGLE_API_KEY
from libs.youtube_data import VideoSelector
from datetime import datetime, timezone
from pathlib import Path
import csv


times = []
category_ids = []
for month in range(1, 13, 4):
    for year in [2025, 2024, 2019, 2018, 2020, 2021, 2022, 2023]:
        times.append(
            [
                datetime(year=year, month=month, day=1, tzinfo=timezone.utc),
                datetime(year=year, month=month + 3, day=28, tzinfo=timezone.utc),
            ]
        )
        if year >= 2022:
            category_ids.append("42")  # shorts was world wide published.
        else:
            category_ids.append(
                "25"
            )  # before the shorts was on, news chapter would be the hot field.


video_manager = VideoSelector(api_key=GOOGLE_API_KEY)

selected_video_csv_path = Path("dbs/video_id_lists.csv")


with open(selected_video_csv_path, "w", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow(["video_id", "published_at", "title", "channel_title"])
    for (start_time, end_time), category_id in zip(times, category_ids):
        params = VideoSelector.get_default_param(
            q="충격적",
            category_id=category_id,
            published_before=end_time,
            published_after=start_time,
            order="viewCount",
        )
        print("We send :", params)
        vid_list = video_manager.get_videos_from_params(params, 4)

        print("Grab :", len(vid_list), "numbers of vid info")
        datum = []
        # list of dicts, keys = {title, video_id, published_at, channel_title}
        for vid_info in vid_list:
            datum.append(
                [
                    vid_info["video_id"],
                    vid_info["published_at"],
                    vid_info["title"],
                    vid_info["channel_title"],
                ]
            )

        if datum:
            writer.writerows(datum)
            print("successfully saved selected video ids")
