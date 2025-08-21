from libs.gpt_wrapper import llm_comment_generator as llm_gen
from srcs.config import OPENAI_API_KEY, GOOGLE_API_KEY
from libs.youtube_data import VideoSelector
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from typing import Literal, get_args, get_type_hints
import csv
import time

path_to_video_id_lists = Path("dbs/video_id_lists.csv")
df = pd.read_csv(path_to_video_id_lists)

list_of_video_ids = df["video_id"].tolist()

start_vid_id = "KkeM17bP6GI"
flag = 0

type_hints = get_type_hints(llm_gen.get_system_prompt)

llm_generated_comments_list_path = Path("dbs/gpt_4o_mini.csv")

with open(llm_generated_comments_list_path, "w", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["generated_text", "video_id", "political_type", "age", "gender", "length"]
    )

    for video_id in list_of_video_ids:
        if video_id != start_vid_id and flag == 0:
            continue
        if video_id == start_vid_id:
            flag = 1

        print(f"[INFO] started for {video_id}")
        try:
            transcript = llm_gen.get_youtube_transcript(video_id)
        except Exception as e:
            print(f"error occured when generating comments: {e}")
            continue

        for political_type in get_args(type_hints["political_type"]):
            for age in get_args(type_hints["age"]):
                for gender in get_args(type_hints["gender"]):
                    for length in get_args(type_hints["length"]):

                        prompt = llm_gen.get_system_prompt(
                            age=age,
                            political_type=political_type,
                            gender=gender,
                            length=length,
                        )

                        commentator_bot = llm_gen.CommentGenerator(
                            OPENAI_API_KEY, system_prompt=prompt
                        )

                        comment = commentator_bot.generate_comment(
                            transcript, temperature=1.2
                        )
                        print(
                            f"[INFO] {political_type}, {age}, {gender}, {length} generated {comment}"
                        )

                        writer.writerow(
                            [comment, video_id, political_type, age, gender, length]
                        )

        time.sleep(1)
