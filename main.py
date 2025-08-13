import pandas as pd
from config import API_KEY
from youtube_data import CommentCrawler, CommentFilter


def main():
    # 1. 영상 ID 입력
    video_id = input("Enter YouTube video ID: ").strip()

    # 2. 댓글 크롤링
    crawler = CommentCrawler(API_KEY)
    print(f"[INFO] Fetching comments for video: {video_id} ...")
    comments = crawler.fetch_comments(video_id)

    if not comments:
        print("[ERROR] No comments fetched. Exiting.")
        return

    # 3. CSV 저장
    csv_filename = f"youtube_comments_{video_id}.csv"
    crawler.to_csv(comments, csv_filename)

    # 4. 필터링 적용
    df = pd.read_csv(csv_filename)
    comment_filter = CommentFilter(df)
    filtered_df = comment_filter.apply_rule_based_filter()

    # 5. 스팸 저장
    comment_filter.save_flagged_comments(
        "~/Documents/Workspace/project/ai_finder/flagged_spam_comments.csv"
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()