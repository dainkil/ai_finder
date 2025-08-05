import pandas as pd
from googleapiclient.discovery import build
from config import API_KEY

VIDEO_ID = "ZCngKo4zBH8"

def commentCrawling(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    commentsList = []

    try:
        response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=500
        ).execute()
    
        while response:
            for item in response['items']:

            #댓글
                top_comment = item['snippet']['topLevelComment']['snippet']
                commentsList.append({
                    'author': top_comment['authorDisplayName'],
                    'published_at': top_comment['publishedAt'],
                    'like_count': top_comment['likeCount'],
                    'text': top_comment['textDisplay'],
                    'is_reply': False # 대댓글 여부 표시
                })
            #대댓글
                if 'replies' in item:
                    for reply_item in item['replies']['comments']:
                        reply = reply_item['snippet']
                        commentsList.append({
                            'author': reply['authorDisplayName'],
                            'published_at': reply['publishedAt'],
                            'like_count': reply['likeCount'],
                            'text': reply['textDisplay'],
                            'is_reply': True # 대댓글 여부 표시
                        })
        
            if 'nextPageToken' in response:
                # 다음 페이지가 있다면, pageToken을 사용하여 다시 요청
                response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100
                ).execute()
            else:
                # 다음 페이지가 없으면 반복문 종료
                break

    except Exception as e:
        print(f"ERROR: {e}")
        # API 할당량 초과, 비공개 동영상 등 다양한 오류가 발생할 수 있습니다.
        return None

    return commentsList

if __name__ == "__main__":
    print(f"'{VIDEO_ID}' start comment crawling...")
    
    all_comments = commentCrawling(API_KEY, VIDEO_ID)
    
    if all_comments:

        df = pd.DataFrame(all_comments)
        
        file_name = f'youtube_comments_{VIDEO_ID}.csv'
        
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        
        print(f"Successfully collected {len(df)} comments and saved to '{file_name}'.")
        print("\n--- Data Sample (Top 5 Rows) ---")
        print(df.head())