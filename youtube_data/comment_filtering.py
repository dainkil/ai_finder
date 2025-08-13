import pandas as pd
from pathlib import Path 

class CommentFilter:
    """댓글 스팸 필터"""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def apply_rule_based_filter(self) -> pd.DataFrame:
        """규칙 기반 스팸 필터링 적용"""
        if 'text' not in self.df.columns:
            raise ValueError("No 'text' column found in DataFrame.")

        self.df['text'] = self.df['text'].astype(str)

        # Rule 1: URL 포함 여부
        self.df['contains_url'] = self.df['text'].str.contains(
            r'http|www\.|\.com|\.net|\.org|\.co|\.kr|bit\.ly',
            case=False, na=False
        )

        # Rule 2: 스팸 키워드 포함 여부
        spam_keywords = [
            'subscribe', 'free', 'event', 'giveaway', 'win',
            'promo', 'crypto', 'lotto', 'winner', 'check out my channel'
        ]
        keyword_pattern = '|'.join(spam_keywords)
        self.df['contains_spam_keyword'] = self.df['text'].str.contains(
            keyword_pattern, case=False, na=False
        )

        # Rule 3: 너무 짧은 댓글
        self.df['is_too_short'] = self.df['text'].str.len() <= 3

        # Rule 4: 스팸 ID 포함 여부
        spam_ids = ['19금', '19', 'Free']
        self.df['contains_spam_id'] = self.df['author'].isin(spam_ids)

        # 스팸 여부 종합
        self.df['is_spam_by_rule'] = (
            self.df['contains_url'] |
            self.df['contains_spam_keyword'] |
            self.df['is_too_short'] |
            self.df['contains_spam_id']
        )

        return self.df

    def save_flagged_comments(self, output_path: str):
        """스팸으로 분류된 댓글을 CSV로 저장"""
        flagged = self.df[self.df['is_spam_by_rule'] == True]
        output_file = Path(output_path).expanduser()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        flagged.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"[INFO] Saved {len(flagged)} flagged comments to {output_file}")
