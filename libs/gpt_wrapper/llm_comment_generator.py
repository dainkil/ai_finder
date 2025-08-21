from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Literal


class CommentGenerator:

    def __init__(self, api_key: str, system_prompt: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = system_prompt

    def generate_comment(self, transcript, temperature=1.2):
        # try:
        #     transcript = get_youtube_transcript(video_id)
        # except Exception as e:
        #     print(f"error occured when generating comments: {e}")
        #     return

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{transcript}"},
            ],
            temperature=temperature,
        )

        return completion.choices[0].message.content


def get_youtube_transcript(video_id):
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(
            video_id, languages=["ko", "en"]
        )
        transcript = ""
        for snippet in fetched_transcript:
            transcript += snippet.text

        return transcript

    except Exception as e:
        print(f"transcript not fetchable: {e}")
        raise e


def get_system_prompt(
    age: Literal[10, 20, 40, 60],
    political_type: Literal["극우", "극좌", "우", "좌", "중도"],
    gender: Literal["남성", "여성"],
    length: Literal[1, 2, 4, 5, 10],
):
    prompt = f"""
    너는 한국에 사는 {age}대 {gender}이고, {political_type}파적 정치 성향을 가지고 있어.
    지금부터 나는 너에게 유튜브 영상의 자막을 제공할거야.
    해당 인물이 남길만한 댓글을 만들어서 답변해.
    인물에 대한 특징을 간략하게 설명해주면 다음과 같아.
    """

    if age == 10:
        prompt += "단어의 수준이 매우 낮고, 문장 길이 또한 짧아야해. 상황을 이해하지 못하고 적어도 좋아."
    elif age == 20:
        prompt += "문장의 길이는 짧고, 음 또는 슴으로 끝나는 말투를 구사해. 예를 들어서 '그렇지 않아?' --> '그렇지 않음?'."
    elif age == 40:
        prompt += "문장 사이에 이모티콘을 적절히 사용하고, 감성적인 짧은 문장을 사용해."
    elif age == 60:
        prompt += "맞춤법이 자주 틀리고, 쉼표나 마침표 여러개 사용해. 띄어쓰기 대신 쉼표를 사용하는 경우가 많아."

    if political_type == "극우":
        prompt += "대한민국의 안보에 대한 큰 걱정을 가지고 있어. 문제점들을 이재명과 문재인의 탓으로 돌려."
    elif political_type == "극좌":
        prompt += (
            "문제를 윤석열과 이명박의 탓으로 돌려. 윤석열과 김건희 부부의 구속을 원해."
        )
    elif political_type == "중도":
        prompt += "정치적인 색이 없는 인물을 연기해야해."
    elif political_type == "우":
        prompt += "약간의 우파적인 성향을 가지고는 있지만 극단적으로 표현하지는 않아."
    elif political_type == "좌":
        prompt += "약간의 우파적인 성향을 가지고는 있지만 극단적으로 표현하지는 않아."

    prompt += (
        f"그럼 이제, 제공되는 유튜브 영상 자막에 대해 {length} 문장의 답변을 생성해줘."
    )

    return prompt
