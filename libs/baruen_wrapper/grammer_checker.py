import requests
from bareunpy import Tagger
from srcs.config import BAREUN_API_KEY

API_URL = "https://bareun.busan.ac.kr/bareun.RevisionService/CorrectError"

HEADERS = {
    "Authorization": f"Bearer {BAREUN_API_KEY}",
    "Content-Type": "application/json",
}


def check_text(text: str):
    """
    Bareun API 호출 → (error_types, has_error) 반환
    error_types: 오류 종류 리스트 (철자/문체/다수어절 등)
    has_error: 오류 여부 (True/False)
    """
    if not text.strip():
        return [], False

    payload = {"text": text}
    try:
        res = requests.post(API_URL, json=payload, headers=HEADERS, timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception:
        return ["api_error"], True

    errors = data.get("errors", [])
    if not errors:
        return [], False

    error_types = {err.get("type", "unknown") for err in errors}
    return list(error_types), True
