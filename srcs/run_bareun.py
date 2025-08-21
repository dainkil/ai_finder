import pandas as pd
from pathlib import Path
from srcs.config import PATH_TO_DB
from libs.baruen_wrapper.grammer_checker import check_text

VIDEO_CODE = "8QKwb3xREmQ"


def run_bareun(video_code: str):
    """
    주어진 영상 코드의 CSV를 읽어서 맞춤법 검사 후 bareun_{video_code}.csv 저장
    출력 컬럼: text, error_types, has_error
    """
    path_to_db = Path(PATH_TO_DB)

    input_csv = path_to_db / f"youtube_comments_{video_code}.csv"
    output_csv = path_to_db / f"bareun_{video_code}.csv"

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[ERROR] 파일 없음: {input_csv}")
        return

    error_types_list = []
    has_error_list = []

    for text in df["text"].fillna(""):
        error_types, has_error = check_text(text)
        error_types_list.append(";".join(error_types) if error_types else "")
        has_error_list.append(has_error)

    result_df = pd.DataFrame({
        "text": df["text"],
        "error_types": error_types_list,
        "has_error": has_error_list
    })

    result_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"검사 완료 → {output_csv}")

    return result_df  # dev.py 같은 곳에서 결과 확인 가능


if __name__ == "__main__":
    # 단독 실행 시 기본 영상코드 하나 테스트
    run_bareun("8QKwb3xREmQ")