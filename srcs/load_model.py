from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_directory = Path("models/temp")

loaded_model = AutoModelForSequenceClassification.from_pretrained(load_directory)
loaded_tokenizer = AutoTokenizer.from_pretrained(load_directory)

loaded_model.to(device)

print("Model loaded")
label_map = {"human": 0, "llm": 1}

inv_label_map = {v: k for k, v in label_map.items()}  # {0: 'human', 1: 'llm'}


def predict(text: str):
    loaded_model.eval()

    encoding = loaded_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_class_idx = torch.max(probabilities, dim=1)

    predicted_label = inv_label_map[int(predicted_class_idx.item())]

    return {"predicted_label": predicted_label, "confidence": confidence.item()}


# --- 함수 사용 예시 ---
my_text_1 = "안녕하세요, 오늘 날씨가 정말 좋네요. 주말에 나들이 가기 딱이겠어요."
my_text_2 = "네, 고객님. 원하시는 정보를 바탕으로 최적의 답변을 생성해 드리겠습니다."

# 예측 실행
result_1 = predict(my_text_1)
result_2 = predict(my_text_2)

print(f'입력 텍스트 1: "{my_text_1}"')
print(
    f"예측 결과: {result_1['predicted_label']} (신뢰도: {result_1['confidence']:.4f})\n"
)

print(f'입력 텍스트 2: "{my_text_2}"')
print(
    f"예측 결과: {result_2['predicted_label']} (신뢰도: {result_2['confidence']:.4f})"
)
