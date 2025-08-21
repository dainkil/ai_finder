from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve, auc
from srcs.get_dataset import get_dataloader
from srcs.config import MAX_LEN, MODEL_NAME, BATCH_SIZE

AUROC = True
prediction = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_directory = Path("models/1")

model_pred_dbs_folder = Path("dbs/1")
path_to_real_comments = Path("dbs/real_comments_list.csv")


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


if AUROC == True:
    _, _, test_loader = get_dataloader(MODEL_NAME, MAX_LEN, BATCH_SIZE)

    def get_labels_and_probabilities(model, data_loader, device):
        loaded_model.eval()
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calculating probabilities for AUROC"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=1)

                llm_probs = probabilities[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(llm_probs.cpu().numpy())

        return np.array(all_labels), np.array(all_probabilities)

    y_true, y_scores = get_labels_and_probabilities(loaded_model, test_loader, device)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(10, 8))

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

if prediction == True:
    os.makedirs(model_pred_dbs_folder, exist_ok=True)
    path_to_save_pred_results = model_pred_dbs_folder / "prediction_result.csv"

    if os.path.exists(path_to_save_pred_results):
        df = pd.read_csv(path_to_save_pred_results)
    else:
        wanted_db = []

        df_real_comments = pd.read_csv(path_to_real_comments)
        df_real_comments["published_at"] = pd.to_datetime(
            df_real_comments["published_at"]
        )

        for index, row in df_real_comments.iterrows():
            date = row["published_at"]
            text = row["text"]
            if date.year < 2022:
                pass
            else:
                wanted_db.append((text, date))

        results_list = []
        for text, date in tqdm(wanted_db, desc="Testing"):
            prediction_result = predict(text)
            results_list.append(
                {
                    "text": text,
                    "date": date,
                    "predicted_label": prediction_result["predicted_label"],
                    "confidence": prediction_result["confidence"],
                }
            )

        df = pd.DataFrame(results_list)
        df.to_csv(path_to_save_pred_results, index=False, encoding="utf-8-sig")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    monthly_counts = (
        df.groupby(["month", "predicted_label"]).size().unstack(fill_value=0)
    )

    monthly_proportions = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100
    start_period = pd.Period("2022-01", "M")
    end_period = pd.Period("2025-08", "M")
    monthly_proportions = monthly_proportions.loc[start_period:end_period]

    ax = monthly_proportions.plot(
        kind="bar",
        stacked=True,
        figsize=(18, 7),
        colormap="viridis",
        width=0.8,
    )
    plt.title(
        "Montly LLM-Human ratio prediction Results (2022-01 ~ 2025-08)", fontsize=16
    )
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("ratio (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")  # x축 레이블 45도 회전
    plt.ylim(0, 100)  # y축 범위를 0-100으로 고정

    plt.legend(title="prediction label", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.5)

    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f%%', label_type='center', color='white', fontsize=8)

    plt.tight_layout()

    plt.show()
