from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path
from srcs.get_dataset import get_dataloader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_EPOCHS = 20
PATIENCE = 1

device = torch.device("cpu")


load_directory = Path("models/temp")

loaded_model = AutoModelForSequenceClassification.from_pretrained(load_directory)
loaded_tokenizer = AutoTokenizer.from_pretrained(load_directory)

loaded_model.to(device)

print("Model loaded")
label_map = {"human": 0, "llm": 1}

inv_label_map = {v: k for k, v in label_map.items()}  # {0: 'human', 1: 'llm'}


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def test_model(model, test_loader, device):
    model.eval()
    test_loss, test_accuracy = eval_model(model, test_loader, device)
    print(f"\n--- Test Set Performance ---")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # detailed results analysis
    print("\n--- Test Set Classification Report ---")
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=["human", "llm"]))


train_loader, val_loader, test_loader = get_dataloader(MODEL_NAME, MAX_LEN, BATCH_SIZE)

test_model(loaded_model, test_loader, device)
