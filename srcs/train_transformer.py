import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from srcs.get_dataset import get_dataloader
import os


save_directory = Path("models/2")
best_model_path = save_directory / "best_model.pt"
os.makedirs(save_directory, exist_ok=True)

MODEL_NAME = "skt/kobert-base-v1"
MAX_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_EPOCHS = 20
PATIENCE = 1


# get dataloader
train_loader, val_loader, test_loader = get_dataloader(MODEL_NAME, MAX_LEN, BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# num_label = 2 /human = 0 llm = 1
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# optimizer for AdamW, Transformer method convention
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"training_loss": f"{loss.item():.3f}"})

    return total_loss / len(data_loader)


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


min_val_loss = np.inf
patience_counter = 0
for epoch in range(MAX_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{MAX_EPOCHS} ---")

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train loss: {train_loss:.4f}")

    val_loss, val_accuracy = eval_model(model, val_loader, device)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")
    if min_val_loss > val_loss:
        print(
            f"Validation Loss improved {val_loss:.4} < {min_val_loss:.4}, saving model"
        )
        min_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        print(
            f"Validation loss didn't improved, patience {patience_counter}/{PATIENCE}."
        )

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print("\nTraining finished.")


final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

final_model.load_state_dict(torch.load(best_model_path))
final_model.to(device)

test_model(final_model, test_loader, device)

final_model.save_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(save_directory)
