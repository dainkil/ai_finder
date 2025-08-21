import pandas as pd
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

path_to_real_comments = Path("dbs/real_comments_list.csv")
path_to_llm_comments = Path("dbs/gpt_4o_mini.csv")
path_to_dataset_pickle = Path("dbs/dataset.pkl")


# return dataset with key "total_db", "wanted_db"
# dataset["total_db"] is fully labeled text pairs
def get_dataset():
    if path_to_dataset_pickle.exists():
        print(f"'{path_to_dataset_pickle}' loaded")
        with open(path_to_dataset_pickle, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    print(f"'{path_to_dataset_pickle}' creation on track")
    total_db = []
    wanted_db = []

    df_real_comments = pd.read_csv(path_to_real_comments)
    df_real_comments["published_at"] = pd.to_datetime(df_real_comments["published_at"])

    for index, row in df_real_comments.iterrows():
        date = row["published_at"]
        text = row["text"]
        if date.year < 2022:
            total_db.append((text, "human"))
        else:
            wanted_db.append((text, "unknown"))

    df_llm_comments = pd.read_csv(path_to_llm_comments)
    llm_comments_list = df_llm_comments["generated_text"].tolist()

    total_db.extend(zip(llm_comments_list, ["llm"] * len(llm_comments_list)))

    dataset = {"total_db": total_db, "wanted_db": wanted_db}

    with open(path_to_dataset_pickle, "wb") as f:
        pickle.dump(dataset, f)
        print(f"'{path_to_dataset_pickle}' dumped")

    return dataset


class TransformerDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text.values
        self.labels = dataframe.label.values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloader(model_name, max_len, batch_size):
    dataset = get_dataset()
    total_db = dataset["total_db"]
    df = pd.DataFrame(total_db, columns=["text", "label"])
    label_map = {"human": 0, "llm": 1}
    df["label"] = df["label"].map(label_map)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TransformerDataset(train_df, tokenizer, max_len)
    val_dataset = TransformerDataset(val_df, tokenizer, max_len)
    test_dataset = TransformerDataset(test_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
