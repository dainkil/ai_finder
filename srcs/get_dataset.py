import pandas as pd
from pathlib import Path
import pickle

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
