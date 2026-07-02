import sqlite3
import pandas as pd
import numpy as np

DATASET_PATH = r"C:\Users\User\Downloads\Loan_status_2007-2020Q3.csv"

FEATURES = [
    "loan_amnt",
    "last_pymnt_amnt",
    "avg_cur_bal",
    "int_rate",
    "grade",
    "installment",
    "annual_inc",
    "dti",
    "revol_util",
    "revol_bal",
]

TARGET = "loan_status"
DATE_COL = "issue_d"

TOTAL_SAMPLE_SIZE = 20000
RATIO_NON_DEFAULT = 0.95
RATIO_DEFAULT = 0.05
GLOBAL_RANDOM_SEED = 42
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

def load_and_preprocess_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    dtype_dict = {
        "loan_amnt": "float32",
        "last_pymnt_amnt": "float32",
        "avg_cur_bal": "float32",
        "int_rate": "object",
        "grade": "object",
        "installment": "float32",
        "annual_inc": "float32",
        "dti": "float32",
        "revol_util": "object",
        "revol_bal": "float32",
        "loan_status": "object",
        "issue_d": "object",
    }

    df = pd.read_csv(
        path,
        low_memory=True,
        usecols=FEATURES + [TARGET, DATE_COL],
        dtype=dtype_dict,
    )

    print("Filtering and preprocessing...")
    df = df[df[TARGET].isin(["Fully Paid", "Charged Off"])].copy()
    df[TARGET] = df[TARGET].map({"Fully Paid": 0, "Charged Off": 1})
    
    # We leave DATE_COL as string to save nicely in SQLite
    for col in ["int_rate", "revol_util"]:
        df[col] = df[col].replace("%", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    df["grade"] = df["grade"].map(GRADE_MAP)
    df = df.dropna(subset=FEATURES + [TARGET, DATE_COL]).reset_index(drop=True)
    return df

def get_imbalanced_sample_955(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    print("Sampling data...")
    fp = df[df[TARGET] == 0]
    co = df[df[TARGET] == 1]

    n_fp = min(len(fp), int(TOTAL_SAMPLE_SIZE * RATIO_NON_DEFAULT))
    n_co = min(len(co), int(TOTAL_SAMPLE_SIZE * RATIO_DEFAULT))

    fp_sample = fp.sample(n=n_fp, random_state=random_state)
    co_sample = co.sample(n=n_co, random_state=random_state)

    out = pd.concat([fp_sample, co_sample], axis=0)
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out

if __name__ == "__main__":
    clean_df = load_and_preprocess_data(DATASET_PATH)
    sample_df = get_imbalanced_sample_955(clean_df, GLOBAL_RANDOM_SEED)
    
    print("Writing to SQLite database (lending_data.db)...")
    with sqlite3.connect("lending_data.db") as conn:
        sample_df.to_sql("lending_sample", conn, if_exists="replace", index=False)
        
    print("Done! Database lending_data.db created.")
