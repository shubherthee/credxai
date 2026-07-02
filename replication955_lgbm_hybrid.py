import time
import re
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (95:5 + HYBRID XAI)
# ==========================================
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

TOTAL_SAMPLE_SIZE = 20000
RATIO_NON_DEFAULT = 0.95
RATIO_DEFAULT = 0.05
TEST_SIZE = 0.2
GLOBAL_RANDOM_SEED = 42

# 5-fold only (as requested)
N_STABILITY_RUNS = 20

# LIME + SHAP controls
LIME_LOCAL_FEATURES = 10
LIME_EVAL_SAMPLES = 20
LIME_NUM_SAMPLES = 3000

SHAP_BACKGROUND_SIZE = 100
SHAP_EVAL_SAMPLES = 100

# Integrated prefilter controls
INTEGRATED_TOP_K = 5
LIME_SELECTION_FREQ_WEIGHT = 0.75

TREE_MODELS = {"LightGBM"}


# ==========================================
# 2. STABILITY METRICS (SRA + CV)
# ==========================================
def calculate_rankstab(rankings_list):
    n_features = len(rankings_list[0])
    rankstab = {}

    for feat_idx in range(n_features):
        positions = []
        for ranking in rankings_list:
            pos = np.where(ranking == feat_idx)[0]
            if len(pos) > 0:
                positions.append(pos[0] + 1)

        if len(positions) >= 2:
            rankstab[feat_idx] = float(np.var(np.array(positions), ddof=1))
        else:
            rankstab[feat_idx] = 0.0

    return rankstab


def calculate_sra_by_depth(rankings_list):
    n_features = len(rankings_list[0])
    rankstab = calculate_rankstab(rankings_list)

    sra_depth = {}
    for depth in range(1, n_features + 1):
        in_all_top_d = set(range(n_features))
        for ranking in rankings_list:
            in_all_top_d &= set(ranking[:depth])

        if in_all_top_d:
            sra_depth[depth] = float(np.mean([rankstab[i] for i in in_all_top_d]))
        else:
            sra_depth[depth] = np.nan

    return sra_depth


def calculate_value_stab_cv(importance_values_list):
    values = np.array(importance_values_list)
    n_features = values.shape[1]

    cv_per_feature = {}
    for feat_idx in range(n_features):
        feat_vals = values[:, feat_idx]
        mean_v = np.mean(feat_vals)
        std_v = np.std(feat_vals, ddof=1) if len(feat_vals) > 1 else 0.0
        cv_per_feature[feat_idx] = float(std_v / mean_v) if mean_v > 0 else 0.0

    overall_cv = float(np.mean(list(cv_per_feature.values())))
    return cv_per_feature, overall_cv


def compute_stability_summary(importance_vectors):
    rankings = [np.argsort(-v) for v in importance_vectors]
    sra_depth = calculate_sra_by_depth(rankings)
    valid = [v for v in sra_depth.values() if not np.isnan(v)]
    sra_overall = float(np.mean(valid)) if valid else np.nan

    _, cv_overall = calculate_value_stab_cv(importance_vectors)
    return sra_overall, cv_overall


# ==========================================
# 3. DATA LOADING + 95:5 SAMPLING
# ==========================================
def load_and_preprocess(path):
    print(f"{'=' * 80}")
    print("STEP 1: LOAD + PREPROCESS")
    print(f"{'=' * 80}")

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
    }

    df = pd.read_csv(path, low_memory=True, usecols=FEATURES + [TARGET], dtype=dtype_dict)
    print(f"Raw shape: {df.shape}")

    df = df[df[TARGET].isin(["Fully Paid", "Charged Off"])].copy()
    df[TARGET] = df[TARGET].map({"Fully Paid": 0, "Charged Off": 1})

    for col in ["int_rate", "revol_util"]:
        cleaned = df[col].replace("%", "", regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce", downcast="float")

    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade"] = df["grade"].map(grade_map)

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Clean shape: {df.shape}")
    return df


def get_imbalanced_sample_955(df, random_state=42):
    fp = df[df[TARGET] == 0]
    co = df[df[TARGET] == 1]

    n_fp = min(len(fp), int(TOTAL_SAMPLE_SIZE * RATIO_NON_DEFAULT))
    n_co = min(len(co), int(TOTAL_SAMPLE_SIZE * RATIO_DEFAULT))

    fp_sample = fp.sample(n=n_fp, random_state=random_state)
    co_sample = co.sample(n=n_co, random_state=random_state)

    out = pd.concat([fp_sample, co_sample])
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(
        f"95:5 sample created: {len(out)} rows "
        f"({n_fp} Fully Paid / {n_co} Charged Off)"
    )
    return out


def make_balanced_test_for_metrics(X_test, y_test, random_state=42):
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    class_counts = test_df[TARGET].value_counts()
    minority_count = int(class_counts.min())

    sampled = []
    for cls, grp in test_df.groupby(TARGET):
        sampled.append(grp.sample(n=minority_count, random_state=random_state))

    balanced_test = pd.concat(sampled).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_bal = balanced_test[FEATURES]
    y_bal = balanced_test[TARGET]
    return X_bal, y_bal


# ==========================================
# 4. LIGHTGBM MODEL + BASELINE METRICS
# ==========================================
def get_models():
    return {
        "LightGBM": LGBMClassifier(
            random_state=0,
            verbose=-1,
            n_estimators=250,
            max_depth=5,
            learning_rate=0.1,
        ),
    }


def evaluate_models(X_train, X_test, y_train, y_test, models):
    rows = []

    for name, model in models.items():
        print(f"Training baseline model: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rec = recall_score(y_test, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append(
            {
                "Model": name,
                "AUC": roc_auc_score(y_test, y_prob),
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": rec,
                "F1-score": f1_score(y_test, y_pred),
                "G-Mean": float(np.sqrt(rec * specificity)),
                "Time(s)": elapsed,
            }
        )

    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


# ==========================================
# 5. XAI HELPERS (SHAP, LIME, HYBRID)
# ==========================================
def _extract_shap_array(shap_values):
    if isinstance(shap_values, list):
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    vals = np.array(vals)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


def shap_importance(model_name, model, X_train_scaled, X_eval_scaled):
    t0 = time.time()

    bg_n = min(SHAP_BACKGROUND_SIZE, len(X_train_scaled))
    ev_n = min(SHAP_EVAL_SAMPLES, len(X_eval_scaled))
    background = X_train_scaled.iloc[:bg_n]
    eval_data = X_eval_scaled.iloc[:ev_n]

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(eval_data)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(eval_data)

    vals = _extract_shap_array(raw)
    imp = np.abs(vals).mean(axis=0)
    return imp, time.time() - t0


def _parse_lime_feature(raw_name, feature_names):
    for feat in feature_names:
        if re.search(rf"\b{re.escape(feat)}\b", raw_name):
            return feat
    return None


def lime_prefilter_from_eval(model, X_train_scaled, X_eval_scaled, feature_names, random_state):
    t0 = time.time()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled.values,
        feature_names=feature_names,
        class_names=["Fully Paid", "Charged Off"],
        mode="classification",
        random_state=random_state,
    )

    feat_weights = defaultdict(list)
    feat_freq = Counter()

    n_samples = min(LIME_EVAL_SAMPLES, len(X_eval_scaled))
    for i in range(n_samples):
        exp = explainer.explain_instance(
            X_eval_scaled.iloc[i].values,
            model.predict_proba,
            num_features=LIME_LOCAL_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
        )

        for desc, weight in exp.as_list():
            feat = _parse_lime_feature(desc, feature_names)
            if feat is not None:
                feat_weights[feat].append(abs(weight))
                feat_freq[feat] += 1

    vec = np.array([np.mean(feat_weights[f]) if feat_weights[f] else 0.0 for f in feature_names])
    return vec, feat_freq, time.time() - t0


def integrated_hybrid_importance(
    model_name,
    fitted_model,
    X_train_scaled,
    y_train,
    X_eval_scaled,
    feature_names,
    random_state,
):
    """
    Hybrid method (paper-style):
    LIME local explanations on test instances -> frequency aggregation ->
    select top-K -> retrain reduced model -> SHAP on reduced set.
    """
    lime_vec, lime_freq, lime_time = lime_prefilter_from_eval(
        fitted_model,
        X_train_scaled,
        X_eval_scaled,
        feature_names,
        random_state,
    )

    if lime_freq:
        max_freq = max(lime_freq.values()) if lime_freq else 1
        lime_sum = float(np.sum(lime_vec)) + 1e-12
        pre_score = {}
        for feat in feature_names:
            idx = feature_names.index(feat)
            freq_score = lime_freq.get(feat, 0) / max_freq
            mag_score = float(lime_vec[idx] / lime_sum)
            pre_score[feat] = LIME_SELECTION_FREQ_WEIGHT * freq_score + (1.0 - LIME_SELECTION_FREQ_WEIGHT) * mag_score
        ranked = sorted(feature_names, key=lambda f: pre_score[f], reverse=True)
    else:
        ranked = [feature_names[i] for i in np.argsort(-lime_vec)]

    selected = ranked[:INTEGRATED_TOP_K]
    selected_idx = [feature_names.index(f) for f in selected]

    reduced_model = clone(fitted_model)
    reduced_model.fit(X_train_scaled[selected], y_train)

    shap_reduced, shap_time = shap_importance(
        model_name,
        reduced_model,
        X_train_scaled[selected],
        X_eval_scaled[selected],
    )

    full_vec = np.zeros(len(feature_names))
    for i, idx in enumerate(selected_idx):
        full_vec[idx] = shap_reduced[i]

    return full_vec, selected, lime_time + shap_time


# ==========================================
# 6. HYBRID EXPLANATIONS SUMMARY PLOT
# ==========================================
def plot_hybrid_summary(hybrid_importances, feature_names, model_name, ratio_tag, filename):
    """
    Generate a hybrid explanations summary plot showing mean hybrid importance
    per feature with std error bars, averaged across all stability runs.
    """
    imp_array = np.array(hybrid_importances)  # shape: (n_runs, n_features)
    mean_imp = imp_array.mean(axis=0)
    std_imp = imp_array.std(axis=0)

    sorted_idx = np.argsort(mean_imp)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_mean = mean_imp[sorted_idx]
    sorted_std = std_imp[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        sorted_features,
        sorted_mean,
        xerr=sorted_std,
        color="steelblue",
        edgecolor="white",
        capsize=4,
        alpha=0.85,
    )
    ax.set_xlabel("Mean Hybrid Importance (SHAP on LIME-selected features)", fontsize=11)
    ax.set_title(
        f"Hybrid SHAP-LIME Explanations Summary\n{model_name} | {ratio_tag} | {N_STABILITY_RUNS} Runs",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved summary plot: {filename}")


# ==========================================
# 7. STABILITY EXPERIMENT (20 FRESH RUNS)
# ==========================================
def create_model_instance(model_name, random_seed):
    """Match prior stability protocol: create model with run-specific seed and fixed params."""
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.3,
            max_depth=3,
            reg_alpha=4,
            reg_lambda=5,
            min_child_samples=20,
            random_state=random_seed,
            verbose=-1,
        )
    return clone(get_models()[model_name])


# ==========================================
# 8. MAIN
# ==========================================
if __name__ == "__main__":
    start_all = time.time()

    print(
        f"Running 95:5 hybrid (LightGBM only) with {N_STABILITY_RUNS} fresh runs | K={INTEGRATED_TOP_K}, "
        f"LIME_LOCAL_FEATURES={LIME_LOCAL_FEATURES}, LIME_NUM_SAMPLES={LIME_NUM_SAMPLES}"
    )

    df = load_and_preprocess(DATASET_PATH)
    df_imb = get_imbalanced_sample_955(df, random_state=GLOBAL_RANDOM_SEED)

    # Baseline metrics on balanced test subset (same spirit as existing 95:5 workflow)
    X = df_imb[FEATURES]
    y = df_imb[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=GLOBAL_RANDOM_SEED,
        stratify=y,
    )

    X_test_bal, y_test_bal = make_balanced_test_for_metrics(X_test, y_test, random_state=GLOBAL_RANDOM_SEED)

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_bal_scaled = pd.DataFrame(scaler.transform(X_test_bal), columns=FEATURES)

    models = get_models()
    baseline_df = evaluate_models(X_train_scaled, X_test_bal_scaled, y_train, y_test_bal, models)
    baseline_df.to_csv("Table6_955_Baseline.csv", index=False)

    print(f"\n{'=' * 80}")
    print("BASELINE PERFORMANCE (95:5 train, balanced test)")
    print(f"{'=' * 80}")
    print(baseline_df.to_string(index=False))
    print("Saved: Table6_955_Baseline.csv")

    model_name = "LightGBM"
    print(
        f"\nComputing HYBRID stability for {model_name} over {N_STABILITY_RUNS} runs "
        f"(fresh 95:5 sample each run)..."
    )

    stability_rows = []
    selected_rows = []

    hybrid_importances = []
    hybrid_times = []
    selected_counter = Counter()

    for run in range(N_STABILITY_RUNS):
        print(f"  Run {run + 1}/{N_STABILITY_RUNS}...")

        fp_full = df[df[TARGET] == 0]
        co_full = df[df[TARGET] == 1]

        fp_run = fp_full.sample(n=min(19000, len(fp_full)), random_state=100 + run)
        co_run = co_full.sample(n=min(1000, len(co_full)), random_state=200 + run)

        df_run = pd.concat([fp_run, co_run]).sample(frac=1, random_state=run).reset_index(drop=True)

        X_run = df_run[FEATURES]
        y_run = df_run[TARGET]

        X_train_run, X_test_run, y_train_run, _ = train_test_split(
            X_run, y_run, test_size=0.2, random_state=run, stratify=y_run
        )

        scaler_run = MinMaxScaler()
        X_train_run_scaled = pd.DataFrame(scaler_run.fit_transform(X_train_run), columns=FEATURES)
        X_test_run_scaled = pd.DataFrame(scaler_run.transform(X_test_run), columns=FEATURES)

        model_run = create_model_instance(model_name, run)
        model_run.fit(X_train_run_scaled, y_train_run)

        test_samples = X_test_run_scaled.iloc[:100]
        h_vec, selected, h_time = integrated_hybrid_importance(
            model_name,
            model_run,
            X_train_run_scaled,
            y_train_run,
            test_samples,
            FEATURES,
            random_state=GLOBAL_RANDOM_SEED + run,
        )

        hybrid_importances.append(h_vec)
        hybrid_times.append(h_time)
        selected_counter.update(selected)

    hybrid_rankings = [np.argsort(-imp) for imp in hybrid_importances]
    hybrid_sra_by_depth = calculate_sra_by_depth(hybrid_rankings)
    valid_sras = [v for v in hybrid_sra_by_depth.values() if not np.isnan(v)]
    hybrid_sra_overall = float(np.mean(valid_sras)) if valid_sras else np.nan
    _, hybrid_cv = calculate_value_stab_cv(hybrid_importances)

    print(f"\n  DEBUG {model_name} Hybrid Importances (first 3 runs):")
    for i in range(min(3, len(hybrid_importances))):
        print(f"    Run {i + 1}: {np.round(hybrid_importances[i], 4)}")
        print(f"    Ranking: {hybrid_rankings[i]}")

    print(f"\n{model_name.upper()} HYBRID STABILITY METRICS:")
    print("  SRA by Depth:")
    for depth in sorted(hybrid_sra_by_depth.keys()):
        print(f"    Depth {depth}: {hybrid_sra_by_depth[depth]:.4f}")
    print(f"  SRA Overall: {hybrid_sra_overall:.4f} (lower = more stable rankings)")
    print(f"  ValueStab (CV) Overall: {hybrid_cv:.4f} (lower = more stable importance values)")

    mean_imp = np.mean(np.array(hybrid_importances), axis=0)
    top5_idx = np.argsort(-mean_imp)[:5]
    top5_features = [FEATURES[i] for i in top5_idx]

    stability_rows.append(
        {
            "Model": model_name,
            "Method": "Hybrid",
            "Runs": N_STABILITY_RUNS,
            "SRA_Overall": hybrid_sra_overall,
            "CV_Overall": hybrid_cv,
            "Avg_Hybrid_Time(s)": float(np.mean(hybrid_times)),
            "Top5_Features": ", ".join(top5_features),
        }
    )

    for feat in FEATURES:
        selected_rows.append(
            {
                "Model": model_name,
                "Feature": feat,
                "Selected_Count": selected_counter.get(feat, 0),
                "Selected_Rate": selected_counter.get(feat, 0) / N_STABILITY_RUNS,
            }
        )

    # Generate hybrid explanations summary plot
    plot_hybrid_summary(
        hybrid_importances,
        FEATURES,
        model_name,
        "95:5",
        "Hybrid_Summary_Plot_955_LightGBM.png",
    )

    stability_df = pd.DataFrame(stability_rows)
    selected_df = pd.DataFrame(selected_rows)

    stability_df.to_csv("Hybrid_XAI_Stability_955.csv", index=False)
    selected_df.to_csv("Hybrid_XAI_LIME_Prefilter_Frequency_955.csv", index=False)

    print(f"\n{'=' * 80}")
    print("HYBRID SHAP-LIME RESULTS (95:5)")
    print(f"{'=' * 80}")
    print("Stability summary:")
    print(stability_df.to_string(index=False))

    print("\nSaved files:")
    print("  - Table6_955_Baseline.csv")
    print("  - Hybrid_XAI_Stability_955.csv")
    print("  - Hybrid_XAI_LIME_Prefilter_Frequency_955.csv")
    print("  - Hybrid_Summary_Plot_955_LightGBM.png")

    print(f"\nTotal runtime: {time.time() - start_all:.2f}s")
