import os
import re
import json
import requests
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
pd.set_option("styler.render.max_elements", 1000000)

import shap
import lime.lime_tabular

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier

from components import (
    load_css,
    bank_header,
    page_heading,
    metric_card,
    report_section,
    decision_banner,
    comparison_card,
)

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="CredXAI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css("styles.css")
bank_header()

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "lending_data.db"

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
TEST_SIZE = 0.2
GLOBAL_RANDOM_SEED = 42

LIME_LOCAL_FEATURES = 10
LIME_NUM_SAMPLES = 3000
INTEGRATED_TOP_K = 5
LIME_SELECTION_FREQ_WEIGHT = 0.75

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
GRADE_REVERSE = {v: k for k, v in GRADE_MAP.items()}

FEATURE_LABELS = {
    "loan_amnt": "Loan Amount",
    "last_pymnt_amnt": "Last Payment Amount",
    "avg_cur_bal": "Average Current Balance",
    "int_rate": "Interest Rate",
    "grade": "Credit Grade",
    "installment": "Monthly Installment",
    "annual_inc": "Annual Income",
    "dti": "Debt-to-Income Ratio",
    "revol_util": "Revolving Utilization",
    "revol_bal": "Revolving Balance",
}

FEATURE_INFO = {
    "loan_amnt": {
        "description": "The total loan amount requested by the applicant.",
        "good": "Lower loan amounts relative to annual income are usually easier to repay.",
        "bad": "A very high loan amount may increase repayment burden.",
        "example": "Example: RM 15,000 requested loan."
    },
    "last_pymnt_amnt": {
        "description": "The most recent payment made by the borrower.",
        "good": "A higher recent payment can suggest stronger repayment behaviour.",
        "bad": "A very low recent payment may suggest weaker repayment capacity.",
        "example": "Example: RM 300 last payment."
    },
    "avg_cur_bal": {
        "description": "The average current balance across the applicant’s accounts.",
        "good": "Higher account balances may suggest stronger financial stability.",
        "bad": "Very low balances may suggest limited financial reserves.",
        "example": "Example: RM 12,000 average balance."
    },
    "int_rate": {
        "description": "The annual interest rate charged on the loan.",
        "good": "Lower interest rates are usually linked to lower-risk borrowers.",
        "bad": "Higher interest rates are often linked to higher-risk borrowers.",
        "example": "Example: 13.5% annual interest."
    },
    "grade": {
        "description": "The borrower credit grade from A to G. This grade comes from the original LendingClub dataset.",
        "good": "A = Excellent / very low risk. B = Strong / low risk. C = Acceptable / moderate risk.",
        "bad": "D = Medium risk. E = High risk. F = Very high risk. G = Highest default risk.",
        "example": "Model mapping: A=1, B=2, C=3, D=4, E=5, F=6, G=7."
    },
    "installment": {
        "description": "The required monthly loan payment.",
        "good": "A monthly payment that is affordable compared with income is safer.",
        "bad": "A high monthly payment may create repayment stress.",
        "example": "Example: RM 450 monthly installment."
    },
    "annual_inc": {
        "description": "The applicant’s yearly income.",
        "good": "Higher stable income supports repayment ability.",
        "bad": "Lower income may make the requested loan harder to repay.",
        "example": "Example: RM 60,000 annual income."
    },
    "dti": {
        "description": "Debt-to-Income Ratio. It shows how much of income is already used for debt payments.",
        "good": "0–20% = strong. 20–35% = acceptable.",
        "bad": "35–50% = high debt burden. Above 50% = very risky.",
        "example": "Example: 15% DTI means 15% of income is used for debt."
    },
    "revol_util": {
        "description": "Revolving credit utilization. It shows how much credit limit is already being used.",
        "good": "0–30% = healthy usage. 30–50% = moderate usage.",
        "bad": "50–75% = high usage. Above 75% = very risky.",
        "example": "Example: 45% utilization means 45% of available credit is used."
    },
    "revol_bal": {
        "description": "The current outstanding revolving credit balance.",
        "good": "Lower balances are generally safer.",
        "bad": "Large balances may suggest higher debt pressure.",
        "example": "Example: RM 15,000 revolving balance."
    },
}

BANK_RED = "#B42318"
BANK_GREEN = "#157347"
BANK_ORANGE = "#B7791F"
BANK_GREY = "#C8C8C8"
BANK_INK = "#081A33"
BANK_SECTION = "#F8FAFC"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku:beta")


# ============================================================
# SMALL HELPERS
# ============================================================

def feature_help(feature_key: str) -> str:
    item = FEATURE_INFO[feature_key]
    return f"""
{item['description']}

Good / Safer:
{item['good']}

Risk / Concern:
{item['bad']}

{item['example']}
"""


def expected_decision_from_actual(actual_status: str) -> str:
    return "Approved" if actual_status == "Fully Paid" else "Rejected"


def normalize_decision(decision: str) -> str:
    return "Approved" if decision.upper() == "APPROVED" else "Rejected"


def compute_prediction_match(actual_status: str, decision: str) -> str:
    expected = expected_decision_from_actual(actual_status)
    predicted = normalize_decision(decision)
    return "Correct" if expected == predicted else "Incorrect"


def format_feature_value(feat: str, value: float) -> str:
    if feat == "grade":
        return GRADE_REVERSE.get(int(value), "?")
    if feat in ["loan_amnt", "last_pymnt_amnt", "avg_cur_bal", "installment", "annual_inc", "revol_bal"]:
        return f"RM {value:,.0f}"
    if feat in ["int_rate", "dti", "revol_util"]:
        return f"{value:.1f}%"
    return f"{value:.2f}"


def calculate_repayment_years(input_data: dict) -> tuple[float, float]:
    installment = float(input_data["installment"])
    loan_amount = float(input_data["loan_amnt"])
    if installment <= 0:
        return 0.0, 0.0
    years = max(round(loan_amount / (installment * 12), 1), 0.1)
    total = installment * years * 12
    return years, total


# ============================================================
# DATA LOADING + MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner=False)
def build_and_train_model(db_path: str):
    import sqlite3
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}. Please run init_db.py first.")
    
    with sqlite3.connect(db_path) as conn:
        sample_df = pd.read_sql("SELECT * FROM lending_sample", conn)
        
    sample_df[DATE_COL] = pd.to_datetime(sample_df[DATE_COL], errors="coerce")

    X = sample_df[FEATURES]
    y = sample_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=GLOBAL_RANDOM_SEED,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.3,
        max_depth=3,
        reg_alpha=4,
        reg_lambda=5,
        min_child_samples=20,
        random_state=GLOBAL_RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_train_scaled, y_train)

    dataset_info = {
        "source": db_path,
        "clean_rows": 20000,
        "clean_fully_paid": int((sample_df[TARGET] == 0).sum()),
        "clean_charged_off": int((sample_df[TARGET] == 1).sum()),
        "sample_rows": len(sample_df),
        "sample_fully_paid": int((sample_df[TARGET] == 0).sum()),
        "sample_charged_off": int((sample_df[TARGET] == 1).sum()),
        "min_date": sample_df[DATE_COL].min(),
        "max_date": sample_df[DATE_COL].max(),
    }

    return model, scaler, X_train_scaled, y_train, sample_df, dataset_info


def get_primary_reason_from_row(row: pd.Series, decision: str) -> str:
    grade_num = int(row["grade"])

    if decision == "Rejected":
        rules = [
            ("Low credit grade", grade_num >= 5),
            ("High debt-to-income ratio", row["dti"] > 25),
            ("High revolving utilization", row["revol_util"] > 75),
            ("Low income relative to loan amount", row["annual_inc"] < 40000),
            ("High interest rate", row["int_rate"] > 20),
            ("High revolving balance", row["revol_bal"] > 30000),
        ]
    else:
        rules = [
            ("Good credit grade", grade_num <= 3),
            ("Low debt-to-income ratio", row["dti"] <= 20),
            ("Moderate revolving utilization", row["revol_util"] <= 50),
            ("Stable annual income", row["annual_inc"] >= 60000),
            ("Lower interest rate", row["int_rate"] <= 15),
            ("Recent payment activity", row["last_pymnt_amnt"] > 500),
        ]

    for reason, condition in rules:
        if condition:
            return reason

    return "Model-estimated default risk" if decision == "Rejected" else "Model-estimated acceptable risk"


@st.cache_data(show_spinner=False)
def generate_decision_log_from_csv_sample_cached(sample_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # This wrapper is not used because model and scaler are not hashable.
    return sample_df


def generate_decision_log_from_csv_sample(sample_df: pd.DataFrame, model, scaler, threshold: float) -> pd.DataFrame:
    dates = pd.to_datetime(sample_df[DATE_COL], errors="coerce")

    if dates.isna().any():
        raise ValueError("Some issue_d values could not be parsed. No synthetic dates are allowed.")

    X_sample = sample_df[FEATURES]
    X_sample_scaled = pd.DataFrame(scaler.transform(X_sample), columns=FEATURES)

    risk_scores = model.predict_proba(X_sample_scaled)[:, 1]
    decisions = np.where(risk_scores <= threshold, "Approved", "Rejected")

    rows = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        decision = decisions[i]

        rows.append(
            {
                "date": dates.iloc[i],
                "applicant_id": f"CSV-{i + 1:06d}",
                "loan_amnt": float(row["loan_amnt"]),
                "last_pymnt_amnt": float(row["last_pymnt_amnt"]),
                "avg_cur_bal": float(row["avg_cur_bal"]),
                "int_rate": float(row["int_rate"]),
                "grade": GRADE_REVERSE.get(int(row["grade"]), "?"),
                "installment": float(row["installment"]),
                "annual_inc": float(row["annual_inc"]),
                "dti": float(row["dti"]),
                "revol_util": float(row["revol_util"]),
                "revol_bal": float(row["revol_bal"]),
                "actual_status": "Fully Paid" if int(row[TARGET]) == 0 else "Charged Off",
                "risk_score": float(risk_scores[i]),
                "decision": decision,
                "primary_reason": get_primary_reason_from_row(row, decision),
            }
        )

    out = pd.DataFrame(rows)
    out["prediction_match"] = out.apply(
        lambda r: compute_prediction_match(r["actual_status"], "APPROVED" if r["decision"] == "Approved" else "REJECTED"),
        axis=1,
    )
    return out


# ============================================================
# HYBRID XAI
# ============================================================

def _parse_lime_feature(raw_name: str):
    for feat in FEATURES:
        if re.search(rf"\b{re.escape(feat)}\b", raw_name):
            return feat
    return None


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


def run_hybrid_xai(model, X_train_scaled: pd.DataFrame, y_train, instance_scaled: pd.DataFrame):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled.values,
        feature_names=FEATURES,
        class_names=["Fully Paid", "Charged Off"],
        mode="classification",
        random_state=GLOBAL_RANDOM_SEED,
    )

    exp = lime_explainer.explain_instance(
        instance_scaled.values[0],
        model.predict_proba,
        num_features=LIME_LOCAL_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )

    feat_weights = defaultdict(list)
    feat_freq = Counter()

    for raw_name, weight in exp.as_list():
        feat = _parse_lime_feature(raw_name)
        if feat:
            feat_weights[feat].append(abs(weight))
            feat_freq[feat] += 1

    lime_vec = np.array(
        [float(np.mean(feat_weights[f])) if feat_weights[f] else 0.0 for f in FEATURES]
    )

    if feat_freq:
        max_freq = max(feat_freq.values())
        lime_sum = float(np.sum(lime_vec)) + 1e-12
        pre_score = {}
        for i, feat in enumerate(FEATURES):
            freq_score = feat_freq.get(feat, 0) / max_freq
            mag_score = lime_vec[i] / lime_sum
            pre_score[feat] = (
                LIME_SELECTION_FREQ_WEIGHT * freq_score
                + (1.0 - LIME_SELECTION_FREQ_WEIGHT) * mag_score
            )
        ranked = sorted(FEATURES, key=lambda f: pre_score[f], reverse=True)
    else:
        ranked = [FEATURES[i] for i in np.argsort(-lime_vec)]
        pre_score = {f: float(lime_vec[FEATURES.index(f)]) for f in FEATURES}

    selected = ranked[:INTEGRATED_TOP_K]
    selected_idx = [FEATURES.index(f) for f in selected]

    reduced_model = clone(model)
    reduced_model.fit(X_train_scaled[selected], y_train)

    shap_explainer = shap.TreeExplainer(reduced_model)
    raw_shap = shap_explainer.shap_values(instance_scaled[selected])
    shap_reduced = _extract_shap_array(raw_shap)[0]

    hybrid_vec = np.zeros(len(FEATURES))
    for i, idx in enumerate(selected_idx):
        hybrid_vec[idx] = shap_reduced[i]

    return hybrid_vec, selected, pre_score


def run_single_assessment(input_data: dict, threshold: float, model, scaler, X_train_scaled, y_train):
    instance_df = pd.DataFrame([input_data])[FEATURES]
    instance_scaled = pd.DataFrame(scaler.transform(instance_df), columns=FEATURES)

    risk_score = float(model.predict_proba(instance_scaled)[0][1])
    hybrid_vec, selected_features, lime_scores = run_hybrid_xai(model, X_train_scaled, y_train, instance_scaled)
    decision = "REJECTED" if risk_score > threshold else "APPROVED"

    return {
        "risk_score": risk_score,
        "hybrid_vec": hybrid_vec,
        "selected_features": selected_features,
        "lime_scores": lime_scores,
        "decision": decision,
    }


# ============================================================
# OPENROUTER EXPLANATIONS
# ============================================================

def call_openrouter_api(prompt: str, model: str = OPENROUTER_MODEL) -> str:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key is not configured. Add OPENROUTER_API_KEY to your .env file to enable AI-generated explanations."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1100,
        "temperature": 0.55,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenRouter explanation unavailable: {str(e)}"


def generate_bank_explanation(decision, risk_score, threshold, hybrid_vec, selected_features, input_data):
    hybrid_series = pd.Series(hybrid_vec, index=FEATURES)
    selected_vals = hybrid_series[selected_features]
    risk_feats = selected_vals[selected_vals > 0].sort_values(ascending=False)
    safe_feats = selected_vals[selected_vals < 0].sort_values()

    risk_drivers = [
        f"{FEATURE_LABELS[feat]}: {format_feature_value(feat, input_data[feat])}, contribution {val:+.4f}"
        for feat, val in risk_feats.items()
    ]
    mitigating_factors = [
        f"{FEATURE_LABELS[feat]}: {format_feature_value(feat, input_data[feat])}, contribution {val:+.4f}"
        for feat, val in safe_feats.items()
    ]

    repayment_years, total_repayment = calculate_repayment_years(input_data)

    prompt = f"""
Act as a professional internal bank credit analyst writing an explanation for a loan application.

Decision: {decision}
Risk Score: {risk_score:.1%}
Threshold: {threshold:.0%}

Loan Terms:
- Loan Amount: RM {input_data['loan_amnt']:,.0f}
- Monthly Installment: RM {input_data['installment']:,.0f}
- Interest Rate: {input_data['int_rate']:.1f}%
- Estimated Term: {repayment_years:.1f} years
- Estimated Total Repayment: RM {total_repayment:,.0f}

Applicant Profile:
- Annual Income: RM {input_data['annual_inc']:,.0f}
- Credit Grade: {GRADE_REVERSE.get(int(input_data['grade']), '?')}
- DTI: {input_data['dti']:.1f}%
- Revolving Utilization: {input_data['revol_util']:.1f}%
- Revolving Balance: RM {input_data['revol_bal']:,.0f}

Risk-Increasing Factors:
{chr(10).join(risk_drivers) if risk_drivers else "None identified"}

Risk-Reducing Factors:
{chr(10).join(mitigating_factors) if mitigating_factors else "None identified"}

Write using these headings:
1. Credit Decision Summary
2. Loan Affordability Review
3. Key Risk Drivers
4. Mitigating Factors
5. Recommended Action

Rules:
- Use formal banking language. Mention model-supported evidence, but do not mention SHAP, LIME, LightGBM, or technical contribution values directly.
- DO NOT output conversational filler like "Here is the explanation...". Output ONLY the report starting with the first heading.
"""
    return call_openrouter_api(prompt)


def generate_customer_explanation(decision, risk_score, threshold, hybrid_vec, selected_features, input_data):
    hybrid_series = pd.Series(hybrid_vec, index=FEATURES)
    selected_vals = hybrid_series[selected_features]
    risk_feats = selected_vals[selected_vals > 0].sort_values(ascending=False)
    safe_feats = selected_vals[selected_vals < 0].sort_values()

    concern_factors = [
        f"{FEATURE_LABELS[feat]}: {format_feature_value(feat, input_data[feat])}"
        for feat in risk_feats.index
    ]
    positive_factors = [
        f"{FEATURE_LABELS[feat]}: {format_feature_value(feat, input_data[feat])}"
        for feat in safe_feats.index
    ]

    repayment_years, total_repayment = calculate_repayment_years(input_data)

    prompt = f"""
Generate a customer-friendly loan decision explanation.

Start with "Dear Applicant," and end with "Warm regards, Bank Team."

Decision: {decision}
Risk Score: {risk_score:.1%}

Loan Details:
- Loan Amount: RM {input_data['loan_amnt']:,.0f}
- Monthly Payment: RM {input_data['installment']:,.0f}
- Interest Rate: {input_data['int_rate']:.1f}%
- Estimated Repayment Term: {repayment_years:.1f} years
- Estimated Total Repayment: RM {total_repayment:,.0f}

Helpful Factors:
{chr(10).join(positive_factors) if positive_factors else "Overall acceptable profile"}

Concern Factors:
{chr(10).join(concern_factors) if concern_factors else "No major concern factors identified"}

Rules:
- Use simple language.
- Do not mention AI, model, SHAP, LIME, LightGBM, feature importance, or contribution values.
- If approved, explain what helped and what happens next.
- If declined, explain concerns gently and give 3 practical improvement tips.
- Include monthly payment, interest rate, estimated term, and affordability meaning.
"""
    return call_openrouter_api(prompt)


# ============================================================
# PLOTS
# ============================================================

def _bank_style(fig, ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D8E1EC")
    ax.spines["bottom"].set_color("#D8E1EC")
    ax.tick_params(colors="#667085", labelsize=8)
    fig.patch.set_facecolor(BANK_SECTION)
    ax.set_facecolor(BANK_SECTION)


def plot_risk_gauge(risk_score: float):
    fig, ax = plt.subplots(figsize=(2.6, 1.35), subplot_kw=dict(aspect="equal"))

    thresholds = [0.0, 0.30, 0.70, 1.0]
    colors = [BANK_GREEN, BANK_ORANGE, BANK_RED]
    labels = ["Low\n0–30%", "Medium\n30–70%", "High\n70–100%"]

    for i in range(3):
        t0 = np.pi * (1 - thresholds[i])
        t1 = np.pi * (1 - thresholds[i + 1])
        theta = np.linspace(t0, t1, 80)

        ax.plot(np.cos(theta), np.sin(theta), lw=11, color=colors[i], solid_capstyle="butt", alpha=0.94)

        mid = (thresholds[i] + thresholds[i + 1]) / 2
        angle = np.pi * (1 - mid)
        ax.text(
            np.cos(angle) * 0.80,
            np.sin(angle) * 0.80,
            labels[i],
            ha="center",
            va="center",
            fontsize=5.2,
            color="white",
            fontweight="bold",
        )

    clipped_score = min(max(risk_score, 0.0), 1.0)
    needle_angle = np.pi * (1 - clipped_score)

    ax.annotate(
        "",
        xy=(0.62 * np.cos(needle_angle), 0.62 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=BANK_INK, lw=2.2),
    )

    ax.plot(0, 0, "o", color=BANK_INK, markersize=7)
    ax.text(0, -0.17, f"{risk_score:.1%}", ha="center", va="center", fontsize=12, fontweight="bold", color=BANK_INK)
    ax.text(0, -0.31, "Default Risk Score", ha="center", va="center", fontsize=6, color="#667085")

    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-0.42, 1.02)
    ax.axis("off")
    fig.patch.set_facecolor(BANK_SECTION)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig


def plot_hybrid_importance(hybrid_vec, selected_features):
    fig, ax = plt.subplots(figsize=(9.5, 4.25))

    all_idx = np.argsort(np.abs(hybrid_vec))
    feats = [FEATURES[i] for i in all_idx]
    vals = hybrid_vec[all_idx]

    colors = []
    for f, v in zip(feats, vals):
        if f in selected_features:
            colors.append(BANK_RED if v > 0 else BANK_GREEN)
        else:
            colors.append(BANK_GREY)

    bars = ax.barh(range(len(feats)), vals, color=colors, edgecolor="white", height=0.58)

    for i, (v, feat) in enumerate(zip(vals, feats)):
        marker = " ◆" if feat in selected_features else ""
        label = f"{v:+.4f}{marker}"
        x_pos = v + (0.0006 if v >= 0 else -0.0006)
        ha = "left" if v >= 0 else "right"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=7.2, color=BANK_INK)

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEATURE_LABELS[f] for f in feats], fontsize=8.1)
    ax.axvline(0, color=BANK_INK, linewidth=0.8)
    ax.set_xlabel("Signed hybrid contribution", fontsize=8.5)
    ax.set_title(
        f"Hybrid SHAP-LIME Importance · Top-{INTEGRATED_TOP_K} LIME-selected features marked ◆",
        fontsize=10,
        fontweight="bold",
        color=BANK_INK,
    )

    red_p = mpatches.Patch(color=BANK_RED, label="Raises default risk")
    green_p = mpatches.Patch(color=BANK_GREEN, label="Lowers default risk")
    grey_p = mpatches.Patch(color=BANK_GREY, label="Not selected")
    ax.legend(handles=[red_p, green_p, grey_p], loc="lower right", fontsize=7.5, framealpha=0.9)

    _bank_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_report_bar(df_filtered: pd.DataFrame, period: str):
    grouped = df_filtered.groupby([period, "decision"]).size().unstack(fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    x = np.arange(len(grouped))
    width = 0.36

    approved = grouped.get("Approved", pd.Series(0, index=grouped.index))
    rejected = grouped.get("Rejected", pd.Series(0, index=grouped.index))

    ax.bar(x - width / 2, approved, width, label="Approved", color=BANK_GREEN, edgecolor="white")
    ax.bar(x + width / 2, rejected, width, label="Rejected", color=BANK_RED, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(grouped[period], rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Applications", fontsize=8.5)
    ax.set_title(f"Credit Decisions by {period.title()}", fontsize=10, fontweight="bold", color=BANK_INK)
    ax.legend(fontsize=8)
    _bank_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_reason_breakdown(df_filtered: pd.DataFrame, decision_filter: str):
    sub = df_filtered[df_filtered["decision"] == decision_filter]
    counts = sub["primary_reason"].value_counts().head(8)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    if counts.empty:
        ax.text(0.5, 0.5, f"No {decision_filter} records", ha="center", va="center")
        ax.axis("off")
        return fig

    color = BANK_GREEN if decision_filter == "Approved" else BANK_RED
    ax.barh(counts.index[::-1], counts.values[::-1], color=color, edgecolor="white", height=0.55)

    for i, val in enumerate(counts.values[::-1]):
        ax.text(val + max(counts.values) * 0.01, i, str(val), va="center", fontsize=7.5)

    ax.set_xlabel("Applications", fontsize=8.5)
    ax.set_title(f"Primary Reasons — {decision_filter}", fontsize=10, fontweight="bold", color=BANK_INK)
    _bank_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_grade_distribution(df_filtered: pd.DataFrame):
    grade_counts = df_filtered.groupby(["grade", "decision"]).size().unstack(fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(9.2, 3.0))

    if grade_counts.empty:
        ax.text(0.5, 0.5, "No records", ha="center", va="center")
        ax.axis("off")
        return fig

    grades = grade_counts["grade"].tolist()
    x = np.arange(len(grades))
    width = 0.36

    approved = grade_counts.get("Approved", pd.Series(0, index=grade_counts.index))
    rejected = grade_counts.get("Rejected", pd.Series(0, index=grade_counts.index))

    ax.bar(x - width / 2, approved, width, label="Approved", color=BANK_GREEN, edgecolor="white")
    ax.bar(x + width / 2, rejected, width, label="Rejected", color=BANK_RED, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(grades, fontsize=8.5)
    ax.set_xlabel("Credit Grade", fontsize=8.5)
    ax.set_ylabel("Applications", fontsize=8.5)
    ax.set_title("Decision Distribution by Credit Grade", fontsize=10, fontweight="bold", color=BANK_INK)
    ax.legend(fontsize=8)
    _bank_style(fig, ax)
    plt.tight_layout()
    return fig


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

with st.sidebar:
    st.markdown("### Decision Threshold")
    threshold = st.slider("Risk Score Cutoff", 0.1, 0.9, 0.5, 0.05)

    try:
        with st.spinner("Loading SQLite database and training model..."):
            model, scaler, X_train_scaled, y_train, sample_df, dataset_info = build_and_train_model(DB_PATH)
            decision_log = generate_decision_log_from_csv_sample(sample_df, model, scaler, threshold)

        st.success("CredXAI model ready")
        st.info(f"Source: {os.path.basename(dataset_info['source'])}")
        st.info(f"Clean rows: {dataset_info['clean_rows']:,}")
        st.info(f"Sample rows: {dataset_info['sample_rows']:,}")
        st.info(f"Fully Paid: {dataset_info['sample_fully_paid']:,}")
        st.info(f"Charged Off: {dataset_info['sample_charged_off']:,}")
        st.info(
            f"Date range: {pd.to_datetime(dataset_info['min_date']).strftime('%Y-%m')} "
            f"to {pd.to_datetime(dataset_info['max_date']).strftime('%Y-%m')}"
        )
        if OPENROUTER_API_KEY:
            st.success("OpenRouter enabled")
        else:
            st.warning("OpenRouter key missing")

    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown("---")
    st.markdown("### Model Configuration")
    st.markdown(
        """
- **Algorithm**: LightGBM
- **Data Source**: SQLite Database
- **Target**: loan_status
- **Classes**: Fully Paid / Charged Off
- **Training Sample Ratio**: 95:5
- **XAI**: Hybrid LIME + SHAP
        """
    )


# ============================================================
# NAVIGATION
# ============================================================

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Loan History Analyzer"

nav_options = ["Loan History Analyzer", "Loan Assessment Simulation", "Reports & Analytics"]

st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
current_page = st.radio(
    "Navigation",
    nav_options,
    key="nav_page",
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 1 — LOAN HISTORY ANALYZER
# ============================================================

if current_page == "Loan History Analyzer":
    page_heading(
        "Loan History Analyzer",
        "Historical Data vs XAI Validation",
        "Review real historical loan records and analyze any row against the LightGBM + Hybrid XAI decision engine.",
    )

    history_df = decision_log.copy()
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df["year"] = history_df["date"].dt.year.astype(str)
    history_df["month"] = history_df["date"].dt.month
    history_df["month_name"] = history_df["date"].dt.strftime("%B")

    total_loans = len(history_df)
    fully_paid = int((history_df["actual_status"] == "Fully Paid").sum())
    charged_off = int((history_df["actual_status"] == "Charged Off").sum())
    avg_predicted_risk = history_df["risk_score"].mean() if total_loans else 0

    # Overview first: gives judges immediate understanding of the dataset.
    with st.container(border=True):
        report_section("Dataset Overview", "CSV-derived 95:5 historical sample used by CredXAI")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card(f"{total_loans:,}", "Total Historical Loans", "blue")
        with k2:
            metric_card(f"{fully_paid:,}", "Fully Paid", "green")
        with k3:
            metric_card(f"{charged_off:,}", "Charged Off", "red")
        with k4:
            metric_card(f"{avg_predicted_risk:.1%}", "Avg Predicted Risk", "amber")

    # If a row has been analyzed, show the historical-vs-XAI result on the same page.
    selected_history = st.session_state.get("selected_record_for_history")
    if selected_history:
        selected_input = {
            "loan_amnt": float(selected_history["loan_amnt"]),
            "last_pymnt_amnt": float(selected_history["last_pymnt_amnt"]),
            "avg_cur_bal": float(selected_history["avg_cur_bal"]),
            "int_rate": float(selected_history["int_rate"]),
            "grade": float(GRADE_MAP[selected_history["grade"]]),
            "installment": float(selected_history["installment"]),
            "annual_inc": float(selected_history["annual_inc"]),
            "dti": float(selected_history["dti"]),
            "revol_util": float(selected_history["revol_util"]),
            "revol_bal": float(selected_history["revol_bal"]),
        }
        selected_instance = pd.DataFrame([selected_input])[FEATURES]
        selected_scaled = pd.DataFrame(scaler.transform(selected_instance), columns=FEATURES)

        with st.spinner("Running Hybrid XAI validation for selected historical record..."):
            replay_risk_score = float(model.predict_proba(selected_scaled)[0][1])
            replay_hybrid_vec, replay_selected_features, replay_lime_scores = run_hybrid_xai(
                model, X_train_scaled, y_train, selected_scaled
            )
            replay_decision = "REJECTED" if replay_risk_score > threshold else "APPROVED"

        with st.container(border=True):
            report_section("Selected Loan XAI Analysis", "Historical outcome compared against the live CredXAI model result")
            comparison_card(selected_history["actual_status"], replay_decision)

            a1, a2, a3, a4, a5 = st.columns(5)
            with a1:
                metric_card(selected_history["applicant_id"], "Record ID", "neutral")
            with a2:
                metric_card(f"{replay_risk_score:.1%}", "XAI Risk Score", "amber")
            with a3:
                metric_card(selected_history["grade"], "Credit Grade", "blue")
            with a4:
                metric_card(f"RM {selected_history['loan_amnt']:,.0f}", "Loan Amount", "neutral")
            with a5:
                metric_card(selected_history["primary_reason"], "Primary Reason", "neutral")

            st.markdown("#### Key Hybrid XAI Factors")
            factor_col1, factor_col2 = st.columns(2, gap="large")
            replay_series = pd.Series(replay_hybrid_vec, index=FEATURES)
            replay_vals = replay_series[replay_selected_features]
            risk_factors = replay_vals[replay_vals > 0].sort_values(ascending=False)
            safe_factors = replay_vals[replay_vals < 0].sort_values()

            with factor_col1:
                st.markdown("**Raising Default Risk**")
                if risk_factors.empty:
                    st.info("No dominant risk-raising factor in the selected local explanation.")
                for feat, val in risk_factors.items():
                    st.markdown(
                        f"""
                        <div class="factor-risk">
                            <strong>{FEATURE_LABELS[feat]}</strong><br>
                            Value: {format_feature_value(feat, selected_input[feat])} · Impact: {val:+.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with factor_col2:
                st.markdown("**Reducing Default Risk**")
                if safe_factors.empty:
                    st.info("No dominant risk-reducing factor in the selected local explanation.")
                for feat, val in safe_factors.items():
                    st.markdown(
                        f"""
                        <div class="factor-safe">
                            <strong>{FEATURE_LABELS[feat]}</strong><br>
                            Value: {format_feature_value(feat, selected_input[feat])} · Impact: {val:+.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("View complete selected record fields"):
                d1, d2, d3, d4 = st.columns(4)
                d1.write(f"**Loan Amount:** RM {selected_history['loan_amnt']:,.0f}")
                d1.write(f"**Installment:** RM {selected_history['installment']:,.0f}")
                d1.write(f"**Last Payment:** RM {selected_history['last_pymnt_amnt']:,.0f}")
                d2.write(f"**Annual Income:** RM {selected_history['annual_inc']:,.0f}")
                d2.write(f"**Interest Rate:** {selected_history['int_rate']:.1f}%")
                d2.write(f"**Credit Grade:** {selected_history['grade']}")
                d3.write(f"**DTI:** {selected_history['dti']:.1f}%")
                d3.write(f"**Revolving Util:** {selected_history['revol_util']:.1f}%")
                d3.write(f"**Revolving Balance:** RM {selected_history['revol_bal']:,.0f}")
                d4.write(f"**Avg Current Balance:** RM {selected_history['avg_cur_bal']:,.0f}")
                d4.write(f"**Historical Outcome:** {selected_history['actual_status']}")
                d4.write(f"**Model Decision:** {normalize_decision(replay_decision)}")

    with st.container(border=True):
        report_section("Search & Filter Historical Loans", "Filter records, then click Analyze in the row to validate with Hybrid XAI")
        f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1])
        with f1:
            search_text = st.text_input("Search Applicant ID / Reason / Status", placeholder="CSV-000001 / high DTI / charged off")
        with f2:
            selected_year = st.selectbox("Year", ["All"] + sorted(history_df["year"].dropna().unique().tolist(), reverse=True))
        with f3:
            selected_month = st.selectbox(
                "Month",
                ["All"] + sorted(history_df["month"].dropna().unique().tolist()),
                format_func=lambda x: "All" if x == "All" else pd.Timestamp(month=int(x), day=1, year=2000).strftime("%B"),
            )
        with f4:
            n_rows = st.slider("Rows", 10, 500, 80)

        f5, f6, f7, f8 = st.columns(4)
        with f5:
            status_only = st.selectbox("Actual Outcome", ["All", "Fully Paid", "Charged Off"])
        with f6:
            dec_only = st.selectbox("ML-XAI Decision", ["All", "Approved", "Rejected"])
        with f7:
            grade_filter = st.multiselect(
                "Credit Grade",
                ["A", "B", "C", "D", "E", "F", "G"],
                default=["A", "B", "C", "D", "E", "F", "G"],
            )
        with f8:
            risk_band = st.selectbox("Risk Range", ["All", "Low (≤30%)", "Medium (30–70%)", "High (>70%)"])

    filtered = history_df.copy()
    if search_text:
        search = search_text.lower()
        filtered = filtered[
            filtered["applicant_id"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["grade"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["primary_reason"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["actual_status"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["decision"].astype(str).str.lower().str.contains(search, na=False)
        ]
    if selected_year != "All":
        filtered = filtered[filtered["year"] == selected_year]
    if selected_month != "All":
        filtered = filtered[filtered["month"] == selected_month]
    if status_only != "All":
        filtered = filtered[filtered["actual_status"] == status_only]
    if dec_only != "All":
        filtered = filtered[filtered["decision"] == dec_only]
    if grade_filter:
        filtered = filtered[filtered["grade"].isin(grade_filter)]
    if risk_band == "Low (≤30%)":
        filtered = filtered[filtered["risk_score"] <= 0.30]
    elif risk_band == "Medium (30–70%)":
        filtered = filtered[(filtered["risk_score"] > 0.30) & (filtered["risk_score"] <= 0.70)]
    elif risk_band == "High (>70%)":
        filtered = filtered[filtered["risk_score"] > 0.70]

    filtered = filtered.sort_values("date", ascending=False)

    with st.container(border=True):
        report_section("Historical Loan Records", "Click Analyze on any record to compare historical outcome against Hybrid XAI")

        if filtered.empty:
            st.warning("No records match the selected filters.")
        else:
            header = st.columns([0.65, 0.95, 0.85, 0.9, 0.8, 0.85, 0.75, 0.75, 1.5, 0.8])
            headers = ["Date", "Applicant", "Amount", "Rate", "Actual", "ML-XAI Decision", "Risk", "Match", "Reason", "Action"]
            for h_col, h_text in zip(header, headers):
                h_col.markdown(f"**{h_text}**")

            st.markdown('<div class="history-divider"></div>', unsafe_allow_html=True)

            for idx, row in filtered.head(n_rows).iterrows():
                row_cols = st.columns([0.65, 0.95, 0.85, 0.9, 0.8, 0.85, 0.75, 0.75, 1.5, 0.8])
                row_cols[0].write(pd.to_datetime(row["date"]).strftime("%Y-%m"))
                row_cols[1].write(row["applicant_id"])
                row_cols[2].write(f"RM {row['loan_amnt']:,.0f}")
                row_cols[3].write(f"{row['int_rate']:.1f}%")
                row_cols[4].write(row["actual_status"])
                row_cols[5].write(row["decision"])
                row_cols[6].write(f"{row['risk_score']:.1%}")
                match_class = "match-good" if row["prediction_match"] == "Correct" else "match-bad"
                row_cols[7].markdown(f'<span class="match-badge {match_class}">{row["prediction_match"]}</span>', unsafe_allow_html=True)
                row_cols[8].write(row["primary_reason"])
                if row_cols[9].button("Analyze", key=f"analyze_{row['applicant_id']}", use_container_width=True):
                    st.session_state.selected_record_for_history = row.to_dict()
                    st.rerun()

                st.markdown('<div class="history-divider"></div>', unsafe_allow_html=True)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Filtered Historical CSV",
            csv_data,
            file_name="credxai_historical_records.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# PAGE 2 — LOAN ASSESSMENT SIMULATION
# ============================================================

elif current_page == "Loan Assessment Simulation":
    page_heading(
        "Loan Assessment Simulation",
        "Real-Time Explainable AI Loan Decisioning",
        "Simulate a new loan application with full Hybrid XAI reasoning.",
    )

    # This page is intentionally for new loan simulations only.
    # Historical comparison stays inside Loan History Analyzer.
    selected_record = None
    auto_run = False

    col_form, col_result = st.columns([1, 1.25], gap="large")

    with col_form:
        with st.container(border=True):
            report_section("Applicant Profile", "Income, grade, debt, and utilisation inputs")

            c1, c2 = st.columns(2)
            with c1:
                loan_amnt = st.number_input(
                    "Loan Amount (RM)",
                    1000.0,
                    None,
                    float(selected_record["loan_amnt"]) if selected_record else 15000.0,
                    step=500.0,
                    help=feature_help("loan_amnt"),
                )
                annual_inc = st.number_input(
                    "Annual Income (RM)",
                    10000.0,
                    500000.0,
                    float(selected_record["annual_inc"]) if selected_record else 60000.0,
                    step=1000.0,
                    help=feature_help("annual_inc"),
                )
                default_grade = selected_record["grade"] if selected_record else "C"
                grade_label = st.selectbox(
                    "Credit Grade",
                    ["A", "B", "C", "D", "E", "F", "G"],
                    index=["A", "B", "C", "D", "E", "F", "G"].index(default_grade),
                    help=feature_help("grade"),
                )
                dti = st.number_input(
                    "Debt-to-Income Ratio (%)",
                    0.0,
                    60.0,
                    float(selected_record["dti"]) if selected_record else 15.0,
                    step=0.5,
                    help=feature_help("dti"),
                )

            with c2:
                revol_util = st.number_input(
                    "Revolving Utilization (%)",
                    0.0,
                    100.0,
                    float(selected_record["revol_util"]) if selected_record else 45.0,
                    step=1.0,
                    help=feature_help("revol_util"),
                )
                avg_cur_bal = st.number_input(
                    "Avg Current Balance (RM)",
                    0.0,
                    200000.0,
                    float(selected_record["avg_cur_bal"]) if selected_record else 12000.0,
                    step=500.0,
                    help=feature_help("avg_cur_bal"),
                )
                revol_bal = st.number_input(
                    "Revolving Balance (RM)",
                    0.0,
                    200000.0,
                    float(selected_record["revol_bal"]) if selected_record else 15000.0,
                    step=500.0,
                    help=feature_help("revol_bal"),
                )
                last_pymnt_amnt = st.number_input(
                    "Last Payment Amount (RM)",
                    0.0,
                    50000.0,
                    float(selected_record["last_pymnt_amnt"]) if selected_record else 300.0,
                    step=50.0,
                    help=feature_help("last_pymnt_amnt"),
                )

        with st.container(border=True):
            report_section("Loan Terms", "Repayment and interest variables")
            installment = st.number_input(
                "Monthly Installment (RM)",
                50.0,
                5000.0,
                float(selected_record["installment"]) if selected_record else 450.0,
                step=10.0,
                help=feature_help("installment"),
            )
            int_rate = st.number_input(
                "Interest Rate (%)",
                0.0,
                100.0,
                float(selected_record["int_rate"]) if selected_record else 13.5,
                step=0.1,
                help=feature_help("int_rate"),
            )

            with st.expander("Credit Grade Guide"):
                st.markdown(
                    """
| Grade | Score Used by Model | Meaning | Risk Level |
|---|---:|---|---|
| A | 1 | Excellent borrower profile | Very Low Risk |
| B | 2 | Strong borrower profile | Low Risk |
| C | 3 | Acceptable / average borrower | Moderate Risk |
| D | 4 | Fair credit profile | Medium Risk |
| E | 5 | Weak credit profile | High Risk |
| F | 6 | Poor credit profile | Very High Risk |
| G | 7 | Highest default likelihood | Critical Risk |
                    """
                )

        run_btn = st.button("Run Loan Assessment", use_container_width=True)

    should_run = run_btn or auto_run

    with col_result:
        if should_run:
            input_data = {
                "loan_amnt": float(loan_amnt),
                "last_pymnt_amnt": float(last_pymnt_amnt),
                "avg_cur_bal": float(avg_cur_bal),
                "int_rate": float(int_rate),
                "grade": float(GRADE_MAP[grade_label]),
                "installment": float(installment),
                "annual_inc": float(annual_inc),
                "dti": float(dti),
                "revol_util": float(revol_util),
                "revol_bal": float(revol_bal),
            }

            with st.spinner("Running LightGBM prediction and Hybrid SHAP-LIME analysis..."):
                result = run_single_assessment(input_data, threshold, model, scaler, X_train_scaled, y_train)

            risk_score = result["risk_score"]
            hybrid_vec = result["hybrid_vec"]
            selected_features = result["selected_features"]
            lime_scores = result["lime_scores"]
            decision = result["decision"]

            decision_banner(decision, risk_score, threshold)

            if selected_record:
                comparison_card(selected_record["actual_status"], decision)

            repayment_years, total_repayment = calculate_repayment_years(input_data)

            with st.container(border=True):
                report_section("Assessment Summary", "Decision metrics and repayment estimate")
                m1, m2, m3 = st.columns(3)
                with m1:
                    metric_card(f"RM {loan_amnt:,.0f}", "Loan Amount", "blue")
                with m2:
                    metric_card(f"RM {installment:,.0f}", "Monthly Payment", "neutral")
                with m3:
                    metric_card(f"{int_rate:.1f}%", "Interest Rate", "amber")

                m4, m5, m6 = st.columns(3)
                with m4:
                    metric_card(f"{repayment_years:.1f} yrs", "Estimated Term", "neutral")
                with m5:
                    metric_card(grade_label, "Credit Grade", "blue")
                with m6:
                    metric_card(f"{dti:.1f}%", "DTI Ratio", "amber")

                with st.container(border=True):
                    report_section("Risk Indicator", "Low, medium, and high risk bands")
                    left, right = st.columns([1.2, 0.9], gap="large")

                    with left:
                        fig = plot_risk_gauge(risk_score)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with right:
                        tier_label = "Low" if risk_score <= 0.30 else "Medium" if risk_score <= 0.70 else "High"
                        tier_class = tier_label.lower()
                        st.markdown(
                            f"""
                            <div class='risk-summary'>
                                <div class='risk-summary-title'>Current Risk Assessment</div>
                                <div class='risk-score'>{risk_score:.1%}</div>
                                <div class='risk-tier-pill {tier_class}'>{tier_label} Risk</div>
                                <div class='risk-band'>
                                    <div class='risk-band-step low'>Low<br><span>0–30%</span></div>
                                    <div class='risk-band-step medium'>Medium<br><span>30–70%</span></div>
                                    <div class='risk-band-step high'>High<br><span>70–100%</span></div>
                                </div>
                                <div class='risk-detail'>The score sits in the <strong>{tier_label}</strong> band and is color-coded by risk level.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            with st.container(border=True):
                st.info("Complete the form or replay a historical record to unlock the explainable assessment.")

    if should_run:
        st.markdown("---")
        page_heading(
            "Explainable Intelligence",
            "Bank and Customer Decision Narratives",
            "OpenRouter converts model-supported XAI factors into user-specific explanations.",
        )

        with st.spinner("Generating OpenRouter explanations..."):
            bank_text = generate_bank_explanation(decision, risk_score, threshold, hybrid_vec, selected_features, input_data)
            customer_text = generate_customer_explanation(decision, risk_score, threshold, hybrid_vec, selected_features, input_data)

        exp_col1, exp_col2 = st.columns(2, gap="large")
        with exp_col1:
            with st.container(border=True):
                report_section("Bank Internal Explanation")
                st.markdown(f"<div class='xai-box'>{bank_text}</div>", unsafe_allow_html=True)

        with exp_col2:
            with st.container(border=True):
                report_section("Customer-Friendly Explanation")
                st.markdown(f"<div class='xai-box'>{customer_text}</div>", unsafe_allow_html=True)

        xai_col1, xai_col2 = st.columns([3, 2], gap="large")

        with xai_col1:
            with st.container(border=True):
                report_section("Hybrid XAI Importance Table")
                lime_df = pd.DataFrame(
                    [
                        {
                            "Feature": FEATURE_LABELS[f],
                            "LIME Pre-Score": round(lime_scores.get(f, 0.0), 5),
                            "Selected": "Yes" if f in selected_features else "No",
                            "Hybrid SHAP": round(hybrid_vec[FEATURES.index(f)], 5),
                        }
                        for f in sorted(FEATURES, key=lambda x: lime_scores.get(x, 0), reverse=True)
                    ]
                )
                st.dataframe(lime_df, use_container_width=True, hide_index=True)

        with xai_col2:
            with st.container(border=True):
                report_section("Risk Factor Snapshot")
                hybrid_series = pd.Series(hybrid_vec, index=FEATURES)
                selected_vals = hybrid_series[selected_features]
                top_risk = selected_vals[selected_vals > 0].sort_values(ascending=False)
                top_safe = selected_vals[selected_vals < 0].sort_values()

                if not top_risk.empty:
                    st.markdown("**Raising Default Risk**")
                    for feat, val in top_risk.items():
                        st.markdown(
                            f"""
                            <div class="factor-risk">
                                <strong>{FEATURE_LABELS[feat]}</strong><br>
                                Value: {format_feature_value(feat, input_data[feat])} · Impact: {val:+.4f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if not top_safe.empty:
                    st.markdown("**Reducing Default Risk**")
                    for feat, val in top_safe.items():
                        st.markdown(
                            f"""
                            <div class="factor-safe">
                                <strong>{FEATURE_LABELS[feat]}</strong><br>
                                Value: {format_feature_value(feat, input_data[feat])} · Impact: {val:+.4f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        with st.container(border=True):
            report_section("Unified Hybrid Importance Chart")
            fig = plot_hybrid_importance(hybrid_vec, selected_features)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ============================================================
# PAGE 3 — REPORTS & ANALYTICS
# ============================================================

elif current_page == "Reports & Analytics":
    page_heading(
        "Reports & Analytics",
        "Portfolio-Level Loan Risk Analytics",
        "Weekly, monthly, and annual views of risk behaviour across the real CSV-derived portfolio.",
    )

    log_view = decision_log.copy()
    log_view["date"] = pd.to_datetime(log_view["date"])
    log_view["week"] = log_view["date"].dt.to_period("W").astype(str)
    log_view["month"] = log_view["date"].dt.to_period("M").astype(str)
    log_view["year"] = log_view["date"].dt.year.astype(str)
    log_view["month_num"] = log_view["date"].dt.month

    with st.container(border=True):
        report_section("Report Filters", "Analyze the portfolio by date, risk, grade, and decision")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            selected_year = st.selectbox("Year", ["All"] + sorted(log_view["year"].dropna().unique().tolist(), reverse=True), key="report_year")
        with f2:
            selected_month = st.selectbox(
                "Month",
                ["All"] + sorted(log_view["month_num"].dropna().unique().tolist()),
                format_func=lambda x: "All" if x == "All" else pd.Timestamp(month=int(x), day=1, year=2000).strftime("%B"),
                key="report_month",
            )
        with f3:
            report_period = st.selectbox("Group By", ["Monthly", "Weekly", "Annual"])
        with f4:
            search_text = st.text_input("Search", placeholder="CSV-000001 / grade / reason", key="report_search")

        f5, f6, f7 = st.columns(3)
        with f5:
            decision_filter = st.selectbox("Decision", ["All", "Approved", "Rejected"])
        with f6:
            grade_filter = st.multiselect("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"], default=["A", "B", "C", "D", "E", "F", "G"], key="report_grade")
        with f7:
            risk_band = st.selectbox("Risk Band", ["All", "Low (≤30%)", "Medium (30–70%)", "High (>70%)"], key="report_risk")

    filtered = log_view.copy()

    if search_text:
        search = search_text.lower()
        filtered = filtered[
            filtered["applicant_id"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["grade"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["primary_reason"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["actual_status"].astype(str).str.lower().str.contains(search, na=False)
            | filtered["decision"].astype(str).str.lower().str.contains(search, na=False)
        ]

    if selected_year != "All":
        filtered = filtered[filtered["year"] == selected_year]
    if selected_month != "All":
        filtered = filtered[filtered["month_num"] == selected_month]
    if decision_filter != "All":
        filtered = filtered[filtered["decision"] == decision_filter]
    if grade_filter:
        filtered = filtered[filtered["grade"].isin(grade_filter)]
    if risk_band == "Low (≤30%)":
        filtered = filtered[filtered["risk_score"] <= 0.30]
    elif risk_band == "Medium (30–70%)":
        filtered = filtered[(filtered["risk_score"] > 0.30) & (filtered["risk_score"] <= 0.70)]
    elif risk_band == "High (>70%)":
        filtered = filtered[filtered["risk_score"] > 0.70]

    total = len(filtered)
    approved = int((filtered["decision"] == "Approved").sum())
    rejected = int((filtered["decision"] == "Rejected").sum())
    approval_rate = approved / total if total else 0
    rejection_rate = rejected / total if total else 0
    avg_risk = filtered["risk_score"].mean() if total else 0
    high_risk_pct = (filtered["risk_score"] > 0.70).mean() if total else 0

    with st.container(border=True):
        report_section("Dataset Summary", "Filtered dataset metrics update based on the selected filters above")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            metric_card(f"{total:,}", "Total Applications", "blue")
        with k2:
            metric_card(f"{approval_rate:.1%}", "Approval Rate", "green")
        with k3:
            metric_card(f"{rejection_rate:.1%}", "Rejection Rate", "red")
        with k4:
            metric_card(f"{avg_risk:.1%}", "Avg Risk", "amber")
        with k5:
            metric_card(f"{high_risk_pct:.1%}", "High Risk Loans", "red" if high_risk_pct > 0.25 else "amber")


    period_col = {"Monthly": "month", "Weekly": "week", "Annual": "year"}[report_period]

    with st.container(border=True):
        report_section("Decision Trend", f"Grouped by {report_period.lower()}")
        fig = plot_report_bar(filtered, period_col)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    r1, r2 = st.columns(2, gap="large")
    with r1:
        with st.container(border=True):
            report_section("Rejected Reason Breakdown")
            fig = plot_reason_breakdown(filtered, "Rejected")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with r2:
        with st.container(border=True):
            report_section("Approved Reason Breakdown")
            fig = plot_reason_breakdown(filtered, "Approved")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.container(border=True):
        report_section("Grade-Level Risk Distribution")
        fig = plot_grade_distribution(filtered)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.container(border=True):
        report_section("Filtered Records")
        display_df = filtered.sort_values("date", ascending=False).copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
        display_df = display_df[
            [
                "date",
                "applicant_id",
                "actual_status",
                "decision",
                "risk_score",
                "prediction_match",
                "primary_reason",
                "loan_amnt",
                "annual_inc",
                "grade",
                "dti",
                "revol_util",
            ]
        ].head(500)
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=360)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Portfolio CSV",
            csv_data,
            file_name="credxai_portfolio_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "CredXAI · LightGBM-Trained Hybrid XAI Loan Risk Assessment · Historical validation and loan simulation are decision-support tools only."
)
