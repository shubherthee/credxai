# CredXAI: Explainable AI for Transparency in Credit Assessment

## Overview
Credit risk assessment is fundamental to financial institutions, guiding loan approvals and cushioning lenders against default losses. However, regulatory frameworks such as the General Data Protection Regulation (GDPR) and the Equal Credit Opportunity Act (ECOA) mandate that lending practices must be transparent, explainable, and reasonable.

While modern Machine Learning (ML) models offer high predictive power, they often act as "black boxes." Existing explainability techniques, such as SHAP and LIME, suffer from critical weaknesses when applied to imbalanced credit data. These include:
- Volatile feature rankings
- Inadequate ability to differentiate between classes of borrower risk
- High computational expenses that hinder real-time application

## The Research
To address these challenges, this study conducted a comprehensive evaluation of six machine learning algorithms:
1. Logistic Regression
2. Support Vector Machines (SVM)
3. Random Forest
4. Balanced Random Forest
5. XGBoost
6. LightGBM

These models were tested across five different degrees of data imbalance. Depending on performance, the three best models for each level were selected for detailed analysis. 

We generated separate SHAP and LIME explanations to rigorously compare their stability, discriminative power, and computational efficiency.

## Integrated SHAP-LIME Solution & Dashboard
Building upon the research findings, this project introduces a novel approach: **integrating SHAP and LIME techniques** to maximize their combined potential. 

This repository contains the **Proof-of-Concept Dashboard** designed for real-world banking decision-making. The deployed dashboard utilizes:
- **Algorithm**: LightGBM (identified as a top-performing model)
- **Data Imbalance Ratio**: 95:5 (Non-Default to Default)
- **XAI Engine**: Integrated Hybrid SHAP-LIME explanation engine

This hybrid approach ensures high computational efficiency for real-time assessments while providing stable, highly discriminative, and regulatory-compliant explanations for both credit analysts and applicants.

## Deployment & Setup
This application is designed to be deployed on **Streamlit Community Cloud**. It utilizes a pre-processed SQLite database (`lending_data.db`) to overcome memory constraints and ensure lightning-fast real-time inference without needing to load massive CSV files into memory.
