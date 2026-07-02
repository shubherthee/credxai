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
