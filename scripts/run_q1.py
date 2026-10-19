#!/usr/bin/env python3
"""
Question 1 Reproduction Script:
Decision Trees, Bootstrap Aggregation, OOB Metrics, and Occlusion Robustness.
"""
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_splits
from src.bagging import (
    run_decision_tree_grid,
    train_bagging_ensemble,
    evaluate_bagging_metrics,
    evaluate_ensemble
)

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("random_seed", 26005)
    splits = load_splits()
    x_train, y_train = splits["x_train"], splits["y_train"]
    x_val, y_val     = splits["x_val"], splits["y_val"]
    x_test, y_test   = splits["x_test"], splits["y_test"]
    x_test_masked    = splits["x_test_masked"]

    os.makedirs("logs", exist_ok=True)
    log_file = "logs/q1_results.txt"
    log_json = "logs/q1_results.json"

    with open(log_file, "w") as out_f:
        def log(msg=""):
            print(msg)
            out_f.write(msg + "\n")

        log("=" * 70)
        log("QUESTION 1: DECISION TREES & BAGGING ENSEMBLE")
        log("=" * 70)

        # -------------------------------------------------------------
        # Q1(a): Decision Tree Grid Search
        # -------------------------------------------------------------
        log("\n--- Q1(a): Hyperparameter Tuning across Depths & Min Leaf Samples ---")
        depths = config["question_1"]["grid_search"]["depths"]
        min_leaves = config["question_1"]["grid_search"]["min_samples_leaf"]
        df_q1a = run_decision_tree_grid(x_train, y_train, x_val, y_val, x_test, y_test,
                                        depths=depths, min_leaves=min_leaves, seed=seed)
        log(df_q1a.to_string(index=False))

        # Best on validation
        best_val_idx = df_q1a["val_acc"].idxmax()
        best_cfg = df_q1a.iloc[best_val_idx]
        log(f"\nBest Validation Config: max_depth={best_cfg['max_depth']}, min_samples_leaf={best_cfg['min_samples_leaf']}")
        log(f"Train Acc: {best_cfg['train_acc']:.4f} | Val Acc: {best_cfg['val_acc']:.4f} | Test Acc: {best_cfg['test_acc']:.4f}")

        # -------------------------------------------------------------
        # Q1(b) & Q1(c): Bagging across B values
        # -------------------------------------------------------------
        log("\n--- Q1(b) & Q1(c): Bootstrap Aggregation Metrics across B ---")
        N_train = x_train.shape[0]
        exact_distinct = N_train * (1 - (1 - 1 / N_train) ** N_train)
        asymp_distinct = N_train * (1 - np.exp(-1))
        log(f"Theoretical Distinct per Bootstrap: Exact = {exact_distinct:.2f} | Asymptotic = {asymp_distinct:.2f}")

        B_values = config["question_1"]["bagging"]["b_values"]
        frozen_depth = config["question_1"]["frozen_single_tree"]["max_depth"]
        frozen_leaf = config["question_1"]["frozen_single_tree"]["min_samples_leaf"]

        trees, oob_masks = train_bagging_ensemble(x_train, y_train, B_values=B_values,
                                                  max_depth=frozen_depth, min_samples_leaf=frozen_leaf,
                                                  base_seed=seed)
        df_q1bc = evaluate_bagging_metrics(trees, oob_masks, x_train, y_train, B_values=B_values, num_classes=40)
        log(df_q1bc.to_string(index=False))

        # -------------------------------------------------------------
        # Q1(d): Frozen Single Tree vs Final Bagged Model (B=200)
        # -------------------------------------------------------------
        log("\n--- Q1(d): Robustness to Occlusion (Clean vs Center-Masked Test Set) ---")
        single_tree = DecisionTreeClassifier(max_depth=frozen_depth, min_samples_leaf=frozen_leaf, random_state=seed)
        single_tree.fit(x_train, y_train)

        st_clean_preds = single_tree.predict(x_test)
        st_clean_acc   = float(np.mean(st_clean_preds == y_test))
        st_clean_conf  = float(np.mean(np.max(single_tree.predict_proba(x_test), axis=1)))

        st_mask_preds  = single_tree.predict(x_test_masked)
        st_mask_acc    = float(np.mean(st_mask_preds == y_test))
        st_mask_conf   = float(np.mean(np.max(single_tree.predict_proba(x_test_masked), axis=1)))

        final_B = config["question_1"]["bagging"]["final_b"]
        bag_trees_final = trees[:final_B]
        bag_clean_preds, bag_clean_acc, bag_clean_conf, _ = evaluate_ensemble(bag_trees_final, x_test, y_test)
        bag_mask_preds, bag_mask_acc, bag_mask_conf, _    = evaluate_ensemble(bag_trees_final, x_test_masked, y_test)

        q1d_summary = pd.DataFrame([
            {"Model": "Single Tree (d=None)", "Test Condition": "Clean", "Accuracy": st_clean_acc, "Confidence": st_clean_conf},
            {"Model": "Single Tree (d=None)", "Test Condition": "Masked", "Accuracy": st_mask_acc, "Confidence": st_mask_conf},
            {"Model": f"Bagging (B={final_B})", "Test Condition": "Clean", "Accuracy": bag_clean_acc, "Confidence": bag_clean_conf},
            {"Model": f"Bagging (B={final_B})", "Test Condition": "Masked", "Accuracy": bag_mask_acc, "Confidence": bag_mask_conf},
        ])
        log(q1d_summary.to_string(index=False))

    # Save JSON summary
    json_data = {
        "q1a_grid": df_q1a.to_dict(orient="records"),
        "q1bc_bagging": df_q1bc.to_dict(orient="records"),
        "q1d_summary": q1d_summary.to_dict(orient="records")
    }
    with open(log_json, "w") as jf:
        json.dump(json_data, jf, indent=2)

    print(f"\n[Q1 Complete] Logs written to {log_file} and {log_json}")

if __name__ == "__main__":
    main()
