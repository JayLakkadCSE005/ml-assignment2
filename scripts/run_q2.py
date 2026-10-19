#!/usr/bin/env python3
"""
Question 2 Reproduction Script:
SAMME Multi-class Boosting, Diagnostics, Benchmark vs Bagging, Hardest Sample, and Occlusion Analysis.
"""
import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_splits
from src.bagging import evaluate_ensemble
from src.samme import samme_single_step, fit_samme_full, predict_samme

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
    log_file = "logs/q2_results.txt"
    log_json = "logs/q2_results.json"

    with open(log_file, "w") as out_f:
        def log(msg=""):
            print(msg)
            out_f.write(msg + "\n")

        log("=" * 70)
        log("QUESTION 2: MULTI-CLASS SAMME BOOSTING")
        log("=" * 70)

        # -------------------------------------------------------------
        # Q2(a): 3-Class Toy Example (N=6, K=3)
        # -------------------------------------------------------------
        log("\n--- Q2(a): 3-Class Toy Dataset with SAMME Single Step ---")
        X_toy = np.array([
            [1.0, 2.0], [1.5, 1.8],  # Class 0
            [5.0, 8.0], [6.0, 9.0],  # Class 1
            [9.0, 1.0], [8.5, 2.0]   # Class 2
        ])
        y_toy = np.array([0, 0, 1, 1, 2, 2])
        K_toy = 3
        N_toy = len(y_toy)
        w_init = np.ones(N_toy, dtype=np.float64) / N_toy

        _, toy_preds, toy_err, toy_alpha, w_next, valid = samme_single_step(
            X_toy, y_toy, w_init, max_depth=1, K=K_toy, tree_seed=seed
        )
        log(f"True Labels y        : {y_toy}")
        log(f"Initial Weights w^(1): {np.round(w_init, 4)}")
        log(f"Weak Learner Preds   : {toy_preds}")
        log(f"Misclassified Mask   : {(toy_preds != y_toy).astype(int)}")
        log(f"Weighted Error (eps) : {toy_err:.4f}  (Threshold 1 - 1/K = {1 - 1/K_toy:.4f})")
        log(f"Learner Weight alpha : {toy_alpha:.4f}")
        log(f"Updated Weights w^(2): {np.round(w_next, 4)}  (Sum = {np.sum(w_next):.4f})")

        # -------------------------------------------------------------
        # Q2(b): SAMME on Olivetti Faces (Depths 1,2,3,4 x Rounds 10,25,50,100)
        # -------------------------------------------------------------
        log("\n--- Q2(b): SAMME Hyperparameter Grid Search ---")
        boost_depths = config["question_2"]["samme_grid"]["depths"]
        boost_rounds = config["question_2"]["samme_grid"]["rounds"]
        max_T = max(boost_rounds)
        K = config["dataset"]["num_identities"]

        results_q2b = []
        models_by_depth = {}
        for d in boost_depths:
            t0 = time.time()
            model_d = fit_samme_full(x_train, y_train, max_depth=d, n_rounds=max_T, K=K, base_seed=seed)
            fit_time = time.time() - t0
            models_by_depth[d] = model_d

            for r_count in boost_rounds:
                tr_preds, _, _ = predict_samme(model_d, x_train, n_rounds=r_count, K=K)
                val_preds, _, _ = predict_samme(model_d, x_val, n_rounds=r_count, K=K)
                te_preds, _, _ = predict_samme(model_d, x_test, n_rounds=r_count, K=K)

                results_q2b.append({
                    "max_depth": d,
                    "n_rounds": r_count,
                    "train_acc": round(float(np.mean(tr_preds == y_train)), 4),
                    "val_acc": round(float(np.mean(val_preds == y_val)), 4),
                    "test_acc": round(float(np.mean(te_preds == y_test)), 4),
                    "fit_time_100": round(fit_time, 2)
                })

        df_q2b = pd.DataFrame(results_q2b)
        log(df_q2b.to_string(index=False))

        best_boost_depth = config["question_2"]["selected_model"]["depth"]
        best_boost_rounds = config["question_2"]["selected_model"]["rounds"]
        sel_model = models_by_depth[best_boost_depth]
        log(f"\nSelected Boosting Config: depth={best_boost_depth}, rounds={best_boost_rounds}")

        # -------------------------------------------------------------
        # Q2(c): Compare Boosting and Bagging (25, 50, 100 learners)
        # -------------------------------------------------------------
        log("\n--- Q2(c): Benchmark: Bagging vs. SAMME Boosting (Speed & Accuracy) ---")
        learner_counts = config["question_2"]["learner_comparison"]["counts"]
        results_q2c = []
        N_train = x_train.shape[0]

        for M in learner_counts:
            # Bagging with M learners
            t0 = time.time()
            rng_b = np.random.default_rng(seed)
            bag_trees_M = []
            for b in range(M):
                boot_idx = rng_b.choice(N_train, size=N_train, replace=True)
                t_seed = int(rng_b.integers(0, 2**31 - 1))
                tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=t_seed)
                tree.fit(x_train[boot_idx], y_train[boot_idx])
                bag_trees_M.append(tree)
            bag_time = time.time() - t0

            _, bag_tr_acc, _, _ = evaluate_ensemble(bag_trees_M, x_train, y_train)
            _, bag_val_acc, _, _ = evaluate_ensemble(bag_trees_M, x_val, y_val)
            _, bag_te_acc, _, _ = evaluate_ensemble(bag_trees_M, x_test, y_test)

            results_q2c.append({
                "Method": "Bagging (depth=None)",
                "Learners": M,
                "train_acc": round(bag_tr_acc, 4),
                "val_acc": round(bag_val_acc, 4),
                "test_acc": round(bag_te_acc, 4),
                "train_time_sec": round(bag_time, 2)
            })

            # Boosting with M learners
            t0 = time.time()
            boost_M = fit_samme_full(x_train, y_train, max_depth=best_boost_depth, n_rounds=M, K=K, base_seed=seed)
            boost_time = time.time() - t0

            b_tr_preds, _, _ = predict_samme(boost_M, x_train, n_rounds=M, K=K)
            b_val_preds, _, _ = predict_samme(boost_M, x_val, n_rounds=M, K=K)
            b_te_preds, _, _ = predict_samme(boost_M, x_test, n_rounds=M, K=K)

            results_q2c.append({
                "Method": f"Boosting (depth={best_boost_depth})",
                "Learners": M,
                "train_acc": round(float(np.mean(b_tr_preds == y_train)), 4),
                "val_acc": round(float(np.mean(b_val_preds == y_val)), 4),
                "test_acc": round(float(np.mean(b_te_preds == y_test)), 4),
                "train_time_sec": round(boost_time, 2)
            })

        df_q2c = pd.DataFrame(results_q2c)
        log(df_q2c.to_string(index=False))

        # -------------------------------------------------------------
        # Q2(d): Heaviest Training Sample Analysis & Occlusion Test
        # -------------------------------------------------------------
        w_hist = sel_model["weight_history"][:-1]  # shape (100, 240)
        peak_round, hardest_sample_idx = np.unravel_index(np.argmax(w_hist), w_hist.shape)
        true_hard_class = int(y_train[hardest_sample_idx])

        hard_sample_x = x_train[hardest_sample_idx:hardest_sample_idx+1]
        weak_preds_hard = np.array([t.predict(hard_sample_x)[0] for t in sel_model["trees"]])
        was_correct = (weak_preds_hard == true_hard_class)

        log("\n--- Q2(d): Heaviest Training Face Diagnostics ---")
        log(f"Hardest Sample Index  : #{hardest_sample_idx} (True Class: {true_hard_class})")
        log(f"Initial Weight (t=1)   : {w_hist[0, hardest_sample_idx]:.5f}")
        log(f"Peak Weight            : {w_hist[peak_round, hardest_sample_idx]:.5f} (at Round {peak_round + 1})")
        log(f"Mean Weight (100 rds)  : {np.mean(w_hist[:, hardest_sample_idx]):.5f}")
        log(f"Correct Weak Classifications: {np.sum(was_correct)} / 100 rounds")

        # Robustness comparison on Clean vs Masked test
        bst_clean_preds, bst_clean_conf, _ = predict_samme(sel_model, x_test, n_rounds=best_boost_rounds, K=K)
        bst_mask_preds,  bst_mask_conf,  _ = predict_samme(sel_model, x_test_masked, n_rounds=best_boost_rounds, K=K)
        bst_clean_acc = float(np.mean(bst_clean_preds == y_test))
        bst_mask_acc  = float(np.mean(bst_mask_preds == y_test))

        log(f"\nSAMME (d={best_boost_depth}, T={best_boost_rounds}) Clean Test Acc: {bst_clean_acc:.4f} (Conf: {bst_clean_conf:.4f})")
        log(f"SAMME (d={best_boost_depth}, T={best_boost_rounds}) Masked Test Acc: {bst_mask_acc:.4f} (Conf: {bst_mask_conf:.4f})")

    # Save JSON summary
    json_data = {
        "toy_example": {
            "preds": toy_preds.tolist(),
            "err": toy_err,
            "alpha": toy_alpha,
            "w_next": w_next.tolist()
        },
        "q2b_grid": df_q2b.to_dict(orient="records"),
        "q2c_benchmark": df_q2c.to_dict(orient="records"),
        "hardest_sample": {
            "index": int(hardest_sample_idx),
            "true_class": true_hard_class,
            "peak_weight": float(w_hist[peak_round, hardest_sample_idx]),
            "peak_round": int(peak_round + 1),
            "correct_rounds_count": int(np.sum(was_correct))
        },
        "q2d_occlusion": {
            "clean_acc": bst_clean_acc,
            "clean_conf": bst_clean_conf,
            "masked_acc": bst_mask_acc,
            "masked_conf": bst_mask_conf
        }
    }
    with open(log_json, "w") as jf:
        json.dump(json_data, jf, indent=2)

    print(f"\n[Q2 Complete] Logs written to {log_file} and {log_json}")

if __name__ == "__main__":
    main()
