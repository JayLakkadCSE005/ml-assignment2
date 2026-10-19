import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def samme_single_step(X_data, y_data, w, max_depth, K, tree_seed):
    """
    Runs one complete round of SAMME (Stagewise Additive Modeling using Multi-class Exponential loss).
    Returns:
      tree, preds, err, alpha, w_updated, is_valid
    Formula:
      alpha = log((1 - err) / err) + log(K - 1)
      w_{i, t+1} = w_{i, t} * exp(alpha * I(y_i != c_t(x_i)))
    Handles weak learners not better than random guessing (err >= 1 - 1/K).
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=tree_seed)
    tree.fit(X_data, y_data, sample_weight=w)
    preds = tree.predict(X_data)

    incorrect = (preds != y_data).astype(np.float64)
    err = float(np.dot(w, incorrect) / np.sum(w))

    max_allowed_err = 1.0 - (1.0 / K)
    if err >= max_allowed_err - 1e-12:
        return tree, preds, err, 0.0, w.copy(), False

    err_clipped = np.clip(err, 1e-12, max_allowed_err - 1e-12)
    alpha = float(np.log((1.0 - err_clipped) / err_clipped) + np.log(K - 1.0))

    w_unnorm = w * np.exp(alpha * incorrect)
    w_updated = w_unnorm / np.sum(w_unnorm)
    return tree, preds, err, alpha, w_updated, True

def fit_samme_full(X_tr, y_tr, max_depth, n_rounds, K=40, base_seed=26005):
    """
    Fits SAMME for n_rounds using samme_single_step.
    Logs diagnostics for each round:
      error, alpha, max sample weight, weight of 10 heaviest samples, weight trajectory.
    """
    rng_boost = np.random.default_rng(base_seed)
    N = X_tr.shape[0]
    w = np.ones(N, dtype=np.float64) / N

    trees = []
    alphas = []
    errors = []
    max_weights = []
    top10_weights = []
    weight_history = [w.copy()]

    for t in range(n_rounds):
        tree_seed = int(rng_boost.integers(0, 2**31 - 1))
        tree, preds, err, alpha, w_next, is_valid = samme_single_step(
            X_tr, y_tr, w, max_depth=max_depth, K=K, tree_seed=tree_seed
        )

        if not is_valid:
            if len(trees) == 0:
                trees.append(tree)
                alphas.append(1.0)
                errors.append(err)
                max_weights.append(np.max(w))
                top10_weights.append(np.sum(np.sort(w)[-10:]))
            break

        trees.append(tree)
        alphas.append(alpha)
        errors.append(err)
        max_weights.append(np.max(w))
        top10_weights.append(np.sum(np.sort(w)[-10:]))

        w = w_next
        weight_history.append(w.copy())

    return {
        "trees": trees,
        "alphas": np.array(alphas, dtype=np.float64),
        "errors": np.array(errors, dtype=np.float64),
        "max_weights": np.array(max_weights, dtype=np.float64),
        "top10_weights": np.array(top10_weights, dtype=np.float64),
        "weight_history": np.array(weight_history, dtype=np.float64),  # shape (n_rounds+1, N)
        "final_weights": w
    }

def predict_samme(model_dict, X_eval, n_rounds=None, K=40):
    """
    Computes ensemble predictions and normalized confidence vectors for SAMME:
      S_k(x) = sum_{t=1}^T alpha_t * I(c_t(x) == k)
      q_k(x) = S_k(x) / sum(alpha_t)
    """
    if n_rounds is None:
        n_rounds = len(model_dict["trees"])

    trees = model_dict["trees"][:n_rounds]
    alphas = model_dict["alphas"][:n_rounds]
    n_samples = X_eval.shape[0]

    S = np.zeros((n_samples, K), dtype=np.float64)
    for tree, alpha in zip(trees, alphas):
        preds = tree.predict(X_eval)
        S[np.arange(n_samples), preds] += alpha

    total_alpha = np.sum(alphas)
    if total_alpha > 0:
        q = S / total_alpha
    else:
        q = np.ones((n_samples, K)) / K

    preds_final = np.argmax(q, axis=1)
    mean_conf = float(np.mean(np.max(q, axis=1)))
    return preds_final, mean_conf, q
