import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.samme import fit_samme_full, predict_samme

def generate_stratified_resamples(y_train, R=30, K=40, samples_per_id=6, seed=26005):
    """
    Generates R stratified resampled training sets by sampling 6 samples
    with replacement WITHIN each identity.
    """
    rng = np.random.default_rng(seed)
    resampled_indices = []
    for r in range(R):
        r_idx = []
        for person_id in range(K):
            id_mask = np.where(y_train == person_id)[0]
            sampled = rng.choice(id_mask, size=samples_per_id, replace=True)
            r_idx.extend(sampled)
        resampled_indices.append(np.array(r_idx))
    return resampled_indices

def compute_bias_variance(P_array, E_target, y_true):
    """
    P_array: shape (R, N_test, K) predicted probability vectors across R refits
    E_target: shape (N_test, K) one-hot encoded ground truth
    y_true: shape (N_test,) ground truth class labels

    Formulas:
      P_bar(x) = mean_r P_r(x)
      Bias^2   = mean_i ||P_bar(x_i) - e_{y_i}||_2^2
      Variance = mean_i (1/R sum_r ||P_r(x_i) - P_bar(x_i)||_2^2)
    """
    P_bar = np.mean(P_array, axis=0)  # (N_test, K)
    bias_sq = float(np.mean(np.sum((P_bar - E_target) ** 2, axis=-1)))
    variance = float(np.mean(np.sum((P_array - P_bar[None, :, :]) ** 2, axis=-1)))

    preds_r = np.argmax(P_array, axis=-1)  # (R, N_test)
    accs_r = np.mean(preds_r == y_true[None, :], axis=1)  # (R,)
    mean_acc = float(np.mean(accs_r))
    std_acc = float(np.std(accs_r))

    return bias_sq, variance, mean_acc, std_acc
