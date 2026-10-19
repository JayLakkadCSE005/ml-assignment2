import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def run_decision_tree_grid(x_train, y_train, x_val, y_val, x_test, y_test, 
                           depths=[2, 4, 8, 16, None], min_leaves=[1, 2, 4], seed=26005):
    """
    Evaluates DecisionTreeClassifier across depths and min_samples_leaf.
    """
    records = []
    for d in depths:
        for m in min_leaves:
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=seed)
            clf.fit(x_train, y_train)

            tr_acc = clf.score(x_train, y_train)
            val_acc = clf.score(x_val, y_val)
            te_acc = clf.score(x_test, y_test)

            n_leaves = clf.get_n_leaves()
            leaf_ids = clf.apply(x_train)
            _, leaf_counts = np.unique(leaf_ids, return_counts=True)
            mean_samples_leaf = float(np.mean(leaf_counts))

            records.append({
                "max_depth": str(d),
                "min_samples_leaf": m,
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "test_acc": te_acc,
                "n_leaves": n_leaves,
                "mean_samples_per_leaf": mean_samples_leaf
            })
    return pd.DataFrame(records)

def majority_vote_and_margin(tree_preds_matrix, num_classes=40):
    """
    tree_preds_matrix: shape (B, N_samples)
    Returns:
      ensemble_preds: (N_samples,)
      vote_margin: float, mean difference between top-1 and top-2 vote fractions
    """
    B, n_samples = tree_preds_matrix.shape
    vote_counts = np.zeros((n_samples, num_classes), dtype=np.float64)
    for b in range(B):
        vote_counts[np.arange(n_samples), tree_preds_matrix[b]] += 1.0

    ensemble_preds = np.argmax(vote_counts, axis=1)
    
    # Sort vote counts descending to get top-1 and top-2 fractions
    sorted_votes = np.sort(vote_counts, axis=1)[:, ::-1]
    top1 = sorted_votes[:, 0] / B
    top2 = sorted_votes[:, 1] / B
    margins = top1 - top2
    return ensemble_preds, float(np.mean(margins))

def train_bagging_ensemble(x_train, y_train, B_values=[1, 5, 10, 25, 50, 100, 200],
                           max_depth=None, min_samples_leaf=1, base_seed=26005):
    """
    Trains bagging models up to max(B_values) and evaluates OOB accuracy,
    pairwise disagreement, and vote margin for each B.
    """
    N_train = x_train.shape[0]
    max_B = max(B_values)
    rng = np.random.default_rng(base_seed)

    trees = []
    oob_masks = []  # oob_masks[b] is boolean array of shape (N_train,)

    for b in range(max_B):
        boot_idx = rng.choice(N_train, size=N_train, replace=True)
        in_bag = np.zeros(N_train, dtype=bool)
        in_bag[boot_idx] = True
        oob_mask = ~in_bag

        tree_seed = int(rng.integers(0, 2**31 - 1))
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=tree_seed)
        tree.fit(x_train[boot_idx], y_train[boot_idx])

        trees.append(tree)
        oob_masks.append(oob_mask)

    return trees, oob_masks

def evaluate_bagging_metrics(trees, oob_masks, x_train, y_train, B_values=[1, 5, 10, 25, 50, 100, 200], num_classes=40):
    """
    Computes distinct samples, OOB accuracy, pairwise disagreement, and vote margin across B.
    """
    N_train = x_train.shape[0]
    results = []
    
    # Precompute train predictions for all trees: shape (max_B, N_train)
    max_B = len(trees)
    all_train_preds = np.zeros((max_B, N_train), dtype=int)
    for b in range(max_B):
        all_train_preds[b] = trees[b].predict(x_train)

    for B in B_values:
        sub_preds = all_train_preds[:B]
        
        # 1. Empirical distinct samples across B bootstraps
        # An in-bag sample is one that is NOT out of bag at least once
        in_bag_any = np.zeros(N_train, dtype=bool)
        for b in range(B):
            in_bag_any |= (~oob_masks[b])
        emp_distinct = int(np.sum(in_bag_any))

        # 2. OOB Accuracy
        oob_vote_counts = np.zeros((N_train, num_classes), dtype=np.float64)
        oob_counts = np.zeros(N_train, dtype=int)
        for b in range(B):
            oob_indices = np.where(oob_masks[b])[0]
            if len(oob_indices) > 0:
                oob_vote_counts[oob_indices, sub_preds[b, oob_indices]] += 1.0
                oob_counts[oob_indices] += 1

        evaluable = oob_counts > 0
        if np.sum(evaluable) > 0:
            oob_preds = np.argmax(oob_vote_counts[evaluable], axis=1)
            oob_acc = float(np.mean(oob_preds == y_train[evaluable]))
        else:
            oob_acc = 0.0

        # 3. Average Pairwise Disagreement on training set
        if B > 1:
            disagreements = []
            for b1 in range(B):
                for b2 in range(b1 + 1, B):
                    disagreements.append(np.mean(sub_preds[b1] != sub_preds[b2]))
            avg_disagreement = float(np.mean(disagreements))
        else:
            avg_disagreement = 0.0

        # 4. Mean Vote Margin
        _, mean_margin = majority_vote_and_margin(sub_preds, num_classes=num_classes)

        results.append({
            "B": B,
            "empirical_distinct": emp_distinct,
            "oob_acc": oob_acc,
            "disagreement": avg_disagreement,
            "vote_margin": mean_margin
        })

    return pd.DataFrame(results)

def evaluate_ensemble(trees, X_eval, y_eval, num_classes=40):
    """
    Evaluates bagged ensemble on an evaluation set.
    """
    B = len(trees)
    n_samples = X_eval.shape[0]
    vote_counts = np.zeros((n_samples, num_classes), dtype=np.float64)
    for tree in trees:
        preds = tree.predict(X_eval)
        vote_counts[np.arange(n_samples), preds] += 1.0

    probs = vote_counts / B
    final_preds = np.argmax(probs, axis=1)
    acc = float(np.mean(final_preds == y_eval))
    mean_conf = float(np.mean(np.max(probs, axis=1)))
    return final_preds, acc, mean_conf, probs
