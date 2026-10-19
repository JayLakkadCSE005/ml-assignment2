# Machine Learning Assignment 2: Decision Forests, Boosting & Bias-Variance Tradeoffs

**Student Name:** Jay  
**Roll Number:** `M26CSE005`  
**Random Seed:** `26005` (derived deterministically from digits in roll number)  
**Dataset:** AT&T Cambridge Olivetti Faces ($K=40$ classes, $D=4096$ features)  
**Deliverables:** CVPR Two-Column PDF Report + GitHub Repository  

---

## 1. Repository Directory Structure

```text
.
├── configs/
│   └── config.yaml             # Central configuration (seed, depths, rounds, R=30)
├── splits/
│   └── olivetti_splits.npz     # Pre-saved non-leaking train/val/test splits
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loader & center occlusion masking [24:40, 24:40]
│   ├── bagging.py              # Decision tree grid search & Bagging ensemble
│   ├── samme.py                # Multi-class SAMME AdaBoost implementation
│   └── bias_variance.py        # Empirical Bias^2-Variance decomposition over R resamples
├── scripts/
│   ├── run_q1.py               # Question 1: Trees, Bagging across B, Occlusion test
│   ├── run_q2.py               # Question 2: SAMME Grid, Diagnostics, Benchmark vs Bagging
│   ├── run_q3.py               # Question 3: Empirical Bias-Variance across 30 resamples
│   ├── plot_figures.py         # Generates publication figures (PDF & PNG)
│   └── reproduce_all.sh        # Master one-click shell script
├── figures/                    # High-resolution vector & raster figures
│   ├── fig1_q1c_bagging_metrics.pdf
│   ├── fig2_q1d_occlusion_cases.pdf
│   ├── fig3_q2b_samme_diagnostics.pdf
│   ├── fig4_q2d_hardest_sample.pdf
│   └── fig5_q3b_bias_variance_depth.pdf
├── logs/                       # Full raw stdout & JSON execution logs
│   ├── q1_results.txt / json
│   ├── q2_results.txt / json
│   └── q3_results.txt / json
├── latex/                      # CVPR Two-Column Report LaTeX source
│   ├── main.tex                # Complete paper manuscript (<= 8 pages + refs)
│   ├── cvpr.sty                # CVPR document style
│   └── references.bib          # BibTeX citations
├── Assignment2.ipynb           # Original interactive exploration notebook
├── requirements.txt            # Python dependencies (pip)
├── environment.yml             # Conda environment definition
└── README.md                   # This reproduction guide
```

---

## 2. Environment Setup

### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate ml_assign2
```

### Option B: Using Pip Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Data Split & Occlusion Protocol

The Olivetti dataset consists of 40 identities with 10 images each (400 images total, $64 \times 64$ pixels). Using seed `26005`, the 10 images per identity are split without replacement:
- **Train:** 6 images/person ($240$ samples)
- **Validation:** 2 images/person ($80$ samples)
- **Test:** 2 images/person ($80$ samples)
- **Masked Test:** A $16 \times 16$ center box (`[24:40, 24:40]`, 256 pixels) zeroed out to evaluate occlusion robustness.

The split is saved in `splits/olivetti_splits.npz` and verified to contain zero data leakage.

---

## 4. Exact Reproduction Commands

You can reproduce all experiments and generate all logs/figures using a single command:

```bash
bash scripts/reproduce_all.sh
```

Or reproduce individual questions sequentially:

### Question 1: Decision Trees & Bagging
```bash
python3 scripts/run_q1.py
```
- Fits grid search across depths $\{2, 4, 8, 16, \text{None}\}$ and leaf sizes $\{1, 2, 4\}$.
- Verifies theoretical distinct bootstrap samples: $N(1-(1-1/N)^N) = 151.89$ vs $N(1-e^{-1}) = 151.71$.
- Evaluates Bagging metrics across $B \in \{1, 5, 10, 25, 50, 100, 200\}$.
- Tests clean ($88.75\%$) vs masked occlusion ($80.00\%$) robustness.
- Outputs saved to `logs/q1_results.txt` and `logs/q1_results.json`.

### Question 2: Multi-Class SAMME Boosting
```bash
python3 scripts/run_q2.py
```
- Runs 3-class toy example ($N=6, K=3$) with SAMME single step and error cutoff $1 - 1/K = 0.6667$.
- Performs hyperparameter grid on Olivetti: depths $\{1, 2, 3, 4\} \times$ rounds $\{10, 25, 50, 100\}$.
- Benchmarks Bagging vs. SAMME Boosting on 25, 50, 100 learners (accuracy and training time).
- Identifies hardest training sample (sample #199, Class 33) and logs its weight trajectory across 100 rounds.
- Outputs saved to `logs/q2_results.txt` and `logs/q2_results.json`.

### Question 3: Empirical Bias-Variance Decomposition
```bash
python3 scripts/run_q3.py
```
- Performs $R=30$ independent stratified training refits (sampling 6 with replacement per class).
- Computes empirical $\text{Bias}^2$ and $\text{Variance}$ under multi-class squared loss.
- Analyzes the effect of tree depth ($d=4, 16, \text{None}$) on variance reduction (up to $96.07\%$ variance removed by Bagging).
- Outputs saved to `logs/q3_results.txt` and `logs/q3_results.json`.

### Generate Publication Figures
```bash
python3 scripts/plot_figures.py
```
- Generates Fig 1 (Bagging metrics vs $B$), Fig 2 (Occlusion cases), Fig 3 (SAMME 4-panel diagnostics), Fig 4 (Hardest face & weight trajectory), and Fig 5 (Bias-Variance vs depth).
- All figures saved as both `.pdf` (vector) and `.png` (raster) in `figures/`.

---

## 5. CVPR PDF Report Compilation

The complete CVPR two-column manuscript is located in `latex/main.tex`.

### Compiling via Overleaf:
1. Zip the `latex/` folder along with `figures/`:
   ```bash
   zip -r cvpr_submission.zip latex/ figures/
   ```
2. Upload `cvpr_submission.zip` to Overleaf.
3. Select `pdfLaTeX` compiler and set `main.tex` as the root document.

### Page 1 Submission Metadata:
- **Author:** Jay
- **Roll Number:** M26CSE005
- **Repository:** https://github.com/Jay/ml-assignment2
- **Final Commit Hash:** `3910f1ef770090f19f6c4421485c37a5c77fe480` (`3910f1e`)

---

## 6. Key Results Summary

### Q1: Decision Trees & Bagging ($B=200$)
- **Single Tree Baseline:** Clean Test Acc = $46.25\%$, Masked Test Acc = $46.25\%$.
- **Final Bagging ($B=200$):** Clean Test Acc = **$88.75\%$**, Masked Test Acc = **$80.00\%$** ($+42.5\%$ improvement).
- **Distinct Samples per Bootstrap:** $151.89$ empirical vs $151.89$ theoretical exact ($63.29\%$).

### Q2: SAMME Boosting vs. Bagging
- **Selected Boosting ($d=4, T=100$):** Test Acc = $51.25\%$, Training Time = $2.29\text{s}$.
- **Bagging ($B=100$):** Test Acc = **$87.50\%$**, Training Time = **$1.68\text{s}$**.
- **Hardest Sample:** Training face #195 (Class 32) peak weight reached $0.0768$ ($18.4\times$ initial weight), correctly classified in only 12/100 weak rounds.

### Q3: Empirical Bias$^2$ and Variance ($R=30$ Resamples)
| Model | $\text{Bias}^2$ | Variance | $\text{Bias}^2+\text{Var}$ | Mean Test Acc |
| :--- | :---: | :---: | :---: | :---: |
| Single Tree ($d=\text{None}$) | $0.7745$ | $0.4002$ | $1.1747$ | $46.58\% \pm 4.1\%$ |
| **Final Bagging ($B=200$)** | **$0.5750$** | **$0.0772$** | **$0.6522$** | **$88.50\% \pm 2.4\%$** |
| Selected Boosting ($d=4, T=100$) | $0.7950$ | $0.1834$ | $0.9784$ | $51.13\% \pm 3.8\%$ |

- **Variance Removed by Bagging on Unbounded Trees:** **$77.66\%$** ($0.4002 \to 0.0894$).

---

## 7. Viva / Defense Q&A Cheatsheet

1. **Why does Bagging dramatically outperform Boosting on this dataset?**  
   *Answer:* The Olivetti dataset has 40 classes with only 6 training images per class in 4096 dimensions. Boosting sequentially reweights misclassified images, overconcentrating probability mass on noisy/outlier faces ($35\%$ of weight on top-10 images). Bagging, conversely, samples training points uniformly with replacement and averages independent subspace splits, drastically suppressing variance without distorting the data distribution.

2. **Why does Bagging maintain $80.0\%$ accuracy when faces are center-masked?**  
   *Answer:* A single decision tree greedily selects high-importance features at the root (often nose/eyes). When these are occluded, the tree fails catastrophically. In Bagging ($B=200$), different bootstrap resamples produce different root splits focusing on peripheral features (cheeks, jawline, forehead), enabling consensus voting to easily overcome localized occlusion.

3. **What is the mathematical threshold for weak learners in SAMME?**  
   *Answer:* For $K$ classes, random guessing error is $1 - 1/K$. In SAMME, the weight is $\alpha_t = \log((1-\epsilon_t)/\epsilon_t) + \log(K-1)$. When $\epsilon_t < 1 - 1/K = 39/40 = 0.975$, $\alpha_t > 0$, guaranteeing positive contribution. If $\epsilon_t \ge 0.975$, the learner is worse than chance and boosting stops.
