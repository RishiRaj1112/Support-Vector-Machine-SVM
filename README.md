# Support Vector Machine (SVM) Classification

An end-to-end implementation and evaluation of **Support Vector Machines (SVM)** for binary classification using **Scikit-Learn** on the Breast Cancer Wisconsin diagnostic dataset.

---

## 📌 Project Overview

Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression tasks. An SVM maps data points into a multi-dimensional feature space to identify the optimal hyperplane that maximizes the geometric margin between distinct classes. Data points closest to this decision boundary are called **Support Vectors**, and they exclusively determine the orientation and position of the hyperplane.

This project covers:
1. **Data Preprocessing & Standard Scaling**: Feature normalization using `StandardScaler` to ensure scale-sensitive distance metrics perform accurately.
2. **Exploratory Data Analysis (EDA)**: Visualizing feature correlations and target distributions.
3. **Model Training & Hyperparameter Tuning**: Evaluating the regularizing parameter $C$ and kernel choices (Linear vs. Radial Basis Function - RBF).
4. **Performance Evaluation**: Classification metrics including Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

---

## 🔬 Mathematical Intuition

Given training pairs $(x_i, y_i)$ where $y_i \in \{-1, 1\}$:

### 1. Optimal Margin Hyperplane
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i (w^T x_i + b) \ge 1, \quad \forall i$$

### 2. Soft-Margin Regularization
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{subject to} \quad y_i (w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* **$C$ Parameter**: Controls the trade-off between maximizing the decision margin and minimizing training classification errors.
  * *High $C$*: Harder margin, fewer misclassifications, potential overfitting risk.
  * *Low $C$*: Softer margin, permits more margin violations, smoother decision surface.

### 3. Radial Basis Function (RBF) Kernel
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
* Maps input data into an infinite-dimensional Hilbert space to separate non-linearly separable boundaries.

---

## 📊 Dataset

* **Source**: Scikit-Learn built-in Wisconsin Breast Cancer dataset (`sklearn.datasets.load_breast_cancer`).
* **Instances**: 569 samples (212 Malignant, 357 Benign).
* **Dimensionality**: 30 real-valued continuous features computed from digitized images of fine needle aspirates (FNA) of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity).

---

## 🛠️ Tech Stack & Requirements

* **Python 3.x**
* **Scikit-Learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RishiRaj1112/Support-Vector-Machine-SVM.git
   cd Support-Vector-Machine-SVM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook SVM.ipynb
   ```

---

## 📈 Key Insights

* **Feature Scaling is Essential**: Because SVM relies on Euclidean distances to define margins, features with large variances would otherwise dominate the objective function without `StandardScaler`.
* **High Dimensional Efficacy**: SVM excels in high-dimensional domains (such as genomic and diagnostic data) where the number of dimensions may approach the sample size.

---

## 👤 Author
* **Rishi Raj Talari** - [GitHub Profile](https://github.com/RishiRaj1112)
