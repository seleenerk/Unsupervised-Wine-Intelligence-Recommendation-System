# 🍷 Unsupervised Wine Intelligence & Recommendation System

This project is a comprehensive study of **Unsupervised Learning** techniques, demonstrating how to discover hidden patterns in data without the need for pre-defined labels. By analyzing the chemical composition of various wines, the system performs clustering, dimensionality reduction, and builds a content-based recommendation engine.

---

## 🌟 Key Features

The project implements a full data science pipeline using industry-standard techniques:

* **Feature Standardization (`StandardScaler`):** Ensures that features with high variance do not dominate the model, providing a balanced analysis across all chemical attributes.
* **Optimal Clustering (`K-Means` & `Elbow Method`):** Mathematically determines the natural number of groups within the dataset by measuring inertia.
* **Intrinsic Dimension Analysis (`PCA`):** Identifies the core components that capture the most information, effectively removing noise from the 13-dimensional data.
* **Manifold Learning (`t-SNE`):** Maps complex, multi-dimensional data into a 2D scatter plot to visualize how different wine varieties naturally cluster together.
* **Intelligent Recommendation Engine (`NMF` & `Cosine Similarity`):** Decomposes wine features into "interpretable topics" and recommends products by calculating the angular similarity between chemical profiles.

---

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (PCA, NMF, KMeans, t-SNE) |
| **Visualization** | Matplotlib |
| **Scientific Computing** | SciPy |

---

## 📂 Project Workflow

The execution follows a modular structure within `wine.py`:

1. **Data Acquisition:** Fetches real-world wine data from the UCI Machine Learning Repository via URL.
2. **Preprocessing:** Scales features to a uniform range for accurate distance calculations.
3. **Exploratory Visualizations:**
    * `Elbow Plot`: To find the "sweet spot" for the number of clusters.
    * `PCA Variance`: To visualize how much information each component holds.
    * `Dendrogram`: To inspect the hierarchical relationships between samples.
    * `t-SNE Map`: To project clusters into a 2D visual space.
4. **Recommendation Engine:** Calculates similarity scores for a given Wine ID and returns the top matches.

---

## 💻 Installation & Usage

### 1. Install Dependencies
Ensure you have the necessary libraries installed:
```bash
pip install pandas scikit-learn matplotlib scipy


