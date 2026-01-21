import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import make_pipeline


#1. DATA UPLOAD (Via URL)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

column_names=[
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280", "proline"
]

df= pd.read_csv(url, names=column_names)
wine_features= df.drop("class", axis=1)


# 2. PREPROCESSING AND PIPELINE
scaler= StandardScaler()


# 3. K-MEANS AND ELBOW METHOD (Finding the Optimal Number of Clusters)
ks= range(1,10)
inertias=[]
for k in ks:
    model= KMeans(n_clusters=k)
    pipeline= make_pipeline(scaler, model)
    pipeline.fit(wine_features)
    inertias.append(model.inertia_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks, inertias, '-o')
plt.title('Elbow Method (Optimal K Selection)') # "Elbow" point is selected [cite: 233, 234]
plt.xlabel('Number of Clusters (k)')
plt.xlabel('Inertia')


# 4. PCA and Intrinsic Dimension Analysis
pca= PCA()
scaled_features= scaler.fit_transform(wine_features)
pca.fit(scaled_features)

plt.subplot(1, 2, 2)
plt.bar(range(pca.n_components_), pca.explained_variance_) # Features with high variance are preserved [cite: 1531, 1532]
plt.title('Intrinsic Dimension (Variance Graph)')
plt.xlabel('PCA Components')
plt.show()


# 5. Hierarchical Grouping (Dendrogram)
mergings = linkage(scaled_features, method='complete')
plt.figure(figsize=(10, 5))
dendrogram(mergings, leaf_rotation=90, leaf_font_size=6) # Dikey çizgilerin yüksekliği mesafeyi gösterir [cite: 878, 901]
plt.title('Wine Hierarchy Dendrogram')
plt.show()


# 6. 2D VISUALIZATION WITH t-SNE
tsne = TSNE(learning_rate=100)
tsne_features = tsne.fit_transform(scaled_features) 

plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=df['class'])
plt.title('t-SNE Visualization')
plt.show()


# 7. NMF AND RECOMMENDATION ENGINE
nmf = NMF(n_components=3) 
nmf_features = nmf.fit_transform(wine_features)

norm_features = normalize(nmf_features)
df_recommender = pd.DataFrame(norm_features, index=df.index)

def recommend(wine_id, df_input, count=3):
    current_wine = df_input.loc[wine_id]
    similarities = df_input.dot(current_wine)
    return similarities.nlargest(count + 1)

print("\n--- Recommendations for Wine ID 0 ---")

print(recommend(0, df_recommender))