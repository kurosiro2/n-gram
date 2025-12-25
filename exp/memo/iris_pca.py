import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. データ読み込み
iris = load_iris()
X = iris.data          # (150, 4)
y = iris.target        # 0=setosa, 1=versicolor, 2=virginica
labels = iris.target_names

# 2. 標準化（Z-score）
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. PCA（2次元）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# 4. 可視化
plt.figure(figsize=(8,6))
for c, name in zip([0,1,2], labels):
    plt.scatter(
        X_pca[y==c, 0],
        X_pca[y==c, 1],
        label=name,
        alpha=0.7
    )

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris PCA")
plt.grid(True)

# --- 追加：PNG 保存 ---
plt.savefig("iris_pca.png", dpi=300, bbox_inches='tight')


# 5. 固有値（寄与率）
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Eigenvalues:", pca.explained_variance_)
print("PC1:", pca.components_[0])
print("PC2:", pca.components_[1])
