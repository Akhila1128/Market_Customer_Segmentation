from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def apply_pca(data):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(scaled_data)

    return pca_data