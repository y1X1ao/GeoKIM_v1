import numpy as np
from sklearn.cluster import KMeans

def spatial_split_kmeans(coords, labels, n_clusters=4, seed=42):
   
    label_mask = np.isin(labels, [-1, 1])
    coords_labeled = coords[label_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_ids = kmeans.fit_predict(coords_labeled)

    region_id = np.full(len(labels), -1)
    region_id[label_mask] = cluster_ids
    return region_id
