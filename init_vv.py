from matplotlib import pyplot as plt
from kmeans import *

result_feature_map = get_feature_maps()
stacked_tensor = get_stacked_feature_maps(result_feature_map)
y_km = get_kmeans_clusters(stacked_tensor)

# Too plot the centroids

# plt.scatter(
#     km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
#     s=250, marker='*',
#     c='red', edgecolor='black',
#     label='centroids'
# )