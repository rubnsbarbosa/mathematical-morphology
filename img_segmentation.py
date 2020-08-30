from sklearn.cluster import KMeans


def segmentation_Kmeans(image, k):
    return KMeans(random_state=1, n_clusters = k,  init='k-means++')  \
        .fit(image.reshape((-1,1))).labels_.reshape(image.shape)
