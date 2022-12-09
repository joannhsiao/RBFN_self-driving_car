import numpy as np

def get_distance(c, x):
    sum = 0
    for i in range(len(c)):
        sum += (c[i] - x[i]) ** 2
    return np.sqrt(sum)

def kmeans(X, k, max_iters=10000):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []
        for j in range(len(cluster_list)):
            # use the mean of cluster to re-calculate the new centroid
            centroids.append(np.mean(cluster_list[j], axis=0))
        # pattern: test whether it's the same centroid
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        converged = (pattern == 0)
        current_iter += 1
    
    """
    std = []
    for i in range(len(centroids)):
        s = 0
        for x in cluster_list[i]:
            s += np.asscalar(get_distance(centroids[i], x))
        s /= len(cluster_list[i])
        s = np.sqrt(s)
        std.append(s)
    """

    return np.array(centroids), np.array([[np.std(x) for x in cluster_list]])
    #return np.array(centroids), np.array([std])