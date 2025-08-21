import numpy as np

from .misc import find_center

def find_rectangles_containing_point(rectangles, point):
    """
    Return all indices of rectangles that contain the given point.
    """
    containing_indices = [
        i for i, (x1, y1, x2, y2) in enumerate(rectangles)
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2
    ]
    return containing_indices


def which_cluster(y_values, x_values, peaks_df, cluster_rectangles):
    """
    Find the indices of rectangles containing the point.
    """
    y_index = np.where(y_values == peaks_df['RT2_center'])[0][0]
    x_index = np.where(x_values == peaks_df['RT1_center'])[0][0]
    return find_rectangles_containing_point(cluster_rectangles, (y_index, x_index))


def find_clusters_within_threshold(points, threshold_x, threshold_y):
    """
    Find clusters of points where the maximum distance in x and y within each cluster is within certain thresholds,
    and return the bounding rectangles for these clusters.
    """
    clusters = []
    included = set()

    for i in range(len(points)):
        if i in included:
            continue

        current_cluster = {i}
        queue = [i]

        while queue:
            point = queue.pop(0)
            for j in range(len(points)):
                if j not in included:
                    dist_x = abs(points[point][0] - points[j][0])
                    dist_y = abs(points[point][1] - points[j][1])
                    if dist_x <= threshold_x and dist_y <= threshold_y:
                        if all(
                            abs(points[j][0] - points[k][0]) <= threshold_x and
                            abs(points[j][1] - points[k][1]) <= threshold_y
                            for k in current_cluster
                        ):
                            current_cluster.add(j)
                            queue.append(j)
                            included.add(j)

        clusters.append(current_cluster)

    rectangles = [
        (
            np.min(points[list(cluster)][:, 0]),
            np.min(points[list(cluster)][:, 1]),
            np.max(points[list(cluster)][:, 0]),
            np.max(points[list(cluster)][:, 1])
        )
        for cluster in clusters
    ]

    return rectangles


def find_cluster_features(filtered_binary, threshold_x=10, threshold_y=5):
    """
    Find the clusters of features in the binary matrix and return their centers and bounding rectangles.
    """
    coordinates = np.column_stack(np.where(filtered_binary))

    if len(coordinates) > 10000:
        return [], []

    cluster_rectangles = find_clusters_within_threshold(coordinates, threshold_y, threshold_x)
    cluster_centers = [find_center(cluster) for cluster in cluster_rectangles]

    return cluster_centers, cluster_rectangles