
import numpy as np
from scipy import stats
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import cv2


def mean_shift_cluster(img):

    height, width = img.shape[0:2]
    b, g, r = cv2.split(img)
    brg_extract = r
    bgr_img_for_cluster = np.transpose(np.vstack((b.ravel(), g.ravel(), r.ravel())))

    # Flatten Image
    nr_color_cols = 3
    if len(img.shape) < 3:
        nr_color_cols = 1

    bgr_img_for_cluster = img.reshape((-1, nr_color_cols))

    # # Normalize Features
    scaler = StandardScaler()
    bgr_img_for_cluster = scaler.fit_transform(bgr_img_for_cluster)

    # meanshift
    n_tot_samples = bgr_img_for_cluster.shape[0]
    bw_search_samples = 3000  # int(n_tot_samples*0.01)
    bandwidth = estimate_bandwidth(
        bgr_img_for_cluster, quantile=.06, n_samples=bw_search_samples)
    bandwidth = 1
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True,
                      max_iter=1000, n_jobs=-1)
    model.fit(bgr_img_for_cluster)
    labeled = model.labels_

    # get number of segments
    labels = np.unique(labeled)
    segments = labels
    print('Number of segments: ', segments.shape[0])

    # Assign the aggregated color to each segment

    bgr_img_for_cluster = scaler.inverse_transform(bgr_img_for_cluster)
    flat_segm_image = np.zeros(bgr_img_for_cluster.shape)
    for i, label in enumerate(labels):
        row_sel = labeled == label
        cluster_color = np.mean(bgr_img_for_cluster[row_sel, :], axis=0)
        flat_segm_image[row_sel, :] = cluster_color

    flat_segm_image = np.uint8(flat_segm_image)
    segmented_image = flat_segm_image.reshape((img.shape))

    return segmented_image, flat_segm_image, labeled