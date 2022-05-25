from utils import test_postfix_dir, test_and_make_dir, currentTime, json_dump
from sklearn.cluster import KMeans, MiniBatchKMeans
import joblib
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from collections import Counter
import cv2


class SPM:

    def __init__(self, M = 200, L = 2, cluster_batch = 1000, do_train_clusters = True, do_train_classifier = True,
        max_num_sample_train_cluster = 20000, save_root = "save", use_matrix_in_kernel = False, save_sift_feature = True):
        self.save_root = test_postfix_dir(save_root)
        test_and_make_dir(self.save_root)

        self.do_train_clusters = do_train_clusters
        self.do_train_classifier = do_train_classifier
        self.save_sift_feature = save_sift_feature

        self.use_matrix_in_kernel = use_matrix_in_kernel

        self.L = L
        self.M = M
        self.num_cells = (4 **(L + 1) - 1) // 3
        self.sp_dim = M * self.num_cells # dim of spatial-pyramid-feature

        # self.clusters = KMeans(M) # clusters as vocabulary
        self.clusters_batch = cluster_batch
        self.clusters = MiniBatchKMeans(M, batch_size=self.clusters_batch)
        # self.classifier = SVC(kernel="precomputed")
        self.classifier = SVC(kernel=self.spatial_pyramid_matching_kernel)
        self.MAX_NUM_SAMPLE_TRAIN_CLUSTER  = max_num_sample_train_cluster # maximum number of training samples in training KMeans

    def _set_feature_hw(self, images):
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

    def feature_dense_sift(self, images):
        """ Extract Dense sift features from a batch of images
        """
        sift = cv2.SIFT_create()
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

        kp = [cv2.KeyPoint(x, y, patch_size[0]) for y in coord_y
              for x in coord_x]
        features = np.zeros([images.shape[0], self.num_feature_h * self.num_feature_w, 128])
        for i, image in enumerate(tqdm(images)):
            _, dense_sift = sift.compute(image, kp)
            features[i] = dense_sift

        return features

    def spatial_pyramid_matching_kernel(self, x, y):
        """ spatial pyramid matching kernel function of svm,
            calculate the matching score of two vector
        """
        num_samples_x, num_features = x.shape
        num_samples_y, num_features = y.shape

        x = x.reshape(-1, self.num_feature_h, self.num_feature_w)
        y = y.reshape(-1, self.num_feature_h, self.num_feature_w)

        x_feature = np.zeros((num_samples_x, self.num_cells, self.M))
        y_feature = np.zeros((num_samples_y, self.num_cells, self.M))

        # extract feature
        count = 0
        for level in range(0, self.L + 1):
            if level == 0:
                coef = 1 / (2 ** self.L)
            else:
                coef = 1 / (2 ** (self.L + 1 - level))

            num_level_cells = 4 ** level
            num_segments = 2 ** level
            cell_hs = np.linspace(0, self.num_feature_h, 2 ** level + 1, dtype=int)
            cell_ws = np.linspace(0, self.num_feature_w, 2 ** level + 1, dtype=int)

            # cells in each level
            for cell in range(num_level_cells):

                idx_y = cell // num_segments
                idx_x = cell % num_segments

                # histogram of BOF in one cell
                x_block_feature = x[:, cell_hs[idx_y]:cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_x):
                    counter = Counter(x_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype=int)
                    level_feature = np.zeros((self.M,))  # 注意可能在cell内有未出现的类
                    if counts.size != 0:
                        level_feature[counts[:, 0]] = counts[:, 1]
                    x_feature[s, count, :] = level_feature * coef

                y_block_feature = y[:, cell_hs[idx_y]:cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_y):
                    counter = Counter(y_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype=int)
                    level_feature = np.zeros((self.M,))
                    if counts.size != 0:
                        level_feature[counts[:, 0]] = counts[:, 1]
                    y_feature[s, count, :] = level_feature * coef

                count += 1

        x_feature = x_feature.reshape(num_samples_x, -1)
        y_feature = y_feature.reshape(num_samples_y, -1)

        # 此处直接用matrix计算的话两个repeat的内存占用会非常大，仅能对非常小的样本使用
        # 如果改成循环内存占用会小很多，但是牺牲时间效率
        if self.use_matrix_in_kernel:
            xf = x_feature.reshape(num_samples_x, 1, -1).repeat(num_samples_y, axis=1)

            yf = y_feature.reshape(num_samples_y, 1, -1).repeat(num_samples_x, axis=1)
            yf = np.transpose(yf, (1, 0, 2))
            t = np.min([xf, yf], axis=0).sum(axis=-1)
        else:
            t = np.zeros((num_samples_x, num_samples_y))
            for i in range(num_samples_x):
                for j in range(num_samples_y):
                    a = x_feature[i, :]
                    b = y_feature[j, :]
                    t[i][j] = np.min([a, b], axis=0).sum(axis=-1)

        return t

    def save_model(self):
        joblib.dump(self.clusters, self.save_root + "clusters_" + currentTime() + ".pt")
        joblib.dump(self.classifier, self.save_root + "classifier_" + currentTime() + ".pt")
