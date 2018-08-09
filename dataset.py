import numpy as np
import glob
import random
from sklearn.datasets import load_sample_image

import truth_roc

class Dataset:

    def random(self, path_pictures, path_poses_file, size_of_batch):
        x1 = []
        x2 = []
        y = np.empty((size_of_batch, size_of_batch))
        numbers1 = []
        numbers2 = []

        list_of_pictures = glob.glob(path_pictures)
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)))

        for i in range(size_of_batch):
            numbers1.append(random.randint(0, len(list_of_pictures)))
            x1.append(load_sample_image(list_of_pictures[numbers1[-1]]))
        for i in range(size_of_batch):
            numbers2.append(random.randint(0, len(list_of_pictures)))
            x2.append(load_sample_image(list_of_pictures[numbers2[-1]]))
        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return x1, x2, y
