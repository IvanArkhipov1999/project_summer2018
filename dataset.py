import numpy as np
import glob
import random
from PIL import Image

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
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)

        for i in range(size_of_batch):
            numbers1.append(random.randint(0, len(list_of_pictures)))
            pic = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            x1.append(np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3))
            
        for i in range(size_of_batch):
            numbers2.append(random.randint(0, len(list_of_pictures)))
            pic = Image.open(list_of_pictures[numbers1[len(numbers2) - 1]])
            x2.append(np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3))

        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return x1, x2, y

    def random_rotated(self, path_pictures, path_poses_file, size_of_batch):
        x1 = []
        x2 = []
        y = np.empty((size_of_batch, size_of_batch))
        numbers1 = []
        numbers2 = []

        list_of_pictures = glob.glob(path_pictures)
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)

        for i in range(size_of_batch):
            numbers1.append(random.randint(0, len(list_of_pictures)))
            pic = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            pic_rotated = pic.rotate(random.randint(0, 360))
            x1.append(np.array(pic_rotated.getdata()).reshape(pic_rotated.size[0], pic_rotated.size[1], 3))

        for i in range(size_of_batch):
            numbers2.append(random.randint(0, len(list_of_pictures)))
            pic = Image.open(list_of_pictures[numbers2[len(numbers2) - 1]])
            pic_rotated = pic.rotate(random.randint(0, 360))
            x2.append(np.array(pic_rotated.getdata()).reshape(pic_rotated.size[0], pic_rotated.size[1], 3))

        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return x1, x2, y

    def verification(self, path_pictures, path_poses_file, size_of_batch):
        x1 = []
        x2 = []
        y = np.empty((size_of_batch, size_of_batch))
        numbers1 = []
        numbers2 = []

        list_of_pictures = glob.glob(path_pictures)
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)
        rand = random.randint(0, len(list_of_pictures) - size_of_batch)

        for i in range(rand, rand + size_of_batch):
            numbers1.append(i)
            pic = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            x1.append(np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3))

        rand = random.randint(0, len(list_of_pictures) - size_of_batch)
        for i in range(rand, rand + size_of_batch):
            numbers2.append(i)
            pic = Image.open(list_of_pictures[numbers2[len(numbers2) - 1]])
            x2.append(np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3))

        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return x1, x2, y