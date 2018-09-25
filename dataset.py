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

        list_of_pictures = glob.glob(path_pictures + "/*.png")
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)

        for i in range(size_of_batch):
            numbers1.append(random.randint(0, len(list_of_pictures) - 1))
            pic1 = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            x1.append(np.array(pic1.getdata()).reshape(pic1.size[0], pic1.size[1], 1))

            numbers2.append(random.randint(0, len(list_of_pictures) - 1))
            pic2 = Image.open(list_of_pictures[numbers1[len(numbers2) - 1]])
            x2.append(np.array(pic2.getdata()).reshape(pic2.size[0], pic2.size[1], 1))

        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return np.asarray(x1), np.asarray(x2), np.reshape(y, (1, size_of_batch * size_of_batch))

    def random_rotated(self, path_pictures, path_poses_file, size_of_batch):
        x1 = []
        x2 = []
        y = np.empty((size_of_batch, size_of_batch))
        numbers1 = []
        numbers2 = []

        list_of_pictures = glob.glob(path_pictures + "/*.png")
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)

        for i in range(size_of_batch):
            numbers1.append(random.randint(0, len(list_of_pictures) - 1))
            pic1 = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            pic1_array = np.array(pic1.getdata())
            pic1_rotated = pic1.rotate(random.randint(0, 360))
            pic1_rotated_array = np.array(pic1_rotated.getdata())
            for i in range(pic1_rotated_array.shape[0]):
                if pic1_rotated_array[i] == 0:
                    pic1_rotated_array[i] = pic1_array[i]
            x1.append(pic1_rotated_array)

            numbers2.append(random.randint(0, len(list_of_pictures) - 1))
            pic2 = Image.open(list_of_pictures[numbers2[len(numbers2) - 1]])
            pic2_array = np.array(pic2.getdata())
            pic2_rotated = pic2.rotate(random.randint(0, 360))
            pic2_rotated_array = np.array(pic2_rotated.getdata())
            for i in range(pic2_rotated_array.shape[0]):
                if pic2_rotated_array[i] == 0:
                    pic2_rotated_array[i] = pic2_array[i]
            x2.append(pic2_rotated_array)

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

        list_of_pictures = glob.glob(path_pictures + "/*.png")
        matrix_of_similar = truth_roc.matrix_of_similar(
            truth_roc.transform_12x1_to_3x4(np.loadtxt(path_poses_file)), 1, 1)
        rand = random.randint(0, len(list_of_pictures) - size_of_batch)

        for i in range(rand, rand + size_of_batch):
            numbers1.append(i)
            pic = Image.open(list_of_pictures[numbers1[len(numbers1) - 1]])
            x1.append(np.array(pic.getdata()))

        rand = random.randint(0, len(list_of_pictures) - size_of_batch)
        for i in range(rand, rand + size_of_batch):
            numbers2.append(i)
            pic = Image.open(list_of_pictures[numbers2[len(numbers2) - 1]])
            x2.append(np.array(pic.getdata()))

        for i in numbers1:
            for j in numbers2:
                y[numbers1.index(i)][numbers2.index(j)] = matrix_of_similar[i][j]

        return x1, x2, y
