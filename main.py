import cmath
import scipy
import numpy as np
import matplotlib.pyplot as plt


def transform_12x1_to_3x4(data):
    transformed_data = []

    for i in data:
        transformed_data.append(i.reshape(3, 4))
    return transformed_data


def list_of_similar(data, n, m):
    similar = []

    for i in range(len(data)):
        for j in range(i, len(data)):
            rotation_vector = np.array([0, 0, 1])
            matrixi = data[i][:3, :3]
            biasi = data[i][:, 3]
            matrixj = data[j][:3, :3]
            biasj = data[j][:, 3]
            if np.linalg.norm(biasi - biasj) < n \
                and cmath.acos(np.dot(np.dot(matrixi, rotation_vector),
                               np.dot(matrixj, rotation_vector)) /
                               (np.linalg.norm(np.dot(matrixi, rotation_vector)) *
                               np.linalg.norm(np.dot(matrixj, rotation_vector)))) < m:
                similar.append((i, j))
    return similar


def visualisation_roc(roc_x, roc_y):
    plt.plot(roc_x, roc_y)
    plt.show()


# alg_result - list of (distance, picture1, picture2)
def evaluation(alg_result, true_result):
    roc_x = []
    roc_y = []
    FP = 0
    TP = 0
    P = len(true_result)
    N = len(alg_result) - P

    alg_result.sort(key=lambda item: item[0])
    for i in alg_result:
        if true_result.count((i[1], i[2])) == 0:
            FP = FP + 1
        else:
            TP = TP + 1
        roc_x.append(FP / float(N))
        roc_y.append(TP / float(P))
    visualisation_roc(roc_x, roc_y)
    return scipy.trapz(roc_y, roc_x)


dataset = np.loadtxt("00.txt", dtype=np.float)