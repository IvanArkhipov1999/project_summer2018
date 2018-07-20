import cmath
import numpy as np
import matplotlib.pyplot as plt


def transform_12x1_to_3x4(data):
    transformed_data = []

    for i in data:
        transformed_data.append(i.reshape(3, 4))
    return transformed_data


def list_of_similar(data, n, m):
    similar = np.zeros((len(data), len(data)))

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
                similar[i][j] = 1
    return similar


def visualisation_roc(roc_x, roc_y):
    plt.plot(roc_x, roc_y)
    plt.show()


def evaluation(alg_result, true_result):
    alg_result_reshaped = np.reshape(alg_result,
                                     (1, alg_result.shape[0] * alg_result.shape[0]))
    true_result_reshaped = np.reshape(true_result,
                                      (1, true_result.shape[0] * true_result.shape[0]))
    result = np.reshape(np.dstack([alg_result_reshaped, true_result_reshaped]),
                        (alg_result.size, 2))
    result = np.asarray(sorted(result, key=lambda item: item[0]))
    P = np.sum(true_result)
    N = true_result.size - P
    a = np.cumsum(result[:, 1])
    b = np.arange(alg_result.size) + 1
    roc_x = (b - a) / float(N)
    roc_y = a / float(N)
    visualisation_roc(roc_x, roc_y)


dataset = np.loadtxt("00.txt", dtype=np.float)