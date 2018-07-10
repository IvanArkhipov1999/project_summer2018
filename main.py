import numpy as np
import math as m
import matplotlib.pyplot as plt

def transform_12x1_to_3x4(data):
    transformed_data = []

    for i in data:
        transformed_data.append(i.reshape(3, 4))
    return transformed_data

def list_of_similars(data, n, m):
    similars = []

    for i in range(len(data)):
        for j in range(i, len(data)):
            rotation_vector = np.array([0, 0, 1])
            matrixi = data[i][:3, :3]
            biasi = data[i][:, 3]
            matrixj = data[j][:3, :3]
            biasj = data[j][:, 3]
            if np.linalg.norm(biasi - biasj) < 2 \
                    and np.dot(np.dot(matrixi, rotation_vector), np.dot(matrixj, rotation_vector)) / \
                    (np.linalg.norm(np.dot(matrixi, rotation_vector)) * \
                     np.linalg.norm(np.dot(matrixj, rotation_vector))) > 0.9:
                similars.append((i, j))
    return similars

def evaluation(alg_result, true_result):
    auc = 0.0
    roc_x = []
    roc_y = []
    min_distance = alg_result.min
    max_distance = alg_result.max
    t = min_distance
    step = 0.1
    FP = 0
    TP = 0
    P = len(true_result)
    N = m.factorial(alg_result.shape[0])/(m.factorial(2) * m.factorial(alg_result.shape[0] - 2)) - P

    while t <= max_distance:
        for i in range(alg_result.shape[0]):
            for j in range(i, alg_result.shape[1]):
                if alg_result[i, j] >= t:
                    if true_result.count((i, j)) == 0:
                        FP = FP + 1
                    else:
                        TP = TP + 1
        roc_x.append(FP / float(N))
        roc_y.append(TP / float(P))
        FP = 0
        TP = 0
        t = t + step
    plt.plot(roc_x, roc_y)
    plt.show()
    return auc

data = np.loadtxt("00.txt", dtype=np.float)