import numpy as np
import matplotlib.pyplot as plt

datas = [np.loadtxt("00.txt", dtype=np.float), np.loadtxt("01.txt", dtype=np.float),
         np.loadtxt("02.txt", dtype=np.float), np.loadtxt("03.txt", dtype=np.float),
         np.loadtxt("04.txt", dtype=np.float), np.loadtxt("05.txt", dtype=np.float),
         np.loadtxt("06.txt", dtype=np.float), np.loadtxt("07.txt", dtype=np.float),
         np.loadtxt("08.txt", dtype=np.float), np.loadtxt("09.txt", dtype=np.float),
         np.loadtxt("10.txt", dtype=np.float)]
x = []
y = []
x1 = []
y1 = []
whattocheck = 0

for j in range(datas[0].shape[0]):
    x.append(datas[0][j][3])
    y.append(datas[0][j][7])
plt.scatter(x, y, color='gray', s=1)

for i in range(len(datas)):
    for j in range(datas[i].shape[0]):
        if j == whattocheck:
            plt.scatter(datas[i][j][3], datas[i][j][7], color='green', s=10)
        for k in range(j, datas[i].shape[0]):
            rotation_vector = np.array([1, 1, 1])
            matrixj = np.array([[datas[i][j][0], datas[i][j][1], datas[i][j][2]],
                               [datas[i][j][4], datas[i][j][5], datas[i][j][6]],
                               [datas[i][j][8], datas[i][j][9], datas[i][j][10]]])
            biasj = np.array([datas[i][j][3], datas[i][j][7], datas[i][j][11]])
            matrixk = np.array([[datas[i][k][0], datas[i][k][1], datas[i][k][2]],
                               [datas[i][k][4], datas[i][k][5], datas[i][k][6]],
                               [datas[i][k][8], datas[i][k][9], datas[i][k][10]]])
            biask = np.array([datas[i][k][3], datas[i][k][7], datas[i][k][11]])
            if np.linalg.norm(biask - biasj) < 2 \
                    and np.dot(np.dot(matrixj, rotation_vector), np.dot(matrixk, rotation_vector)) / \
                    (np.linalg.norm(np.dot(matrixj, rotation_vector)) * \
                    np.linalg.norm(np.dot(matrixk, rotation_vector))) > 0.9:
                print(i, j, k)
                if j == whattocheck and k != j:
                    x1.append(datas[i][k][3])
                    y1.append(datas[i][k][7])
        if j == whattocheck:
            plt.scatter(x1, y1, color='red', s=10)
            plt.show()