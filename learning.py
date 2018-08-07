import tensorflow as tf
import tflearn
import numpy as np


import siamese
import truth_roc


y = np.loadtxt("00.txt", dtype=np.float)
Y = truth_roc.matrix_of_similar(truth_roc.transform_12x1_to_3x4(y), 1, 1)

network = siamese.Siamese()
regression = tflearn.regression(network.loss(),  loss='contrastive_loss', learning_rate=0.002)
model = tflearn.DNN(network)
#model.fit([X, X], Y)
