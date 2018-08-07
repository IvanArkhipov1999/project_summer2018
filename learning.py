import tensorflow as tf
import tflearn


import siamese


network = siamese.Siamese()
regression = tflearn.regression(network.loss(),  loss='contrastive_loss', learning_rate=0.002)
model = tflearn.DNN(network)
model.fit([X, X], Y)
