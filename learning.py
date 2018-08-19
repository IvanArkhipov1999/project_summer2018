import tflearn

import siamese
import dataset

data = dataset.Dataset()
x1, x2, y = data.random("C:/Users/Мой Господин/PycharmProjects/practice_task1/data_odometry_gray/dataset/sequences/00/image_0",
                        "C:/Users/Мой Господин/PycharmProjects/practice_task1/00.txt", 100)

network = siamese.Siamese()
regression = tflearn.regression(network.loss(), learning_rate=0.002)
model = tflearn.DNN(regression)
model.fit([x1, x2], y)
