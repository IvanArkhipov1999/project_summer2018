import tensorflow as tf

import siamese
import dataset

data = dataset.Dataset()
# x1, x2, y = data.random("C:/Users/Мой Господин/PycharmProjects/practice_task1/data_odometry_gray/dataset/sequences/00/image_0",
#                          "C:/Users/Мой Господин/PycharmProjects/practice_task1/00.txt", 100)
# print(x1[0].shape[0])
sess = tf.InteractiveSession()
network = siamese.Siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(network.loss())
tf.initialize_all_variables().run()
for step in range(50000):
    x1, x2, y = data.random("C:/Users/Мой Господин/PycharmProjects/practice_task1/data_odometry_gray/dataset/sequences/00/image_0",
                        "C:/Users/Мой Господин/PycharmProjects/practice_task1/00.txt", 100)
    _, loss_v = sess.run([train_step, network.loss], feed_dict={
        network.x1: x1,
        network.x2: x2,
        network.y_: y})
    if step % 10 == 0:
        print('step %d: loss %.3f' % (step, loss_v))


