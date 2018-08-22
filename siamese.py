import tensorflow as tf
import tflearn


class Siamese:

    def __init__(self):
        self.x1 = tflearn.input_data([None, 466616])
        self.x2 = tflearn.input_data([None, 466616])


        self.network1 = self.network1(self.x1)
        self.network2 = self.network2(self.x2)

        self.y_ = tf.placeholder(tf.float32, [None])

    def network1(self, x):
        with tf.name_scope('siamese_net1'):
            net = x
            net = tflearn.fully_connected(net, 32, activation='relu')
            net = tflearn.fully_connected(net, 32, activation='relu')
            net = tflearn.fully_connected(net, 1, activation='relu')
            return net

    def network2(self, x):
        with tf.name_scope('siamese_net2'):
            net = x
            net = tflearn.fully_connected(net, 32, activation='relu')
            net = tflearn.fully_connected(net, 32, activation='relu')
            net = tflearn.fully_connected(net, 1, activation='relu')
            return net

    def loss(self):
        margin = 8.0
        y = self.y_
        m = tf.constant(margin)
        d = tf.sqrt(tf.reduce_sum(tf.pow(
            tf.subtract(self.network1, self.network2), 2), 1), name="difference")
        losses = tf.add(tf.multiply(tf.subtract(1.0, y), tf.pow(d, 2)),
                        tf.multiply(y, tf.pow(tf.maximum(0.0, tf.subtract(m, d)), 2)))
        loss = tf.reduce_mean(losses)
        loss = tf.reshape(loss, [-1])
        return loss
