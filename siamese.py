import tensorflow as tf
import tflearn


class Siamese:

    def __init__(self):
        self.x1 = tflearn.input_data([None, 466616])
        self.x2 = tflearn.input_data([None, 466616])

        with tf.variable_scope("siamese") as scope:
            self.network1 = self.network(self.x1)
            scope.reuse_variables()
            self.network2 = self.network(self.x2)

        self.y_ = tf.placeholder(tf.float32, [None])

    def network(self, x):
        net = x
        net = tflearn.fully_connected(net, 32, activation='relu')
        net = tflearn.fully_connected(net, 32, activation='relu')
        net = tflearn.fully_connected(net, 2, activation='relu')
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
        return loss