import tensorflow as tf


class Siamese:

    def __init__(self):
        self.x1 = tf.placeholder([None, 466616])
        self.x2 = tf.placeholder([None, 466616])


        with tf.variable_scope("siamese") as scope:
            self.network1 = self.network(self.x1)
            scope.reuse_variables()
            self.network2 = self.network(self.x2)

        self.y_ = tf.placeholder(tf.float32, [None])

    def network(self, x):
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

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