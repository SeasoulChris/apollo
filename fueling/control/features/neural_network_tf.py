import tensorflow as tf
import tensorflow.contrib
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetworkTF(object):
    """
    neural network for auto calibration
    """

    def __init__(self, layer):
        """
        init fucntion
        """
        tf.reset_default_graph()
        self.layer = layer
        self.session = tf.Session()

    def create_placeholder(self, n_x, n_y):
        """
        create placeholder
        """
        self.X = tf.placeholder(tf.float32, [None, n_x])
        self.Y = tf.placeholder(tf.float32, [None, n_y])

    def forward_propagation(self, X, params):
        """
        forward propagation
        """
        Z1 = tf.add(tf.matmul(X, params['W1']), params['b1'])
        A1 = tf.sigmoid(Z1)
        Z2 = tf.add(tf.matmul(A1, params['W2']), params['b2'])

        return Z2

    def initialize_parameters(self):
        """
        initialize weight and bias
        """
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        W1 = tf.get_variable(
            'W1', [self.layer[0], self.layer[1]], initializer=xavier_initializer)
        W2 = tf.get_variable(
            'W2', [self.layer[1], self.layer[2]], initializer=xavier_initializer)
        b1 = tf.get_variable('b1', [1, self.layer[1]],
                             initializer=tf.zeros_initializer())
        b2 = tf.get_variable('b2', [1, self.layer[2]],
                             initializer=tf.zeros_initializer())

        params = {'W1': W1,
                  'W2': W2,
                  'b1': b1,
                  'b2': b2}

        return params

    def train(self, X_train, Y_train, X_test=None, Y_test=None, alpha=0.01, w_lambda=0.01,
              num_epoch=10000, print_loss=False):
        """
        train model
        """
        (m, n_x) = X_train.shape
        (_, n_y) = Y_train.shape
        costs = []

        self.create_placeholder(n_x, n_y)
        params = self.initialize_parameters()
        self.Y_hat = self.forward_propagation(self.X, params)

        # loss function
        self.cost = tf.reduce_mean(tf.square((self.Y - self.Y_hat)))
        tf.add_to_collection(
            "losses", tf.contrib.layers.l2_regularizer(w_lambda)(params['W1']))
        tf.add_to_collection(
            "losses", tf.contrib.layers.l2_regularizer(w_lambda)(params['W2']))
        tf.add_to_collection("losses", self.cost)
        loss = tf.add_n(tf.get_collection("losses"))
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

        self.session.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            feed_dict = {self.X: X_train, self.Y: Y_train}
            _, epoch_cost = self.session.run(
                [optimizer, loss], feed_dict=feed_dict)
            if print_loss and epoch % 100 == 0:
                print("loss after epoch %d: %f" % (epoch, epoch_cost))
            if epoch % 10 == 0:
                costs.append(epoch_cost)

        # store params
        params = self.session.run(params)

        # check accuracy
        train_cost, test_cost = 0.0, 0.0
        with self.session.as_default():
            train_cost = self.cost.eval(
                feed_dict={self.X: X_train, self.Y: Y_train})
            if X_test is not None and Y_test is not None:
                test_cost = self.cost.eval(
                    feed_dict={self.X: X_test, self.Y: Y_test})

        return params, train_cost, test_cost

    def predict(self, X_data, Y_data=None):
        """
        predict output, if Y is provided, return cost; else, return predicted Y
        """
        if Y_data is not None:
            with self.session.as_default():
                test_cost = self.cost.eval(
                    feed_dict={self.X: X_data, self.Y: Y_data})
                return test_cost
        else:
            Y_pred = self.session.run(self.Y_hat, feed_dict={self.X: X_data})
        return Y_pred
