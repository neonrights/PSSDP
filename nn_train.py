import tensorflow as tf
import pandas as pd


def init_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def init_model(x, y_, layer_size):
    weights = list()
    biases = list()
    layers = list()

    for i, size in enumerate(layer_size):
        if i == 0: # first hidden layer
            weights.append(init_weight([x._shape_as_list()[1], size]))
            biases.append(init_bias([size]))
            layers.append(tf.matmul(x,weights[-1]) + biases[-1])
        else: # hidden layers in between
            weights.append(init_weight([layers[-1].get_shape().as_list()[1], size]))
            biases.append(init_bias([size]))
            layers.append(tf.matmul(layers[-1],weights[-1]) + biases[-1])
    
    # last hidden layer
    weights.append(init_weight([layers[-1].get_shape().as_list()[1], y_._shape_as_list()[1]]))
    biases.append(init_weight([y_._shape_as_list()[1]]))
    layers.append(tf.matmul(layers[-1],weights[-1]) + biases[-1])
    return weights, biases, layers


if __name__ == '__main__':
    train = pd.read_csv("data/p-train.csv")

    # create model and placeholders
    x = tf.placeholder(tf.float32, shape=[None, len(train.columns)-2])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    weights, biases, hidden = init_model(x, y_, [30, 20, 30])
    y_nn = tf.sigmoid(hidden[-1]) # TODO: dropout? regularization?

    # define error (check best for GINI score)
    loss = -(y_ * tf.log(y_nn + 1e-12) + (1 - y_) * tf.log(1 - y_nn + 1e-12))
    cross_entropy = tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    # TODO implement GINI and f1 score

    # train model
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            for chunk in pd.read_csv("data/p-train.csv"): 
                # get next chunk
                if i % 100 == 0:
                    train_accuracy = cross_entropy.eval(feed_dict={x : chunk.drop('target').drop('id'), y : chunk['target']})

                train_step.run(feed_dict={x : chunk[~'target'], y : chunk['target']})

    # output training accuracy
    print('test cross entropy %g' % cross_entropy.eval(feed_dict={x : train[~'target'], y : train['target']}))


