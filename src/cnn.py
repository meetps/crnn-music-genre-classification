import sys
import pickle as pk
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
import read_data as rd

batch_size    = 4
learning_rate = 0.003
n_epoch       = 50
n_samples     = 10                              # change to 1000 for entire dataset
cv_split      = 0.8                             
train_size    = int(n_samples * cv_split)                               
test_size     = n_samples - train_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def cnn(melspectrogram, weights, phase_train):

    x = tf.reshape(melspectrogram,[-1,1,96,1366])
    x = batch_norm(melspectrogram, 1366, phase_train)
    x = tf.reshape(melspectrogram,[-1,96,1366,1])
    conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    conv2_1 = tf.nn.relu(batch_norm(conv2_1, 32, phase_train))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
    conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
    conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
    conv2_4 = tf.nn.relu(batch_norm(conv2_4, 192, phase_train))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
    conv2_5 = tf.nn.relu(batch_norm(conv2_5, 256, phase_train))
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat,weights['woutput']),weights['boutput']))

    return p_y_X


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]

if __name__ == '__main__':

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices  = indices[train_size:]

    labels = rd.get_labels()

    X_test = rd.get_melspectrograms_indexed(test_indices)
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    weights = {
        'wconv1':init_weights([3, 3, 1, 32]),
        'wconv2':init_weights([3, 3, 32, 128]),
        'wconv3':init_weights([3, 3, 128, 128]),
        'wconv4':init_weights([3, 3, 128, 192]),
        'wconv5':init_weights([3, 3, 192, 256]),
        'bconv1':init_biases([32]),
        'bconv2':init_biases([128]),
        'bconv3':init_biases([128]),
        'bconv4':init_biases([192]),
        'bconv5':init_biases([256]),
        'woutput':init_weights([256, 10]),
        'boutput':init_biases([10])}

    X = tf.placeholder("float", [None, 96, 1366, 1])
    y = tf.placeholder("float", [None, 10])
    lrate = tf.placeholder("float")
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    y_ = cnn(X, weights, phase_train)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
    train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    predict_op = y_

    tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(n_epoch):
            training_batch = zip(range(0, train_size, batch_size),range(batch_size, train_size+1, batch_size))
            for start, end in training_batch:
                X_train = rd.get_melspectrograms_indexed(train_indices[start:end])
                train_input_dict = {X: X_train, 
                                    y: y_train[start:end],
                                    lrate: learning_rate,
                                    phase_train: True}
                sess.run(train_op, feed_dict=train_input_dict)

            test_indices = np.arange(len(X_test))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            test_input_dict = {X: X_test[test_indices],
                               y: y_test[test_indices],
                               phase_train:True}
            predictions = sess.run(predict_op, feed_dict=test_input_dict)
            print('Epoch : ', i,  'AUC : ', sm.roc_auc_score(y_test[test_indices], predictions, average='samples'))
            # print(i, np.mean(np.argmax(y_test[test_indices], axis=1) == predictions))
            # print sort_result(tags, predictions)[:5]
