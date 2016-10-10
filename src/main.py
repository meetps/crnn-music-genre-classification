import sys
import tensorflow as tf
import data_preprocess as dp


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


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


def conv_net(melspectrogram, weights):
    
    x = batch_norm(melspectrogram, n_out=1966)
    conv2_1 = tf.nn.elu(batch_norm(tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME'), 1966))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2 = tf.nn.elu(batch_norm(tf.nn.conv2d(dropout_1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME'), 1966))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3 = tf.nn.elu(batch_norm(tf.nn.conv2d(dropout_2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME'), 1966))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4 = tf.nn.elu(batch_norm(tf.nn.conv2d(dropout_3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME'), 1966))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='SAME')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    conv2_5 = tf.nn.elu(batch_norm(tf.nn.conv2d(dropout_4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME'), 1966))
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    flat = tf.reshape(dropout_5, [-1, weights['wf1'].get_shape().as_list()[0]])
    p_y_X = tf.nn.sigmoid(tf.matmul(flat,weights['wf1']))
    
    return p_y_X

if __name__ == '__main__':
    weights = {'wc1': init_weights([3, 3, 1, 32]),
               'wc2': init_weights([3, 3, 32, 128]), 
               'wc3': init_weights([3, 3, 128, 128]), 
               'wc4': init_weights([3, 3, 128, 192]), 
               'wc5': init_weights([3, 3, 192, 256]), 
               'wf1': init_weights([7*7*256, 50])}
    
    audio_sample_path = sys.argv[1]
    show_image = sys.argv[2]
    dp.log_scale_melspectrogram(audio_sample_path, show_image)
