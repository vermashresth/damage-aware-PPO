import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_network(convs,
                  hiddens,
                  lstm,
                  inpt,
                  rnn_state_tuple,
                  num_actions,
                  lstm_unit,
                  nenvs,
                  step_size,
                  continuous,
                  scope='network',
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride, padding in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID',
                    activation_fn=tf.nn.relu
                )
            out = layers.flatten(out)

        with tf.variable_scope('hiddens')
            for hidden in (hiddens):
                out = layers.fully_connected(
                    out, hidden, activation_fn=tf.nn.tanh)

        with tf.variable_scope('rnn'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_unit, state_is_tuple=True)
            # sequence to batch
            rnn_in = tf.reshape(out, [nenvs, step_size, lstm_unit])
            sequence_length = tf.ones(nenvs, dtype=tf.int32) * step_size
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=rnn_state_tuple,
                sequence_length=sequence_length, time_major=False)
            # batch to sequence
            rnn_out = tf.reshape(lstm_outputs, [-1, lstm_unit])

        if lstm:
            out = rnn_out

        # policy branch
        if continous:
            mu = layers.fully_connected(
                out, num_actions, activation_fn=None, name='mu')
            mu = tf.nn.tanh(mu + 1e-20)

            sigma = layers.fully_connected(
                out, num_actions, activation_fn=None, name='sigma')
            sigma = tf.nn.softplus(sigma + 1e-20)

            dist = tf.distributions.Normal(mu, sigma)
            policy = tf.squeeze(dist.sample(num_actions), [0])
        else:
            probs = layers.fully_connected(
                out, num_actions, activation_fn=tf.nn.softmax)
            dist = tf.distributions.Categorical(probs=probs)
            policy = tf.squeeze(dist.sample(1), [0])

        # value branch
        value = layers.fully_connected(out, 1, activation_fn=None)
    return policy, value, dist

def make_network(convs, hiddens, lstm):
    return lambda *args, **kwargs: _make_network(convs, hiddens, lstm, *args, **kwargs)
