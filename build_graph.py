import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(network,
                num_actions,
                optimizer,
                nenvs,
                lstm_unit,
                state_shape=[84, 84, 1],
                value_factor=0.5,
                policy_factor=1.0
                entropy_factor=0.01,
                epsilon=0.2,
                scope='ppo',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(
            tf.float32, [None] + state_shape, name='obs_t')
        rnn_state_ph0 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_0')
        rnn_state_ph1 = tf.placeholder(
            tf.float32, [nenvs, lstm_unit], name='rnn_state_1')
        return_t_ph = tf.placeholder(tf.float32, [None], name='return')
        advantage_t_ph = tf.placeholder(tf.float32, [None], name='advantage')
        step_size_ph = tf.placeholder(tf.int32, [], name='step_size')
        mask_ph = tf.placeholder(tf.float32, [None], name='mask')
        if continous:
            act_t_ph = tf.placeholder(
                tf.float32, [None, num_actions], name='action')
        else:
            act_t_ph = tf.placeholder(tf.int32, [None], name='action')

        # rnn state in tuple
        rnn_state_tuple = tf.contrib.rnn.LSTMStateTuple(
            rnn_state_ph0, rnn_state_ph1)

        policy, value, dist = network(
            obs_t_input, rnn_state_tuple, num_actions, lstm_unit,
            nenvs, step_size, continuous, scope='network', reuse=reuse)
        network_func_vars = util.scope_vars(
            util.absolute_scope_name('network'), trainable_only=True)

        old_policy, old_value, old_dist = network(
            obs_t_input, num_actions, scope='old_network', reuse=reuse)
        old_network_func_vars = util.scope_vars(
            util.absolute_scope_name('old_network'),
            trainable_only=True)

        tmp_policy, tmp_value, tmp_dist = network(
            obs_t_input, num_actions, scope='tmp_network', reuse=reuse)
        tmp_network_func_vars = util.scope_vars(
            util.absolute_scope_name('tmp_network'),
            trainable_only=True)

        # reshape inputs
        advantages = tf.reshape(advantage_t_ph, [-1, 1])
        returns = tf.reshape(return_t_ph, [-1, 1])
        masks = tf.reshape(mask_ph, [-1, 1])

        # clipped surrogate objective
        cur_policy = dist.log_prob(act_t_ph)
        old_policy = old_dist.log_prob(act_t_ph)
        ratio = tf.exp(cur_policy - old_policy)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surrogate = -tf.reduce_mean(

            tf.minimum(ratio * advantage_t_ph, clipped_ratio * advantage_t_ph),

            name='surrogate')

        with tf.variable_scope('loss'):
            # value network loss
            value_loss = tf.reduce_mean(tf.square(value - returns) * masks)

            # entropy penalty for exploration
            entropy = tf.reduce_mean(dist.entropy() * mask_ph)
            penalty = -beta * entropy

            # total loss
            loss = policy_factor * surrogate\
                + value_factor * value_loss\
                + entropy_factor * penalty

        # optimize operations
        optimizer = tf.train.AdamOptimizer(3 * 1e-4)
        optimize_expr = optimizer.minimize(loss, var_list=network_func_vars)

        # update old network operations
        with tf.variable_scope('update_old_network'):
            update_old_expr = []
            sorted_tmp_vars = sorted(
                tmp_network_func_vars, key=lambda v: v.name)
            sorted_old_vars = sorted(
                old_network_func_vars, key=lambda v: v.name)
            for var_tmp, var_old in zip(sorted_tmp_vars, sorted_old_vars):
                update_old_expr.append(var_old.assign(var_tmp))
            update_old_expr = tf.group(*update_old_expr)

        # update tmp network operations
        with tf.variable_scope('update_tmp_network'):
            update_tmp_expr = []
            sorted_vars = sorted(
                network_func_vars, key=lambda v: v.name)
            sorted_tmp_vars = sorted(
                tmp_network_func_vars, key=lambda v: v.name)
            for var, var_tmp in zip(sorted_vars, sorted_tmp_vars):
                update_tmp_expr.append(var_tmp.assign(var))
            update_tmp_expr = tf.group(*update_tmp_expr)

        # action theano-style function
        act = util.function(inputs=[obs_t_input], outputs=[policy, value])

        # train theano-style function
        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, return_t_ph, advantage_t_ph
            ],
            outputs=[loss, value_loss, tf.reduce_mean(ratio)],
            updates=[optimize_expr]
        )

        # update target theano-style function
        update_old = util.function([], [], updates=[update_old_expr])
        backup_current = util.function([], [], updates=[update_tmp_expr])

        return act, train, update_old, backup_current
