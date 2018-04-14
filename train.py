import argparse
import cv2
import gym
import copy
import os
import sys
import random
import numpy as np
import tensorflow as tf

from datetime import datetime
from lightsaber.tensorflow.util import initialize
from lightsaber.rl.replay_buffer import ReplayBuffer
from lightsaber.tensorflow.log import TfBoardLogger
from lightsaber.rl.trainer import BatchTrainer
from lightsaber.rl.env_wrapper import BatchEnvWrapper
from network import make_network
from agent import Agent


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + date)

    env_name = args.env
    tmp_env = gym.make(env_name)
    if len(tmp_env.observation_space.shape) == 1:
        observation_space = tmp_env.observation_space
        constants = box_constants
        actions = range(tmp_env.action_space.n)
        state_shape = [observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda s: s
        # (window_size, dim) -> (dim, window_size)
        phi = lambda s: np.transpose(s, [1, 0])
    else:
        constants = atari_constants
        actions = get_action_space(env_name)
        state_shape = constants.STATE_SHAPE + [constants.STATE_WINDOW]
        def state_preprocess(state):
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, tuple(constants.STATE_SHAPE))
            state = np.array(state, dtype=np.float32)
            return state / 255.0
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda s: np.transpose(s, [1, 2, 0])

    # save settings
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    env = BatchEnvWrapper(envs=[gym.make(args.env) for _ in range(args.nenvs)])

    network = make_network(constants.CONVS, constants.FCS, lstm=constants.LSTM)

    agent = Agent(network, obs_dim, n_actions)

    initialize()
    agent.sync_old()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(args.logdir, sess.graph)
    logger = TfBoardLogger(train_writer)
    logger.register('reward', tf.int32)
    end_episode = lambda r, s, e: logger.plot('reward', r, s)

    trainer = BatchTrainer(
        env=env,
        agent=agent,
        state_shape=[obs_dim],
        training=args.demo,
        render=args.render,
        end_episode=end_episode
    )
    trainer.start()

if __name__ == '__main__':
    main()
