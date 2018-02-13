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
    parser.add_argument('--outdir', type=str, default=date)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--nenvs', type=int, default=4)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results' + args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + date)

    env = BatchEnvWrapper(envs=[gym.make(args.env) for _ in range(args.nenvs)])

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    network = make_network([64, 64])

    sess = tf.Session()
    sess.__enter__()

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
