import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf

from lightsaber.rl.trainer import AgentInterface
from lightsaber.rl.util import Rollout, compute_v_and_adv


class Agent(object):
    def __init__(self,
                network,
                obs_dim,
                num_actions,
                gamma=0.9,
                lam=0.95,
                nenvs=4,
                horizon=128,
                reuse=None):
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.nenvs = nenvs
        self.horizon = horizon
        self.t = 0
        self.rollouts = [Rollout() for _ in range(nenvs)]

        self._act,\
        self._train,\
        self._update_old,\
        self._backup_current = build_graph.build_train(
            network=network,
            obs_dim=obs_dim,
            num_actions=num_actions,
            gamma=gamma,
            reuse=reuse
        )

        self.last_obs = None
        self.last_actions = None
        self.last_values = None

    def act(self, obs, rewards, training=True):
        actions, values = self._act(np.reshape(obs, [-1, self.obs_dim]))
        actions = np.reshape(np.clip(actions, -2, 2), [-1, self.num_actions])
        values = np.reshape(values, [-1])

        if self.last_obs is not None:
            for i in range(self.nenvs):
                self.rollouts[i].add(
                    state=self.last_obs[i],
                    action=self.last_actions[i],
                    reward=rewards[i],
                    value=self.last_values[i]
                )
            if self.t > 0 and self.t % self.horizon == 0:
                self.train()

        self.last_obs = obs
        self.last_actions = actions
        self.last_values = values
        self.t += 1
        return actions

    def train(self):
        self._backup_current()
        obs = []
        actions = []
        values = []
        advantages = []
        for i in range(self.nenvs):
            obs.append(self.rollouts[i].states)
            actions.append(self.rollouts[i].actions)
            v, adv = compute_v_and_adv(
                rewards=self.rollouts[i].rewards,
                values=self.rollouts[i].values,
                bootstrapped_value=0,
                gamma=self.gamma,
                lam=self.lam
            )
            # standardize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            values.append(v)
            advantages.append(adv)
            # flush histories
            self.rollouts[i].flush()
        obs = np.reshape(obs, [-1, self.obs_dim])
        actions = np.reshape(actions, [-1, self.num_actions])
        values = np.reshape(values, [-1, 1])
        advantages = np.reshape(advantages, [-1, 1])
        loss, value_loss, ratio = self._train(obs, actions, values, advantages)
        self._update_old()
        return ratio

    def stop_episode(self, obs, rewards, training):
        if training:
            for i in range(self.nenvs):
                self.rollouts[i].add(
                    state=self.obs[i],
                    action=self.last_actions[i],
                    reward=rewards[i],
                    value=self.last_values[i]
                )
        self.last_obs = None
        self.last_actions = None
        self.last_values = None

    def sync_old(self):
        self._update_old()
