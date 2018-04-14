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
                actions,
                optimizer,
                nenvs,
                gamma=0.9,
                lstm_unit=256,
                time_horizon=128,
                policy_factor=1.0,
                value_factor=0.5,
                entropy_factor=0.01,
                epsilon=0.2,
                lam=0.95,
                state_shape=[84, 84, 1],
                phi=lambda s: s,
                continuous=False,
                name='ppo'):
        self.actions = actions
        self.gamma = gamma
        self.lam = lam
        self.name = name
        self.nenvs = nenvs
        self.time_horizon = time_horizon
        self.state_shape = state_shape
        self.phi = phi
        self.continuous = continuous

        self._act,\
        self._train,\
        self._update_old,\
        self._backup_current = build_graph.build_train(
            network=network,
            num_actions=num_actions,
            optimizer=optimizer,
            nenvs=nenvs,
            lstm_unit=lstm_unit,
            state_shape=state_shape,
            value_factor=value_factor,
            policy_factor=policy_factor,
            entropy_factor=entropy_factor,
            epsilon=epsilon,
            gamma=gamma,
            reuse=reuse,
            scope=name
        )
        self.initial_state = np.zeros((nenvs, lstm_unit), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_action = None
        self.last_value = None
        self.rollouts = [Rollout() for _ in range(nenvs)]
        self.t = 0

    def act(self, obs, reward, done, training=True):
        obs = list(map(self.phi, obs))
        action, value = self._act(obs, self.rnn_state0, self.rnn_state1)
        if self.continuous:
            action = np.clip(action, -1, 1)
        value = np.reshape(value, [-1])

        if training:
            if len(self.rollouts[i])
            if self.last_obs is not None:
                for i in range(self.nenvs):
                    self.rollouts[i].add(
                        state=self.last_obs[i],
                        action=self.last_actions[i],
                        reward=rewards[i],
                        value=self.last_values[i]
                        terminal=dones[i]
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
        masks = []
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
            masks.append(list(
                map(lamda t: 0.0 if t else 1.0, self.rollouts[i].terminals)))
            # flush histories
            self.rollouts[i].flush()
        obs = np.reshape(obs, [-1, self.obs_dim])
        actions = np.reshape(actions, [-1, self.num_actions])
        values = np.reshape(values, [-1, 1])
        advantages = np.reshape(advantages, [-1, 1])
        loss, value_loss, ratio = self._train(
            obs, actions, values, advantages, masks)
        self._update_old()
        return ratio

    def stop_episode(self, obs, rewards, training):
        if training:
            for i in range(self.nenvs):
                self.rollouts[i].add(
                    state=self.obs[i],
                    action=self.last_actions[i],
                    reward=rewards[i],
                    value=self.last_values[i],
                    terminal=False
                )
            self.train()
        self.last_obs = None
        self.last_actions = None
        self.last_values = None

    def sync_old(self):
        self._update_old()
