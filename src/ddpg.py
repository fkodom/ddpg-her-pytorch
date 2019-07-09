r"""
ddpg.py
--------
Implementation of the DDPG algorithm for continuous control using the PyTorch machine learning library.

"Continuous control with deep reinforcement learning," Lillicrap et. al.
https://arxiv.org/abs/1509.02971
"""

import time
import copy
from typing import Iterable, Callable, Tuple, Dict

import gym
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal
import imageio

from src.replay import HER, EpisodeMemory
from src.utils import format_observed_state

# Define data type for a robotics observation
Observation = Dict[str, Tensor]


class DDPGCritic(nn.Module):
    r"""Critic neural network for DDPG algorithm -- evaluates the value of actions performed."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)
        )

    def forward(self, state, action) -> Tensor:
        r"""Pushes the environment state through the network, and returns action probabilities.

        :param state: Tensor(s) describing the current environment state
        :param action: Action performed by the RL agent (often taken from hindsight_memory replay)
        :return: Action probabilities (mean + std. dev.) and state value
        """
        x = torch.cat((state, action), dim=-1)
        return self.layers(x)


class DDPGActor(nn.Module):
    r"""Actor neural network for DDPG algorithm -- chooses actions to perform."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        r"""Pushes the environment state through the network, and returns action probabilities.

        :param state: Tensor(s) describing the current environment state
        :return: Action value (before adding exploration noise)
        """
        return 2 * torch.tanh(self.layers(state))


class DDPGAgent(nn.Module):
    r"""Continuous control RL agent, optimized using the Deep Deterministic Policy Gradients (DDPG) algorithm.

    "Continuous control with deep reinforcement learning," Lillicrap et. al.
    https://arxiv.org/abs/1509.02971
    """

    def __init__(self, env: str = 'Pendulum-v0', buffer_size: int = int(1e7)):
        super().__init__()
        self.env = env
        self.episode_memory = EpisodeMemory(maxlen=buffer_size)
        self.hindsight_memory = HER(maxlen=buffer_size)

        self.actor = DDPGActor()
        self.critic = DDPGCritic()
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.rewards = []
        self.avg_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.actor_optimizer = None
        self.critic_optimizer = None

    def forward(self, x):
        r"""Dummy method, so that we implement all required methods of nn.Module"""
        return x

    def get_action(self, state: Observation, noise: float = 0.2) -> Tensor:
        r"""Gets the action probabilities for the current environment state, and uses a weighted random draw to select
        an action to perform.

        :param state: Tensor(s) describing the current environment state
        :param noise: Amount of added Gaussian noise to actions to force exploration
        :return: Action to perform: a random variable (possibly multi-variate)
        """
        state_desired = torch.cat((state['observation'], state['desired_goal']), -1)
        action = self.actor.forward(state_desired)
        exploration_noise = Normal(torch.zeros(action.shape), noise * torch.ones(action.shape))
        action += exploration_noise.sample()

        return action.detach()

    def episode(self, save: bool = False, save_path: str = '', max_turns: int = 1000, noise: float = 0.1,
                frame_rate: float = 9e9, seed: int = None) -> None:
        r"""Plays one game (episode), and visualizes the game environment.

        :param save: If True, saves the episode animation to file at 'save_path'
        :param save_path: File path for (optionally) saving the episode animation
        :param max_turns: Maximum number of turns (or frames) to play in one game
        :param noise: Amount of added Gaussian noise to actions to force exploration
        :param frame_rate: If render = True, controls the frame rate (in frames/sec) of the episode
        :param seed: Seed for the random number generator, for creating repeatable experiments
        """
        # Initialize the game environment
        env = gym.make(self.env)
        if seed is not None:
            env.seed(seed)
        state = format_observed_state(env.reset())

        # Initialize list of episode frames for saving GIF
        frames = []

        env.render(mode='human')
        for turn in range(max_turns):
            # Show the game window, and pause for 1/frame_rate seconds
            time.sleep(1 / (frame_rate + 1e-6))
            if save:
                frames.append(env.render('rgb_array'))
            else:
                env.render(mode='human')

            # Perform the next action, and collect the resulting state, reward, etc.
            action = self.get_action(state, noise=noise)
            state, reward, done, _ = env.step(action.numpy())
            state = format_observed_state(state)

            if done:
                break

        env.close()
        if save:
            imageio.mimsave(save_path, frames)

    def update_target_networks(self, tau: float = 1e-3) -> None:
        r"""Performs a soft update on the target networks.

        :param tau: Fractional amount to update the target networks
        """
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data = target_param.data * (1.0 - tau) + param.data * tau

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data = target_param.data * (1.0 - tau) + param.data * tau

    def step(self, batch_size: int = 128, gamma: float = 0.99, tau: float = 1e-3) -> Tuple[float, float]:
        r"""Draws a random set of transitions from the hindsight_memory replay buffer for training, and then updates the networks
        using those samples.

        :param batch_size: Number of samples to draw for gradient descent training
        :param gamma: Decay rate for delayed action rewards
        :param tau: Fractional amount to update the target networks
        """
        states, actions, rewards, dones, next_states = self.hindsight_memory.get_batch(batch_size)

        next_actions = self.target_actor.forward(next_states).detach()
        next_values = self.target_critic.forward(next_states, next_actions).squeeze()
        next_values = torch.where(dones < 0.999, next_values, rewards)
        truth_values = rewards + gamma * next_values.detach()
        pred_values = self.critic.forward(states, actions).squeeze()

        # Compute loss for critic network
        critic_loss = f.smooth_l1_loss(pred_values, truth_values)
        if len(self.hindsight_memory) > batch_size:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute loss for actor network
        pred_actions = self.actor.forward(states)
        actor_loss = -self.target_critic.forward(states, pred_actions).sum()
        if len(self.hindsight_memory) > batch_size:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update for target networks
            self.update_target_networks(tau=tau)

        return actor_loss.item(), critic_loss.item()

    def fit(self, num_episodes: int = 200,  max_turns: int = 1000,  actor_lr: float = 1e-3,  critic_lr: float = 1e-3,
            batch_size: int = 128,  gamma: float = 0.99,  tau: float = 1e-2,  exploration_noise: float = 0.2,
            target_reward: float = -200,  render_every: int = int(9e9),  frame_rate: float = 40,
            callbacks: Iterable[Callable] = ()) -> None:
        r"""Initiates a complete training sequence for the RL Agent.

        :param num_episodes: Number of episodes (complete games) to play for training
        :param max_turns: Maximum number of turns (frames) to play in each game
        :param actor_lr: Learning rate for optimizing the actor network
        :param critic_lr: Learning rate for optimizing the critic network
        :param batch_size: Number of samples to draw for gradient descent training
        :param gamma: Decay rate for delayed action rewards
        :param tau: Fractional amount to update the target networks
        :param exploration_noise: Amount of added Gaussian noise to actions to force exploration
        :param target_reward: Environment is solved when average reward reaches this value, and training ends.
        :param render_every: Controls how often the game window should be rendered during training
        :param frame_rate: Maximum frame rate for the rendered game window
        :param callbacks: Iterable of functions to perform for each batch (e.g. save, visualize)
        """
        env = gym.make(self.env)
        env.distance_threshold *= 0.25
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        for num_episode in range(1, num_episodes + 1):
            total_reward, avg_actor_loss, avg_critic_loss = 0, 0, 0
            state = format_observed_state(env.reset())

            for turn in range(1, max_turns + 1):
                if num_episode % render_every == 0:
                    env.render()
                    time.sleep(1 / (frame_rate + 1e-4))

                action = self.get_action(state, noise=exploration_noise)
                next_state, reward, done, _ = env.step(action.numpy())
                next_state = format_observed_state(next_state)
                total_reward += reward

                self.episode_memory.append(state, action, reward, done, next_state)
                state = next_state

                # perform optimization
                if len(self.hindsight_memory) > batch_size:
                    actor_loss, critic_loss = self.step(batch_size=batch_size, gamma=gamma, tau=tau)
                    avg_actor_loss += actor_loss / max_turns
                    avg_critic_loss += critic_loss / max_turns
                if done:
                    avg_actor_loss *= max_turns / turn
                    avg_critic_loss *= max_turns / turn
                    break

            self.hindsight_memory.append_episode(self.episode_memory)
            self.actor_losses.append(avg_actor_loss)
            self.critic_losses.append(avg_critic_loss)
            self.rewards.append(total_reward)
            if self.avg_rewards:
                self.avg_rewards.append(0.9 * self.avg_rewards[-1] + 0.1 * total_reward)
            else:
                self.avg_rewards.append(total_reward)

            print('\rEpisode: {} | Reward: {:.2f} | Avg Reward: {:.2f}'.format(
                num_episode, total_reward, self.avg_rewards[-1]), end='')
            for callback in callbacks:
                callback(self)

            if self.avg_rewards[-1] >= target_reward:
                print('\nReached target reward!  Ending training.')
                break
