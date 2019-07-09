r"""
replay.py
----------
Implementation of a Replay Memory buffer using the PyTorch library.
"""

import random
from collections import deque
from typing import Tuple, Dict

import gym
import torch
from torch import Tensor
import numpy as np

# Define data type for a batch of training data, and robotics observations
Batch = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
Observation = Dict[str, Tensor]


class EpisodeMemory(object):
	r"""Generic replay hindsight_memory buffer used for off-policy optimization of RL agents."""

	def __init__(self, maxlen: int = int(1e6)):
		r"""
		:param maxlen: Maximum number of transitions stored in the replay buffer
		"""
		self.buffer = deque(maxlen=maxlen)

	def __len__(self) -> int:
		return len(self.buffer)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(maxlen={self.buffer.maxlen})'

	def append(self, state: Observation, action: Tensor, reward: float, done: bool, next_state: Observation) -> None:
		"""Adds a transition to the replay buffer."""
		self.buffer.append((state, action, reward, done, next_state))


class HER(object):
	r"""Hindsight Experience Replay (HER) for learning from sparse rewards."""

	def __init__(self, env: str = 'FetchReach-v1', maxlen: int = int(1e6), p_positive: float = 0.1,
	             p_negative: float = 0.05):
		r"""
		:param maxlen: Maximum number of transitions stored in the replay buffer
		:param p_positive: Probability of re-sampling the goal for each transition
		"""
		self.env = gym.make(env)
		self.env.distance_threshold *= 0.25
		self.buffer = deque(maxlen=maxlen)
		self.p_positive = p_positive
		self.p_negative = p_negative

	def __len__(self) -> int:
		return len(self.buffer)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(maxlen={self.buffer.maxlen})'

	def append_episode(self, episode_memory: EpisodeMemory, p_future: float = 1.0, p_current: float = 0.5) -> None:
		r"""Takes a set of game transitions (stored in episode_memory), and adds them to the HER buffer.  Also,
		adds supplementary transitions using the game history.  Future game states are used (as described in OpenAI
		paper), as well as the current game states (not in original paper).

		:param episode_memory: Replay memory buffer containing transitions from the most recent episode.
		:param p_future: Probability of adding a transition to HER buffer, using a future state as the desired goal.
		:param p_current: Probability of adding a transition to HER buffer, using the current state as the desired goal.
		"""
		for i, (state, action, reward, done, next_state) in enumerate(episode_memory.buffer):
			# Directly transcribe all of the episode buffer into the HER buffer.
			state_original = torch.cat((state['observation'], state['desired_goal']), -1)
			next_state_original = torch.cat((next_state['observation'], next_state['desired_goal']), -1)
			self.buffer.append((state_original, action, reward, done, next_state_original))

			# Future memory replay -- uses future states in the episode in place of the desired goal.
			# (As described in the OpenAI HER paper)
			if not done and torch.rand(1).item() < p_future:
				j = np.random.randint(i + 1, len(episode_memory))
				end_state = episode_memory.buffer[j][0]
				reward_future = self.env.compute_reward(state['achieved_goal'], end_state['achieved_goal'], {})
				state_future = torch.cat((state['observation'], end_state['achieved_goal']), -1)
				next_state_future = torch.cat((next_state['observation'], end_state['achieved_goal']), -1)
				self.buffer.append((state_future, action, reward_future, done, next_state_future))

			# Also use the current state as a possible achieved goal.
			#
			# Not described in the original HER paper, but I found that this accelerated training.  Essentially, adds
			# lots of 'success' actions (with reward=0) to the replay buffer, in addition to the future memory
			# (which likely carry reward=-1).
			if not done and torch.rand(1).item() < p_current:
				state_achieved = torch.cat((state['observation'], next_state['achieved_goal']), -1)
				reward_achieved = self.env.compute_reward(next_state['achieved_goal'], next_state['achieved_goal'], {})
				next_state_achieved = torch.cat((next_state['observation'], next_state['achieved_goal']), -1)
				self.buffer.append((state_achieved, action, reward_achieved, True, next_state_achieved))

		# Clear the episode memory, since these states are now in the primary (HER) buffer.
		episode_memory.buffer.clear()

	def get_batch(self, batch_size: int) -> Batch:
		"""Samples 'batch_size' transitions from the replay buffer."""
		batch_size = min(batch_size, len(self.buffer))
		batch = random.sample(self.buffer, batch_size)

		states = torch.stack([arr[0] for arr in batch], 0).detach()
		actions = torch.stack([arr[1] for arr in batch], 0).detach()
		rewards = torch.FloatTensor([arr[2] for arr in batch])
		dones = torch.FloatTensor([arr[3] for arr in batch])
		next_states = torch.stack([arr[4] for arr in batch], 0).detach()

		return states, actions, rewards, dones, next_states
