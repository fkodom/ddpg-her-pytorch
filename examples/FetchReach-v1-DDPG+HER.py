r"""
FetchReach-v1-DDPG+HER
-----------------
Example script for training a robotic arm to reach to an assigned location using DDPG+HER
"""

import copy

import torch
from torch import Tensor
import torch.nn as nn
from src.ddpg import DDPGActor, DDPGCritic, DDPGAgent


class Actor(DDPGActor):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 4)
        )

    def forward(self, state: Tensor) -> Tensor:
        return torch.tanh(self.layers(state))


class Critic(DDPGCritic):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(17, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1)
        )


class Agent(DDPGAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = Actor()
        self.critic = Critic()
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)


if __name__ == '__main__':
    import os
    from src.callbacks import model_checkpoint, plot_training_statistics

    # ------------------------------ Runtime Parameters ------------------------------
    environment: str = 'FetchReach-v1'     # Name of the Gym environment
    # model: str = ''                         # Path to pre-trained model to load
    model: str = os.path.join('..', 'models', f'fetch-reach-ddpg+her.dict')

    train: bool = False                     # If True, initiates a full training sequence
    target_reward: float = -5               # Target reward value -- training terminates once reached
    num_episodes: int = 10000               # Total number of episodes (games) to play for training
    max_turns: int = 100                    # Maximum number of turns in each episode
    batch_size: int = 128                   # Batch size for training the agent
    actor_lr: float = 1e-4                  # Learning rate for actor optimization
    critic_lr: float = 1e-4                 # Learning rate for critic optimization
    gamma: float = 0.99                     # Rewards discount factor, for time-delayed rewards
    tau: float = 1e-3                       # Amount of soft update to target networks
    exploration_noise: float = 0.1          # Amount of added Gaussian noise to actions to force exploration

    render_every: int = int(9e9)            # Render the game being played at this interval
    frame_rate: int = 20                    # Max frame rate (Hz) for game rendering
    # --------------------------------------------------------------------------------

    ddpg = Agent(environment)
    if model:
        ddpg.load_state_dict(torch.load(model))

    if train:
        ddpg.fit(
            target_reward=target_reward,
            num_episodes=num_episodes,
            max_turns=max_turns,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            exploration_noise=exploration_noise,
            callbacks=[
                model_checkpoint(os.path.join('..', 'models', f'{environment}-ddpg+her.dict')),
                plot_training_statistics(print_every=5)
            ]
        )

    while True:
        ddpg.episode(max_turns=200, noise=0, frame_rate=frame_rate)
