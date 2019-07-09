"""
callbacks.py
-------------
Callbacks for use during model training
"""

import time
from typing import AnyStr, Callable

import torch
import matplotlib.pyplot as plt


def model_checkpoint(save_path: AnyStr) -> Callable:
    """Defines a callback for saving PyTorch models during training.  The callback function will always accept `self`
    as the first argument.  We include `*args` positional arguments for added flexibility, so that a network could
    pass multiple arguments (e.g. training/validation loss, epoch number, etc.) without breaking it.

    :param save_path:  Absolute path to the model's save file
    :return:  Callback function for model saving
    """
    # noinspection PyUnusedLocal
    def callback(model: torch.nn.Module, *args, **kwargs):
        torch.save(model.state_dict(), save_path)

    return callback


def plot_training_statistics(print_every: int = 25):
    r"""Plots the training statistics for the RL agent in an interactive window.  Training statistics include:
            * total rewards per episode3
            * actor loss
            * critic loss

    :param print_every: Number of episodes played before the plot is updated
    """
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    for a in ax:
        a.plot([0])
    plt.pause(1e-3)

    # noinspection PyUnusedLocal
    def callback(model: torch.nn.Module, *args, **kwargs):
        if len(model.rewards) % print_every != 0:
            return

        for a in ax:
            a.clear()
            a.set_xlabel('Episode')

        ax[0].plot(model.rewards)
        ax[0].plot(model.avg_rewards)
        ax[0].set_title('Rewards')
        ax[1].plot(model.actor_losses)
        ax[1].set_title('Actor Loss')
        ax[2].plot(model.critic_losses)
        ax[2].set_title('Critic Loss')

        plt.tight_layout()
        fig.canvas.flush_events()
        time.sleep(1e-3)

    return callback
