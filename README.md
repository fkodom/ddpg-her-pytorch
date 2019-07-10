# DDPG+HER-PyTorch

Implementation of the [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) algorithm using PyTorch.  Utilizes [Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1509.02971.pdf) for off-policy optimization of the RL agent -- hence, DDPG+HER.  Includes (for now) just one pre-trained example agent (FetchReach-v1), as well as a training script for creating new agents.

### FetchReach-v1 Agent

![DDPG+HER FetchReach-v1 Animation](https://raw.githubusercontent.com/fkodom/ddpg-her-pytorch/master/figures/fetch-reach.gif)